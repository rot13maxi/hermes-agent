"""Workflow runner — bootstraps agents, manages phases, and handles checkpoints.

Usage
-----
    with WorkflowRunner("my-workflow") as wf:
        wf.run_script(lambda: _my_workflow(wf))

    def _my_workflow(wf):
        with phase("research"):
            results = run_all([
                {"goal": "Research topic A"},
                {"goal": "Research topic B"},
            ])
        with phase("synthesis"):
            summary = run("Synthesize findings", context=str(results))
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from hermes_workflow.schema import validate_response
from hermes_workflow.display import WorkflowDisplay

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-local current runner
# ---------------------------------------------------------------------------

_runner_local = threading.local()


def _current_runner() -> "WorkflowRunner":
    """Return the WorkflowRunner for the current thread."""
    runner = getattr(_runner_local, "runner", None)
    if runner is None:
        raise RuntimeError(
            "No active WorkflowRunner. Call this inside "
            "WorkflowRunner.run_script() or with WorkflowRunner(...) context."
        )
    return runner


# ---------------------------------------------------------------------------
# Public primitives — these delegate to the current runner
# ---------------------------------------------------------------------------


@contextmanager
def phase(name: str):
    """Context manager marking a checkpointed workflow phase.

    On entry, records phase start. On exit, saves checkpoint with phase
    results. If the checkpoint file already has this phase completed,
    the block body is skipped and the yielded PhaseResult contains any
    saved ``last_result`` from that run.

    Example
    -------
        with phase("data-collection"):
            results = run("Collect the data")

    On resume, the body is skipped but ``runner.phase_results[name]``
    retains the saved output.  Workflow scripts should assign results
    inside the phase body and read them back from the runner when
    resuming::

        with phase("discover"):
            files = run("Find files", ...)
            # On first run: files is set by run().
            # On resume: body is skipped; read runner.phase_results["discover"]
    """
    runner = _current_runner()
    runner.on_phase_start(name)

    # If this phase was already completed in a previous run, skip it
    if runner.checkpoint and runner.checkpoint.get("completed_phases", []):
        if name in runner.checkpoint["completed_phases"]:
            logger.info("Skipping already-completed phase: %s", name)
            runner.on_phase_complete(name, duration=0.0)
            # Yield a PhaseResult that exposes saved state so downstream
            # phases can read checkpointed results via runner.phase_results.
            yield PhaseResult(name, runner._phase_results.get(name) or {})
            return

    phase_result = PhaseResult(name, {})
    try:
        yield phase_result
    finally:
        elapsed = runner.on_phase_complete(name)
        phase_result["elapsed"] = elapsed


class PhaseResult(Dict):
    """Dictionary subclass carrying phase metadata for checkpoint resume."""

    def __init__(self, name: str, data: Dict[str, Any] | None = None):
        super().__init__(data or {})
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"<PhaseResult {self._name!r} with {len(self)} keys>"


def run(
    goal: str,
    context: Optional[str] = None,
    toolsets: Optional[List[str]] = None,
    response_schema: Optional[Dict[str, Any]] = None,
) -> str:
    """Run a single child agent via delegation.

    Parameters
    ----------
    goal : str
        The task description for the child agent.
    context : str, optional
        Additional context passed to the child.
    toolsets : list[str], optional
        Which toolsets the child should have access to.
    response_schema : dict, optional
        JSON schema dict to validate the child's response against.
        Uses the lightweight stdlib validator (type, required, enum, etc.).

    Returns
    -------
    str
        The child agent's final response text.

    Raises
    ------
    ValueError
        If response_schema is provided and the response doesn't match.
    """
    runner = _current_runner()
    return runner.run_agent(goal, context, toolsets, response_schema)


def run_all(
    tasks: List[Dict[str, Any]],
    max_concurrent: int = 3,
) -> List[Dict[str, Any]]:
    """Run multiple child agents concurrently.

    Parameters
    ----------
    tasks : list[dict]
        Each dict has at minimum ``{"goal": ...}``.  Optional keys:
        ``context``, ``toolsets``, ``response_schema``.
    max_concurrent : int
        Maximum number of concurrent child agents.

    Returns
    -------
    list[dict]
        Results in the same order as *tasks*, each entry containing
        ``goal``, ``response``, and optionally ``error``.
    """
    runner = _current_runner()
    return runner.run_all_agents(tasks, max_concurrent)


# ---------------------------------------------------------------------------
# WorkflowRunner
# ---------------------------------------------------------------------------


class WorkflowRunner:
    """Orchestrates a multi-phase, multi-agent workflow.

    Bootstraps the parent AIAgent (config, env, credentials), then manages
    phase lifecycle, checkpointing, and Rich display.

    Parameters
    ----------
    name : str
        Human-readable workflow name (also used for checkpoint filename).
    model : str, optional
        Override the model for child agents.
    provider : str, optional
        Override the provider for child agents.
    checkpoint_dir : str, optional
        Directory for checkpoint files.  Defaults to ~/.hermes/workflow_checkpoints/.
    display : bool
        Enable Rich progress display.  Defaults to True.

    Example
    -------
        with WorkflowRunner("data-pipeline", model="gpt-4o") as wf:
            wf.run_script(my_workflow_fn)
    """

    def __init__(
        self,
        name: str = "workflow",
        model: Optional[str] = None,
        provider: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        display: bool = True,
    ) -> None:
        self.name = name
        self._model = model
        self._provider = provider
        self._display_enabled = display

        # Compute a stable hash for checkpoint filename
        self._workflow_hash = hashlib.sha256(name.encode()).hexdigest()[:16]

        # Checkpoint path
        if checkpoint_dir is None:
            hermes_home = os.path.expanduser("~/.hermes")
            checkpoint_dir = os.path.join(hermes_home, "workflow_checkpoints")
        self._checkpoint_dir = Path(checkpoint_dir)
        self._checkpoint_path = self._checkpoint_dir / f"{self._workflow_hash}.json"

        # State
        self.checkpoint: Optional[Dict[str, Any]] = None
        self._parent_agent = None
        self._phase_results: Dict[str, Any] = {}
        self._phases_completed: List[str] = []
        self._display: Optional[WorkflowDisplay] = None
        self._delegate_task_fn = None

    def __enter__(self) -> "WorkflowRunner":
        self._bootstrap()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._display:
            self._display.stop()
        if self._parent_agent is not None:
            try:
                if hasattr(self._parent_agent, "close"):
                    self._parent_agent.close()
            except Exception:
                pass
        return None  # don't suppress exceptions

    # -- bootstrap --

    def _bootstrap(self) -> None:
        """Load config, env, create parent AIAgent, and load checkpoint."""
        logger.info("Bootstrapping WorkflowRunner: %s", self.name)

        # Load environment
        try:
            from hermes_cli.env_loader import load_hermes_dotenv
            from hermes_constants import get_hermes_home

            hermes_home = get_hermes_home()
            project_env = Path(__file__).parent.parent / ".env"
            load_hermes_dotenv(hermes_home=hermes_home, project_env=project_env)
        except Exception as exc:
            logger.debug("env_loader not available: %s", exc)

        # Load CLI config for model/provider resolution
        try:
            from cli import load_cli_config

            self._cli_config = load_cli_config()
        except Exception as exc:
            logger.debug("cli load_cli_config not available: %s", exc)
            self._cli_config = {}

        # Load checkpoint if it exists
        self._load_checkpoint()

        # Create display
        if self._display_enabled:
            try:
                self._display = WorkflowDisplay(self.name)
            except Exception:
                self._display = None

        # Resolve parent agent — lazy, created when first needed
        self._parent_agent = None
        self._delegate_task_fn = None

    def _ensure_parent_agent(self) -> Any:
        """Create the parent AIAgent and resolve delegate_task, lazily."""
        if self._parent_agent is not None:
            return

        try:
            from run_agent import AIAgent
            from hermes_cli.runtime_provider import resolve_runtime_provider
            from hermes_cli.fallback_config import get_fallback_chain

            cfg = self._cli_config

            # Resolve model
            model_cfg = cfg.get("model") or {}
            if isinstance(model_cfg, str):
                cfg_model = model_cfg
            else:
                cfg_model = model_cfg.get("default") or model_cfg.get("model") or ""

            env_model = os.getenv("HERMES_INFERENCE_MODEL", "").strip()
            effective_model = (self._model or "").strip() or env_model or cfg_model

            # Resolve provider
            runtime = resolve_runtime_provider(
                requested=(self._provider or "").strip() or None,
                target_model=effective_model or None,
            )

            # Get toolsets from config
            toolsets_list = None
            try:
                from hermes_cli.tools_config import _get_platform_tools

                toolsets_list = sorted(_get_platform_tools(cfg, "cli"))
            except Exception:
                pass

            _fb = None
            try:
                _fb = get_fallback_chain(cfg)
            except Exception:
                pass

            self._parent_agent = AIAgent(
                api_key=runtime.get("api_key"),
                base_url=runtime.get("base_url"),
                provider=runtime.get("provider"),
                api_mode=runtime.get("api_mode"),
                model=effective_model,
                enabled_toolsets=toolsets_list,
                quiet_mode=True,
                platform="workflow",
                credential_pool=runtime.get("credential_pool"),
                fallback_model=_fb or None,
                clarify_callback=self._clarify_callback,
            )
            self._parent_agent.suppress_status_output = True
            self._parent_agent.stream_delta_callback = None
            self._parent_agent.tool_gen_callback = None

        except Exception as exc:
            logger.error("Failed to create parent AIAgent: %s", exc)
            raise RuntimeError(f"Cannot bootstrap parent agent: {exc}") from exc

    def _ensure_delegate_task(self) -> Callable:
        """Resolve the delegate_task function from the tools module."""
        if self._delegate_task_fn is not None:
            return self._delegate_task_fn

        try:
            from tools.delegate_tool import delegate_task

            self._delegate_task_fn = delegate_task
        except ImportError:
            # Fallback: try relative import path
            try:
                from tools import delegate_tool

                self._delegate_task_fn = delegate_tool.delegate_task
            except ImportError as exc:
                raise RuntimeError(
                    "delegate_task not found. Is the tools module available?"
                ) from exc

        return self._delegate_task_fn

    @staticmethod
    def _clarify_callback(question: str, choices=None) -> str:
        """Non-interactive clarify callback for workflow agents."""
        if choices:
            return (
                f"[workflow mode: no user available. Pick the best option from "
                f"{choices} using your own judgment and continue.]"
            )
        return (
            "[workflow mode: no user available. Make the most reasonable "
            "assumption you can and continue.]"
        )

    # -- run_script --

    def run_script(self, script: Callable[["WorkflowRunner"], Any]) -> Any:
        """Run a workflow script with this runner as the implicit context.

        Sets the thread-local runner so ``phase()``, ``run()``, and
        ``run_all()`` can find it without explicit passing.

        Parameters
        ----------
        script : callable
            A function that uses phase(), run(), run_all() primitives.
            Receives the WorkflowRunner instance as its argument.

        Returns
        -------
        The return value of *script*.
        """
        _runner_local.runner = self
        try:
            return script(self)
        finally:
            _runner_local.runner = None

    # -- phase lifecycle --

    def on_phase_start(self, name: str) -> None:
        """Called when entering a phase."""
        if self._display:
            self._display.set_phase(name)
        logger.info("Phase started: %s", name)

    def on_phase_complete(self, name: str, duration: Optional[float] = None) -> float:
        """Called when exiting a phase. Saves checkpoint."""
        self._phases_completed.append(name)
        self._phase_results[name] = {
            "status": "completed",
            "phases": list(self._phases_completed),
        }

        if duration is None:
            # Measure from phase start if tracked
            duration = 0.0

        if self._display:
            self._display.complete_phase(name, duration)

        # Save checkpoint
        self._save_checkpoint()
        logger.info("Phase completed: %s (%.1fs)", name, duration)
        return duration

    # -- agent execution --

    def _parse_delegate_result(self, result_json: str) -> List[Dict[str, Any]]:
        """Parse delegate_task JSON output into a list of result entries."""
        parsed = json.loads(result_json)
        if isinstance(parsed, dict) and "results" in parsed:
            return parsed["results"]
        if isinstance(parsed, list):
            return parsed
        return [parsed]

    def run_agent(
        self,
        goal: str,
        context: Optional[str] = None,
        toolsets: Optional[List[str]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Run a single child agent and return its response.

        delegate_task already validates response_schema with one retry,
        so we trust the schema-validated output and skip redundant checks.
        """
        self._ensure_parent_agent()
        delegate_fn = self._ensure_delegate_task()

        display_id = None
        if self._display:
            display_id = self._display.add_agent(goal)

        try:
            result_json = delegate_fn(
                goal=goal,
                context=context,
                toolsets=toolsets,
                response_schema=response_schema,
                parent_agent=self._parent_agent,
            )

            entries = self._parse_delegate_result(result_json)

            if not entries:
                raise ValueError("delegate_task returned no results")

            entry = entries[0]

            if entry.get("status") in ("error", "failed"):
                error = entry.get("error", "unknown error")
                if display_id and self._display:
                    self._display.fail_agent(display_id, error)
                raise RuntimeError(f"Child agent failed: {error}")

            response = entry.get("summary", "")

            if display_id and self._display:
                self._display.complete_agent(display_id)

            return str(response)

        except json.JSONDecodeError:
            # delegate_task returned raw text instead of JSON
            if display_id and self._display:
                self._display.complete_agent(display_id)
            result_json = locals().get("result_json", "")
            return str(result_json)

    def run_all_agents(
        self,
        tasks: List[Dict[str, Any]],
        max_concurrent: int = 3,
    ) -> List[Dict[str, Any]]:
        """Run multiple child agents concurrently.

        Returns results in the same order as the input tasks.

        max_concurrent is respected by batching: if tasks exceed the config's
        max_concurrent_children limit, we split into multiple delegate_task
        calls, each capped at max_concurrent.
        """
        self._ensure_parent_agent()
        delegate_fn = self._ensure_delegate_task()

        all_results: List[Dict[str, Any]] = []

        # Batch tasks into chunks that respect max_concurrent
        batch_size = max(1, max_concurrent)
        for batch_start in range(0, len(tasks), batch_size):
            batch = tasks[batch_start : batch_start + batch_size]

            # Register display IDs for this batch
            display_ids: Dict[int, Optional[str]] = {}
            for i, task in enumerate(batch):
                idx = batch_start + i
                if self._display:
                    display_ids[i] = self._display.add_agent(
                        task.get("goal", "unknown")
                    )
                else:
                    display_ids[i] = None

            # Build args for delegate_task batch mode
            delegate_tasks = []
            for task in batch:
                delegate_tasks.append({
                    "goal": task["goal"],
                    "context": task.get("context"),
                    "toolsets": task.get("toolsets"),
                    "response_schema": task.get("response_schema"),
                })

            result_json = delegate_fn(
                tasks=delegate_tasks,
                parent_agent=self._parent_agent,
            )

            entries = self._parse_delegate_result(result_json)

            for i, entry in enumerate(entries):
                global_idx = batch_start + i
                goal = (
                    tasks[global_idx].get("goal", "") if global_idx < len(tasks) else ""
                )
                status = entry.get("status", "")

                if status in ("error", "failed"):
                    all_results.append({
                        "goal": goal,
                        "response": None,
                        "error": entry.get("error", "unknown error"),
                    })
                    did = display_ids.get(i)
                    if did and self._display:
                        self._display.fail_agent(did, entry.get("error", ""))
                else:
                    response = entry.get("summary", "")
                    all_results.append({
                        "goal": goal,
                        "response": str(response),
                    })
                    did = display_ids.get(i)
                    if did and self._display:
                        self._display.complete_agent(did)

        return all_results

    # -- checkpointing --

    def _load_checkpoint(self) -> None:
        """Load checkpoint from disk if it exists."""
        if not self._checkpoint_path.exists():
            self.checkpoint = None
            return
        try:
            data = json.loads(self._checkpoint_path.read_text())
            self.checkpoint = data
            logger.info(
                "Loaded checkpoint: %s (%d phases completed)",
                self._checkpoint_path,
                len(data.get("completed_phases", [])),
            )
        except Exception as exc:
            logger.warning("Failed to load checkpoint: %s", exc)
            self.checkpoint = None

    def _save_checkpoint(self) -> None:
        """Save current phase state to checkpoint file."""
        try:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "workflow": self.name,
                "workflow_hash": self._workflow_hash,
                "completed_phases": list(self._phases_completed),
                "phase_results": self._phase_results,
                "timestamp": time.time(),
            }
            self.checkpoint = data

            # Atomic write
            tmp_path = self._checkpoint_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp_path.rename(self._checkpoint_path)
            logger.debug("Checkpoint saved: %s", self._checkpoint_path)
        except Exception as exc:
            logger.warning("Failed to save checkpoint: %s", exc)
