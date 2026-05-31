"""Workflow tool — let the model execute workflow scripts from conversation.

This is the tool the model uses when it wants to run a multi-agent workflow
instead of brute-forcing everything in one turn. Equivalent to Claude Code's
built-in workflow execution.

The model calls this when:
- A task decomposes into 3+ independent subtasks feeding into synthesis.
- The user asks for a workflow, pipeline, or deep analysis.
- The model determines that parallel agents would be more efficient.

Usage from conversation:
  workflow_run(script="workflows/security_audit.py")
  workflow_run(script="workflows/security_audit.py", resume=True)
  workflow_run(script="workflows/security_audit.py", model="gpt-4o")
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.registry import registry, tool_error, tool_result
from hermes_constants import get_hermes_home


# ---------------------------------------------------------------------------
# Workflow discovery helpers
# ---------------------------------------------------------------------------

_DEFAULT_DIRS = [
    "workflows",
    str(get_hermes_home() / "workflows"),
]


def _find_workflow(
    workflow_name: str, search_dirs: Optional[List[str]] = None
) -> Optional[Path]:
    """Find a workflow script by name or path."""
    # Direct path
    p = Path(workflow_name)
    if p.is_file():
        return p

    # Search directories
    dirs = search_dirs or _DEFAULT_DIRS
    for directory in dirs:
        candidate = Path(directory) / f"{workflow_name}.py"
        if candidate.is_file():
            return candidate
        candidate = Path(directory) / f"{workflow_name}"
        if candidate.is_file():
            return candidate

    return None


def _load_workflow_module(script_path: Path) -> Any:
    """Import a workflow script and return the module."""
    spec = importlib.util.spec_from_file_location(
        f"workflow_{script_path.stem}", script_path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load workflow: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module.__name__] = module
    sys.path.insert(0, str(script_path.parent))

    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module.__name__, None)
        sys.path.remove(str(script_path.parent))
        raise RuntimeError(f"Error loading workflow: {exc}") from exc

    return module


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------


def _handle_workflow_run(
    script: str,
    context: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    resume: bool = False,
    force: bool = False,
    search_dirs: Optional[List[str]] = None,
    parent_agent=None,
) -> str:
    """Execute a workflow script and return results."""
    # Resolve workflow file
    wf_path = _find_workflow(script, search_dirs)
    if wf_path is None:
        return tool_error(f"Workflow not found: {script}")

    # Check checkpoint status
    # Use the same stem-based hash as WorkflowRunner so both sides read
    # the same checkpoint file.
    wf_hash = hashlib.sha256(wf_path.stem.encode()).hexdigest()[:16]
    checkpoint_dir = get_hermes_home() / "workflow_checkpoints"
    checkpoint_path = checkpoint_dir / f"{wf_hash}.json"

    if checkpoint_path.exists() and not force:
        try:
            cp = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            completed = cp.get("completed_phases", [])
            if completed and not resume:
                return tool_result({
                    "status": "checkpointed",
                    "workflow": wf_path.name,
                    "completed_phases": completed,
                    "message": f"Workflow has {len(completed)} completed phases. "
                    f"Use resume=True to continue or force=True to restart.",
                })
        except Exception:
            pass

    # Import the workflow script
    try:
        module = _load_workflow_module(wf_path)
    except Exception as exc:
        return tool_error(str(exc))
    finally:
        sys.path.remove(str(wf_path.parent))

    # Find the workflow() function
    workflow_fn = getattr(module, "workflow", None)
    if workflow_fn is None or not callable(workflow_fn):
        return tool_error(f"Workflow must define a workflow() function: {wf_path}")

    # Run it
    try:
        from hermes_workflow import WorkflowRunner

        with WorkflowRunner(
            name=wf_path.stem,
            model=model,
            provider=provider,
            display=False,  # Agent already manages its own display (spinner/
            # activity feed).  A second Rich Live would fight
            # for terminal control and produce ANSI garbage.
        ) as runner:
            runner.run_script(lambda _: workflow_fn())

        # Read checkpoint for status
        checkpoint_data = None
        if checkpoint_path.exists():
            try:
                checkpoint_data = json.loads(
                    checkpoint_path.read_text(encoding="utf-8")
                )
            except Exception:
                pass

        return tool_result({
            "status": "completed",
            "workflow": wf_path.name,
            "file": str(wf_path),
            "completed_phases": checkpoint_data.get("completed_phases", [])
            if checkpoint_data
            else [],
            "message": f"Workflow '{wf_path.stem}' completed successfully.",
        })

    except Exception as exc:
        return tool_error(f"Workflow failed: {exc}")


# ---------------------------------------------------------------------------
# Discover handler
# ---------------------------------------------------------------------------


def _handle_workflow_list(
    search_dirs: Optional[List[str]] = None,
    parent_agent=None,
) -> str:
    """List available workflows."""
    dirs = search_dirs or _DEFAULT_DIRS
    workflows = []

    for directory in dirs:
        d = Path(directory)
        if not d.is_dir():
            continue
        for py in sorted(d.glob("*.py")):
            if py.name.startswith("_"):
                continue

            desc = ""
            try:
                source = py.read_text(encoding="utf-8")
                import ast as _ast

                tree = _ast.parse(source)
                doc = _ast.get_docstring(tree)
                if doc:
                    desc = doc.split("\n")[0]
            except Exception:
                pass

            workflows.append({
                "name": py.stem,
                "file": str(py),
                "directory": str(d),
                "description": desc,
            })

    # Deduplicate by name
    seen = {}
    unique = []
    for wf in workflows:
        if wf["name"] not in seen:
            seen[wf["name"]] = True
            unique.append(wf)

    return tool_result({
        "workflows": unique,
        "count": len(unique),
    })


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

WORKFLOW_RUN_SCHEMA = {
    "name": "workflow_run",
    "toolsets": ["workflow"],
    "description": (
        "Execute a multi-agent workflow script. Workflows decompose large tasks "
        "into independent phases with parallel agents and structured data flow. "
        "Use this for tasks that have 3+ independent subtasks (audits, research, "
        "migrations) feeding into a synthesis step. The workflow script is a "
        "Python file defining a workflow() function using phase(), run(), and "
        "run_all() primitives."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "script": {
                "type": "string",
                "description": (
                    "Workflow name (searches workflows/ and ~/.hermes/workflows/) "
                    "or path to a .py file. Example: 'security_audit' or 'workflows/audit.py'."
                ),
            },
            "context": {
                "type": "string",
                "description": "Optional additional context passed to the workflow.",
            },
            "model": {
                "type": "string",
                "description": "Override model for child agents in this workflow.",
            },
            "provider": {
                "type": "string",
                "description": "Override provider for child agents in this workflow.",
            },
            "resume": {
                "type": "boolean",
                "description": "Resume from the last completed phase (checkpoint).",
            },
            "force": {
                "type": "boolean",
                "description": "Ignore existing checkpoint and re-run from scratch.",
            },
            "search_dirs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional directories to search for workflow scripts.",
            },
        },
        "required": ["script"],
    },
}

WORKFLOW_LIST_SCHEMA = {
    "name": "workflow_list",
    "toolsets": ["workflow"],
    "description": (
        "List available workflow scripts. Returns workflow names, paths, and "
        "descriptions from workflows/ and ~/.hermes/workflows/ directories."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "search_dirs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional directories to search for workflow scripts.",
            },
        },
        "required": [],
    },
}


def _workflow_run_handler(args: Dict[str, Any], **kw) -> str:
    return _handle_workflow_run(
        script=args["script"],
        context=args.get("context"),
        model=args.get("model"),
        provider=args.get("provider"),
        resume=args.get("resume", False),
        force=args.get("force", False),
        search_dirs=args.get("search_dirs"),
        parent_agent=kw.get("parent_agent"),
    )


def _workflow_list_handler(args: Dict[str, Any], **kw) -> str:
    return _handle_workflow_list(
        search_dirs=args.get("search_dirs"),
        parent_agent=kw.get("parent_agent"),
    )


registry.register(
    name="workflow_run",
    toolset="workflow",
    schema=WORKFLOW_RUN_SCHEMA,
    handler=_workflow_run_handler,
    emoji="⚡",
)

registry.register(
    name="workflow_list",
    toolset="workflow",
    schema=WORKFLOW_LIST_SCHEMA,
    handler=_workflow_list_handler,
    emoji="📋",
)
