"""Rich-based progress display for workflow phases and agents.

Provides a live-updating panel that shows:
- Workflow title and current phase
- Phase progress (completed / total with timing)
- Agent tasks with status indicators and elapsed time

Uses Rich's ``Live`` for non-destructive terminal rendering that plays
nicely with agent stderr output.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.spinner import Spinner
    from rich import box

    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


class _NoopDisplay:
    """Fallback when Rich is not installed — all methods are no-ops."""

    def __init__(self, *a: Any, **k: Any) -> None: ...
    def set_workflow(self, *a: Any, **k: Any) -> None: ...
    def set_phase(self, *a: Any, **k: Any) -> None: ...
    def add_phase(self, *a: Any, **k: Any) -> None: ...
    def complete_phase(self, *a: Any, **k: Any) -> None: ...
    def add_agent(self, *a: Any, **k: Any) -> None: ...
    def update_agent(self, *a: Any, **k: Any) -> None: ...
    def complete_agent(self, *a: Any, **k: Any) -> None: ...
    def fail_agent(self, *a: Any, **k: Any) -> None: ...
    def update_footer(self, *a: Any, **k: Any) -> None: ...
    def render(self) -> str:
        return ""

    def stop(self) -> None: ...


class WorkflowDisplay:
    """Rich display for a workflow's phases and agent tasks.

    Parameters
    ----------
    workflow_name : str
        Display name shown at the top of the panel.
    console : Console, optional
        Rich console to render to.  Defaults to stderr so it doesn't
        interfere with agent stdout output.
    """

    def __init__(
        self,
        workflow_name: str = "workflow",
        console: Optional["Console"] = None,
    ) -> None:
        self._workflow_name = workflow_name
        self._phases: Dict[str, Dict[str, Any]] = {}
        self._agents: Dict[str, Dict[str, Any]] = {}
        self._current_phase: Optional[str] = None
        self._agent_counter = 0
        self._footer: Optional[str] = None

        self._live: Any = None
        self._started = False

        # Only use Rich Live if stderr is a real TTY.
        # When run from a tool call inside the TUI, stderr is a PTY pipe
        # and raw ANSI cursor codes from Live render as literal garbage.
        self._use_live = _RICH_AVAILABLE and sys.stderr.isatty()

        if self._use_live:
            # _RICH_AVAILABLE guarantees Console is defined
            self._console = console or Console(stderr=True, force_terminal=True)  # type: ignore[operator]
        else:
            self._console = None

    def _start_live(self) -> None:
        if self._started or not self._use_live:
            return
        self._live = Live(
            self._build_renderable(),
            console=self._console,
            refresh_per_second=2,
            transient=False,
        )
        self._live.start()
        self._started = True

    def set_workflow(self, name: str) -> None:
        self._workflow_name = name

    def set_phase(self, name: str) -> None:
        self._current_phase = name
        self._phases[name] = {"start": time.monotonic(), "end": None}
        self._start_live()

    def add_phase(self, name: str) -> None:
        if name not in self._phases:
            self._phases[name] = {"start": time.monotonic(), "end": None}

    def complete_phase(self, name: str, duration: Optional[float] = None) -> None:
        if name in self._phases:
            self._phases[name]["end"] = time.monotonic()
        if name == self._current_phase:
            self._current_phase = None

    def add_agent(self, goal: str, phase: Optional[str] = None) -> str:
        self._agent_counter += 1
        agent_id = f"agent-{self._agent_counter}"
        self._agents[agent_id] = {
            "id": agent_id,
            "goal": goal[:80],
            "phase": phase or self._current_phase or "",
            "status": "running",
            "start": time.monotonic(),
            "end": None,
            "error": None,
        }
        self._start_live()
        return agent_id

    def update_agent(
        self, agent_id: str, status: Optional[str] = None, error: Optional[str] = None
    ) -> None:
        if agent_id in self._agents:
            if status:
                self._agents[agent_id]["status"] = status
            if error:
                self._agents[agent_id]["error"] = error

    def complete_agent(self, agent_id: str) -> None:
        if agent_id in self._agents:
            self._agents[agent_id]["status"] = "done"
            self._agents[agent_id]["end"] = time.monotonic()

    def fail_agent(self, agent_id: str, error: str) -> None:
        if agent_id in self._agents:
            self._agents[agent_id]["status"] = "error"
            self._agents[agent_id]["end"] = time.monotonic()
            self._agents[agent_id]["error"] = error

    def update_footer(self, text: str) -> None:
        self._footer = text

    def stop(self) -> None:
        if self._live and self._started:
            self._live.stop()
            self._started = False

    # -- rendering --

    def _elapsed(self, t: float) -> str:
        s = time.monotonic() - t
        m, r = divmod(s, 60)
        return f"{int(m):02d}:{int(r):02d}"

    def _build_renderable(self) -> "Panel":
        # Agent table
        table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))  # type: ignore[name-defined]
        table.add_column("Status", width=4)
        table.add_column("Task", overflow="fold")

        for agent in self._agents.values():
            status = agent["status"]
            if status == "running":
                icon = "[yellow]\u27fb[/yellow]"
            elif status == "done":
                icon = "[green]\u2713[/green]"
            elif status == "error":
                icon = "[red]\u2717[/red]"
            else:
                icon = "[dim]\u00b7[/dim]"

            goal = agent["goal"]
            if status == "running":
                goal += f" [dim]({self._elapsed(agent['start'])})[/dim]"
            elif status == "error" and agent.get("error"):
                goal += f" [red]{agent['error'][:40]}[/red]"

            table.add_row(icon, goal)

        # Build subtitle using Text.assemble() so markup is properly
        # interpreted (passing raw markup strings to Text() renders them
        # literally).
        parts: list[tuple[str, str]] = [
            ("", "\u26a1 " + self._workflow_name),
        ]
        if self._current_phase:
            parts.append(("bold yellow", self._current_phase))
            parts.append((" ", ""))
        completed = sum(1 for p in self._phases.values() if p["end"] is not None)
        total = len(self._phases)
        parts.append(("", f"phase {completed}/{total}"))
        if self._footer:
            parts.append(("", " " + self._footer))

        subtitle = Text.assemble(*parts) if _RICH_AVAILABLE else ""  # type: ignore[operator]

        return Panel(  # type: ignore[name-defined]
            table,
            subtitle=subtitle,
            border_style="blue",
            box=box.ROUNDED,  # type: ignore[arg-type]
            padding=(0, 1),
        )

    def render(self) -> str:
        """Return the current display as a string (for logging / non-live use)."""
        if not _RICH_AVAILABLE:
            return f"[workflow: {self._workflow_name}] phases: {len(self._phases)}  agents: {len(self._agents)}"
        from rich.console import Console as _C
        from io import StringIO

        buf = StringIO()
        c = _C(file=buf, force_terminal=True, width=80)
        c.print(self._build_renderable())
        return buf.getvalue()
