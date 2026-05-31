"""hermes_workflow — lightweight multi-agent workflow library.

Provides checkpointed, parallelizable agent workflows that integrate with
Hermes Agent's delegation system.

Primitives
----------
- ``phase(name)``  — context manager that marks a checkpointed workflow phase.
- ``run(...)``     — spin up a single child agent (delegates via delegate_task).
- ``run_all(...)`` — spin up multiple child agents concurrently.
- ``WorkflowRunner`` — top-level orchestrator (bootstraps config, manages phases).

Example
-------
>>> from hermes_workflow import WorkflowRunner, phase, run, run_all

>>> with WorkflowRunner("my-workflow") as wf:
...     with phase("research"):
...         results = run_all([
...             {"goal": "Research topic A"},
...             {"goal": "Research topic B"},
...         ], max_concurrent=2)
...     with phase("synthesis"):
...         summary = run("Synthesize the research findings",
...                       context=str(results))
"""

from __future__ import annotations

from hermes_workflow.runner import WorkflowRunner
from hermes_workflow.runner import phase, run, run_all
from hermes_workflow.schema import validate_response
from hermes_workflow.display import WorkflowDisplay

__all__ = [
    "WorkflowRunner",
    "phase",
    "run",
    "run_all",
    "validate_response",
    "WorkflowDisplay",
]
