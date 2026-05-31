---
sidebar_position: 8
title: "Multi-Agent Workflows"
description: "Author Python-based multi-agent workflows with checkpointed phases, parallel execution, and structured data flow"
---

# Multi-Agent Workflows

Workflows let you decompose large tasks into independent, checkpointed phases with parallel agents and structured data flow. Each workflow is a Python script defining a `workflow()` function that uses `phase()`, `run()`, and `run_all()` primitives.

Unlike `delegate_task` (which spawns children synchronously within a single turn), workflows run as standalone processes with checkpointing — they survive interrupts, can be resumed mid-flight, and enforce structured output between phases.

## Why Workflows

**Use workflows when:**
- A task has 3+ independent subtasks feeding into a synthesis step
- You need structured data flowing between phases (JSON-validated)
- The work is long-running and should survive interrupts
- You want to save and re-run the same pipeline repeatedly

**Use something else when:**
- Single parallel task → `delegate_task` is simpler
- Mechanical multi-step with no reasoning → `execute_code`
- Durable long-running scripts without LLM → `cronjob` with `no_agent=True`

## Quick Start

```bash
# List available workflows
hermes workflow list

# Run a workflow by name
hermes workflow run sample_audit

# Resume from checkpoint
hermes workflow run sample_audit --resume

# Force re-run from scratch
hermes workflow run sample_audit --force
```

## Writing a Workflow

A workflow is a Python file with a `workflow()` function. The `hermes_workflow` library provides three primitives:

| Primitive | Description |
|-----------|-------------|
| `phase(name)` | Context manager marking a checkpointed section. On resume, completed phases are skipped. |
| `run(goal, ...)` | Run a single child agent. Returns the agent's response. |
| `run_all(tasks, ...)` | Run multiple child agents concurrently. Returns a list of results. |

All three support `response_schema` — a JSON Schema dict that validates the child agent's output. If the response doesn't match, the agent retries once automatically.

### Minimal Example

```python
from hermes_workflow import phase, run, run_all

def workflow():
    """Research three topics in parallel, then synthesize."""

    # Phase 1: Parallel research
    with phase("research"):
        results = run_all([
            {"goal": "Research topic A", "toolsets": ["web"]},
            {"goal": "Research topic B", "toolsets": ["web"]},
            {"goal": "Research topic C", "toolsets": ["web"]},
        ], max_concurrent=3)

    # Phase 2: Synthesis — data from phase 1 flows in as context
    with phase("synthesis"):
        summary = run(
            "Synthesize the research findings into a briefing",
            context=str(results),
        )
```

Save as `workflows/research.py` and run:

```bash
hermes workflow run research
```

### Structured Output with `response_schema`

Enforce the shape of child agent responses:

```python
with phase("discover"):
    files = run(
        "Find all route files under src/routes/",
        toolsets=["file", "terminal"],
        response_schema={
            "type": "array",
            "items": {"type": "string"},
        },
    )
    # files is guaranteed to be a list of strings
```

Schema supports: `type`, `required`, `enum`, `const`, `properties`, `items`, `minItems`, `maxItems`, `additionalProperties`. Full JSON Schema is not implemented — just the patterns needed for agent output validation.

### Conditional Flow Control

Workflows use real Python — `if`, `for`, list comprehensions, early returns:

```python
with phase("audit"):
    audits = run_all([
        {"goal": f"Audit {f}", "toolsets": ["file"]}
        for f in files  # from previous phase
    ])

# Real Python branching
critical = [a for a in audits if a.get("severity") in ("critical", "high")]

if critical:
    with phase("escalation"):
        run(
            "Create a ticket for critical findings",
            context=str(critical),
        )
else:
    with phase("ack"):
        run("Note that no critical findings were found")
```

### Data Threading Between Phases

Results from one phase become context for the next. On resume, skipped phases still have their saved results available via `runner.phase_results`:

```python
with phase("discover"):
    files = run("Find route files", ...)

with phase("audit"):
    # On first run: files comes from the phase above
    # On resume: files is skipped, but runner.phase_results["discover"] has it
    audits = run_all([...], max_concurrent=6)

with phase("report"):
    run(
        "Compile findings into a report",
        context=f"Audit results from {len(audits)} files:\n{audits}",
    )
```

## Where Workflows Live

Workflows are discovered from two locations:

| Path | Scope |
|------|-------|
| `workflows/` | Project-level (relative to CWD) |
| `~/.hermes/workflows/` | User-level (available everywhere) |

Pass `--workflows-dir <path>` to search additional directories.

## Checkpointing

Every phase boundary saves a checkpoint to `~/.hermes/workflow_checkpoints/`. If a workflow is interrupted (SIGINT, crash, OOM), resume from the last completed phase:

```bash
hermes workflow run sample_audit --resume
```

To clear a checkpoint and start fresh:

```bash
hermes workflow delete sample_audit
hermes workflow run sample_audit --force
```

Checkpoint files store `completed_phases`, `phase_results`, and a timestamp. The filename is a SHA-256 hash of the workflow name.

## Model Overrides

Override the model or provider for all child agents in a workflow:

```bash
hermes workflow run audit --model gpt-4o --provider openrouter
```

This is useful for running workflows on cheaper models during development or pinning a specific model for production runs.

## Running from Conversation

The agent can trigger workflows directly using the `workflow_run` tool:

```text
Run the security audit workflow.
```

The agent calls `workflow_run(script="security_audit")` under the hood. Resume and force are also available:

```text
Resume the security audit workflow from where it left off.
```

## Workflow vs Delegate vs Cron

| Factor | `workflow_run` | `delegate_task` | `cronjob` |
|--------|---------------|-----------------|-----------|
| **Phases** | Multi-phase with checkpoints | Single call | Single run |
| **Structured output** | Enforced via `response_schema` | Optional via `response_schema` | No built-in schema |
| **Survives interrupts** | Yes (checkpointed) | No (cancelled) | Yes (runs in gateway) |
| **Resume** | `--resume` from last phase | No | No (re-runs) |
| **Parallel agents** | `run_all()` with `max_concurrent` | `tasks=[]` batch mode | `context_from` chains |
| **Flow control** | Real Python (`if`/`for`/`return`) | None (flat call) | None (single prompt) |
| **Best for** | Pipelines with 3+ phases | Parallel subtasks within a turn | Scheduled recurring tasks |

## Sample Workflow

A full example is included at `workflows/sample_audit.py` — an API security audit exercising every primitive:

```python
from hermes_workflow import phase, run, run_all

def workflow():
    """Full pipeline: discover -> audit -> triage -> report."""

    # Phase 1: Discovery — single agent finds all route files
    with phase("discover"):
        files = run(
            "Find all API endpoint files under src/routes/",
            toolsets=["file", "terminal"],
            response_schema={
                "type": "array",
                "items": {"type": "string"},
            },
        )

        if not files:
            with phase("empty"):
                run("Create a note stating no route files were found")
            return

    # Phase 2: Parallel audit — one agent per file
    with phase("audit"):
        audits = run_all(
            [
                dict(
                    goal=f"Audit {f} for missing authentication checks",
                    toolsets=["file"],
                    response_schema={
                        "type": "object",
                        "required": ["file", "endpoints", "severity"],
                        "properties": {
                            "file": {"type": "string"},
                            "endpoints": {"type": "array"},
                            "severity": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "critical"],
                            },
                        },
                    },
                )
                for f in files
            ],
            max_concurrent=6,
        )

    # Phase 3: Conditional triage
    critical = [a for a in audits if a.get("severity") in ("critical", "high")]
    if critical:
        with phase("escalation"):
            run("Create ticket for critical findings", context=str(critical))
    else:
        with phase("ack"):
            run("Note that no critical findings were found")

    # Phase 4: Final report
    with phase("report"):
        run(
            "Compile all findings into a report",
            context=f"Audit results:\n{audits}",
        )
```

## API Reference

### `hermes_workflow` Module

```python
from hermes_workflow import phase, run, run_all, WorkflowRunner, validate_response
```

| Export | Type | Description |
|--------|------|-------------|
| `WorkflowRunner` | class | Top-level orchestrator. Bootstraps config, env, credentials. Manages phases and checkpoints. |
| `phase(name)` | context manager | Marks a checkpointed section. Skips completed phases on resume. |
| `run(goal, context, toolsets, response_schema)` | function | Run a single child agent. Returns response string. |
| `run_all(tasks, max_concurrent)` | function | Run multiple child agents concurrently. Returns list of result dicts. |
| `validate_response(text, schema)` | function | Validate text as JSON matching schema. Returns `None` on success, error string on failure. |

### `WorkflowRunner` Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `name` | `"workflow"` | Human-readable name (used for checkpoint filename) |
| `model` | `None` | Override model for child agents |
| `provider` | `None` | Override provider for child agents |
| `checkpoint_dir` | `~/.hermes/workflow_checkpoints/` | Directory for checkpoint files |
| `display` | `True` | Enable Rich progress display |

### CLI Commands

```bash
hermes workflow run <name|path> [--resume] [--force] [--model MODEL] [--provider PROVIDER] [--workflows-dir DIR]
hermes workflow list [--workflows-dir DIR]
hermes workflow delete <name> [--file PATH]
```

:::tip Skill
The `hermes-workflows` skill (in the `clankwork` category) teaches the agent how to author, run, and manage workflows. Load it with `skill_view(name='hermes-workflows')` or add it to your config.
:::
