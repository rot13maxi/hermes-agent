"""Workflow CLI: run, list, and manage multi-agent workflows."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home


# ---------------------------------------------------------------------------
# Workflow discovery
# ---------------------------------------------------------------------------

_DEFAULT_DIRS = [
    "workflows",  # project-level
    str(get_hermes_home() / "workflows"),  # user-level
]


def _find_workflow_dirs(args: argparse.Namespace) -> List[str]:
    """Return ordered list of directories to search for workflow scripts."""
    dirs = list(getattr(args, "workflows_dir", []) or [])
    for d in _DEFAULT_DIRS:
        if d not in dirs:
            dirs.append(d)
    return dirs


def discover_workflows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Find all .py workflow scripts and return metadata."""
    workflows: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}  # basename -> index for dedup

    for directory in _find_workflow_dirs(args):
        d = Path(directory)
        if not d.is_dir():
            continue
        for py in sorted(d.glob("*.py")):
            if py.name.startswith("_"):
                continue
            if py.name in seen:
                continue
            try:
                stat = py.stat()
                mtime = stat.st_mtime
            except OSError:
                mtime = 0

            # Extract docstring for description
            desc = ""
            try:
                source = py.read_text(encoding="utf-8")
                # Quick heuristic: grab first docstring
                import ast as _ast

                tree = _ast.parse(source)
                if tree.body and isinstance(tree.body[0], _ast.Module):
                    doc = _ast.get_docstring(tree)
                    if doc:
                        desc = doc.split("\n")[0]
            except Exception:
                pass

            idx = len(workflows)
            seen[py.name] = idx
            workflows.append({
                "name": py.stem,
                "file": str(py),
                "directory": str(d),
                "modified": mtime,
                "description": desc,
            })

    return workflows


# ---------------------------------------------------------------------------
# Workflow execution
# ---------------------------------------------------------------------------


def run_workflow(args: argparse.Namespace) -> int:
    """Execute a workflow script."""
    workflow_file = args.workflow
    resume = getattr(args, "resume", False)
    force = getattr(args, "force", False)
    model = getattr(args, "model", None)
    provider = getattr(args, "provider", None)

    # Resolve workflow file
    wf_path = Path(workflow_file)
    if not wf_path.is_file():
        # Search in known directories
        for directory in _find_workflow_dirs(args):
            candidate = Path(directory) / f"{workflow_file}.py"
            if candidate.is_file():
                wf_path = candidate
                break
        else:
            print(f"Workflow not found: {workflow_file}", file=sys.stderr)
            return 1

    # Check for existing checkpoint
    from hermes_workflow import WorkflowRunner

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
                print(f"Workflow has completed phases: {completed}")
                print(f"  Use --resume to continue, --force to restart")
                return 0
        except Exception:
            pass

    # Import the workflow script
    spec = importlib.util.spec_from_file_location("workflow_script", wf_path)
    if spec is None or spec.loader is None:
        print(f"Cannot load workflow: {wf_path}", file=sys.stderr)
        return 1

    module = importlib.util.module_from_spec(spec)
    sys.modules["workflow_script"] = module
    sys.modules["hermes_workflow"] = __import__("hermes_workflow")

    # Ensure the workflow script's directory is importable
    sys.path.insert(0, str(wf_path.parent))

    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"Error loading workflow: {exc}", file=sys.stderr)
        return 1

    # Find the workflow() function
    workflow_fn = getattr(module, "workflow", None)
    if workflow_fn is None or not callable(workflow_fn):
        print(f"Workflow must define a workflow() function: {wf_path}", file=sys.stderr)
        return 1

    # Run it
    try:
        with WorkflowRunner(
            name=wf_path.stem,
            model=model,
            provider=provider,
        ) as runner:
            runner.run_script(lambda _: workflow_fn())
    except Exception as exc:
        print(f"Workflow failed: {exc}", file=sys.stderr)
        return 1
    finally:
        sys.path.remove(str(wf_path.parent))
        sys.modules.pop("workflow_script", None)

    return 0


# ---------------------------------------------------------------------------
# Workflow list
# ---------------------------------------------------------------------------


def list_workflows(args: argparse.Namespace) -> int:
    """List available workflows."""
    workflows = discover_workflows(args)

    if not workflows:
        print("No workflows found.")
        print(f"  Search paths: {_DEFAULT_DIRS}")
        return 0

    print(f"{'NAME':<25} {'FILE':<50} {'DESCRIPTION'}")
    print("-" * 100)
    for wf in workflows:
        print(f"{wf['name']:<25} {wf['file']:<50} {wf.get('description', '')}")

    return 0


# ---------------------------------------------------------------------------
# Workflow delete checkpoint
# ---------------------------------------------------------------------------


def delete_checkpoint(args: argparse.Namespace) -> int:
    """Delete a workflow checkpoint (allows full re-run)."""
    import hashlib
    from pathlib import Path

    workflow_name = args.workflow
    workflow_file = getattr(args, "file", None)

    if workflow_file:
        wf_hash = hashlib.sha256(Path(workflow_file).name.encode()).hexdigest()[:16]
    else:
        wf_hash = hashlib.sha256(workflow_name.encode()).hexdigest()[:16]

    checkpoint_dir = get_hermes_home() / "workflow_checkpoints"
    checkpoint_path = checkpoint_dir / f"{wf_hash}.json"

    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Deleted checkpoint: {checkpoint_path}")
    else:
        print(f"No checkpoint found for workflow: {workflow_name}")

    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def cmd_workflow(args):
    """Handle `hermes workflow <subcommand>`.

    This is called by main.py after argparse dispatches the workflow subparser.
    The actual subcommand logic is in the workflow_subparsers registered in
    main.py — this function handles the no-subcommand case.
    """
    subcommand = getattr(args, "workflow_command", None)
    if subcommand is None:
        args.workflow_parser.print_help()
        return 0

    if subcommand == "run":
        return run_workflow(args)
    elif subcommand == "list":
        return list_workflows(args)
    elif subcommand == "delete":
        return delete_checkpoint(args)
    else:
        print(f"Unknown workflow command: {subcommand}", file=sys.stderr)
        return 1


def register_workflow_subparsers(subparsers):
    """Register the `hermes workflow` subparser and its subcommands.

    Call from main.py after building the top-level subparsers.
    """
    wf_parser = subparsers.add_parser(
        "workflow",
        help="Run multi-agent workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  hermes workflow run sample_audit        Run a workflow by name\n"
            "  hermes workflow run workflows/foo.py    Run a workflow by path\n"
            "  hermes workflow run audit --resume      Resume a paused workflow\n"
            "  hermes workflow run audit --force       Ignore checkpoint, re-run\n"
            "  hermes workflow list                    List available workflows\n"
            "  hermes workflow delete audit            Clear checkpoint for re-run\n"
        ),
    )
    wf_subparsers = wf_parser.add_subparsers(dest="workflow_command")

    # run
    run_parser = wf_subparsers.add_parser(
        "run",
        help="Run a workflow script",
    )
    run_parser.add_argument(
        "workflow",
        help="Workflow name or path to .py file",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the last completed phase",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing checkpoint and re-run from scratch",
    )
    run_parser.add_argument(
        "--model",
        default=None,
        help="Override model for child agents",
    )
    run_parser.add_argument(
        "--provider",
        default=None,
        help="Override provider for child agents",
    )
    run_parser.add_argument(
        "--workflows-dir",
        action="append",
        default=[],
        dest="workflows_dir",
        help="Additional directory to search for workflows (can repeat)",
    )
    run_parser.set_defaults(func=run_workflow)

    # list
    list_parser = wf_subparsers.add_parser(
        "list",
        aliases=["ls"],
        help="List available workflows",
    )
    list_parser.add_argument(
        "--workflows-dir",
        action="append",
        default=[],
        dest="workflows_dir",
        help="Additional directory to search for workflows (can repeat)",
    )
    list_parser.set_defaults(func=list_workflows)

    # delete checkpoint
    delete_parser = wf_subparsers.add_parser(
        "delete",
        aliases=["rm"],
        help="Delete a workflow checkpoint",
    )
    delete_parser.add_argument(
        "workflow",
        help="Workflow name",
    )
    delete_parser.add_argument(
        "--file",
        default=None,
        help="Workflow file path (for hash matching)",
    )
    delete_parser.set_defaults(func=delete_checkpoint)

    # Default: show help
    wf_parser.set_defaults(func=lambda a: a.workflow_parser.print_help() or 0)
    wf_parser.workflow_parser = wf_parser  # back-ref for help

    return wf_parser
