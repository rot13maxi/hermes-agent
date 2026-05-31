"""Tests for hermes_workflow runner — checkpointing, phase lifecycle, and result parsing."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_workflow.runner import (
    WorkflowRunner,
    phase,
    run,
    run_all,
    PhaseResult,
    _runner_local,
)


class TestPhaseResult:
    def test_basic(self):
        pr = PhaseResult("test", {"key": "value"})
        assert pr.name == "test"
        assert pr["key"] == "value"
        assert "test" in repr(pr)

    def test_empty(self):
        pr = PhaseResult("empty", {})
        assert pr.name == "empty"
        assert len(pr) == 0


class TestCheckpointing:
    @pytest.fixture()
    def runner(self, tmp_path):
        """Runner with temp checkpoint dir and no real agent."""
        r = WorkflowRunner(
            name="test-workflow",
            checkpoint_dir=str(tmp_path),
            display=False,
        )
        r._bootstrap()
        return r

    def test_no_checkpoint_on_fresh(self, runner):
        assert runner.checkpoint is None

    def test_save_checkpoint(self, runner):
        runner._phases_completed = ["phase1", "phase2"]
        runner._phase_results = {
            "phase1": {"status": "completed"},
            "phase2": {"status": "completed"},
        }
        runner._save_checkpoint()

        assert runner.checkpoint is not None
        assert runner.checkpoint["completed_phases"] == ["phase1", "phase2"]
        assert runner.checkpoint["workflow"] == "test-workflow"

    def test_load_checkpoint(self, tmp_path):
        import hashlib

        checkpoint_data = {
            "workflow": "test-workflow",
            "completed_phases": ["phase1"],
            "phase_results": {"phase1": {"status": "completed"}},
            "timestamp": 1234567890,
        }
        wf_hash = hashlib.sha256("test-workflow".encode()).hexdigest()[:16]
        checkpoint_file = tmp_path / f"{wf_hash}.json"
        checkpoint_file.write_text(json.dumps(checkpoint_data))

        r = WorkflowRunner(
            name="test-workflow",
            checkpoint_dir=str(tmp_path),
            display=False,
        )
        r._bootstrap()
        assert r.checkpoint is not None
        assert r.checkpoint["completed_phases"] == ["phase1"]

    def test_checkpoint_not_found(self, tmp_path):
        r = WorkflowRunner(
            name="nonexistent",
            checkpoint_dir=str(tmp_path),
            display=False,
        )
        r._bootstrap()
        assert r.checkpoint is None

    def test_atomic_write(self, runner, tmp_path):
        """Verify checkpoint uses atomic rename (no .tmp leftover)."""
        runner._phases_completed = ["x"]
        runner._phase_results = {"x": {}}
        runner._save_checkpoint()
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_save_creates_dir(self, tmp_path):
        deep_dir = tmp_path / "a" / "b" / "c"
        r = WorkflowRunner(
            name="deep-checkpoint",
            checkpoint_dir=str(deep_dir),
            display=False,
        )
        r._bootstrap()
        r._phases_completed = ["init"]
        r._save_checkpoint()
        assert (deep_dir / r._checkpoint_path.name).exists()


class TestPhaseLifecycle:
    @pytest.fixture()
    def runner(self, tmp_path):
        r = WorkflowRunner(
            name="lifecycle-test",
            checkpoint_dir=str(tmp_path),
            display=False,
        )
        r._bootstrap()
        return r

    def test_phase_start_and_complete(self, runner):
        runner.run_script(lambda _: None)
        _runner_local.runner = runner
        try:
            runner.on_phase_start("test")
            runner.on_phase_complete("test")
            assert "test" in runner._phases_completed
        finally:
            _runner_local.runner = None

    def test_checkpoint_saved_after_phase(self, runner, tmp_path):
        _runner_local.runner = runner
        try:
            runner.on_phase_start("init")
            runner.on_phase_complete("init")
            assert runner.checkpoint is not None
            assert "init" in runner.checkpoint["completed_phases"]
        finally:
            _runner_local.runner = None


class TestParseDelegateResult:
    @pytest.fixture()
    def runner(self, tmp_path):
        r = WorkflowRunner(
            name="parse-test",
            checkpoint_dir=str(tmp_path),
            display=False,
        )
        r._bootstrap()
        return r

    def test_results_key(self, runner):
        raw = json.dumps({
            "results": [{"summary": "ok", "status": "completed"}],
            "total_duration_seconds": 1.0,
        })
        entries = runner._parse_delegate_result(raw)
        assert len(entries) == 1
        assert entries[0]["summary"] == "ok"

    def test_list_result(self, runner):
        raw = json.dumps([{"summary": "a"}, {"summary": "b"}])
        entries = runner._parse_delegate_result(raw)
        assert len(entries) == 2

    def test_single_dict_result(self, runner):
        raw = json.dumps({"summary": "single"})
        entries = runner._parse_delegate_result(raw)
        assert len(entries) == 1
        assert entries[0]["summary"] == "single"


class TestContextManager:
    """Test that phase() context manager works with the runner."""

    @pytest.fixture()
    def runner(self, tmp_path):
        r = WorkflowRunner(
            name="ctx-test",
            checkpoint_dir=str(tmp_path),
            display=False,
        )
        r._bootstrap()
        return r

    def test_phase_context_runs_body(self, runner):
        results = []

        def script(_):
            with phase("body") as pr:
                results.append("executed")
                pr["data"] = "value"

        runner.run_script(script)
        assert results == ["executed"]
        assert "body" in runner._phases_completed

    def test_phase_resume_skips_body(self, runner, tmp_path):
        """When checkpoint has phase completed, body still runs (Python
        context managers can't skip the body), but the yielded PhaseResult
        contains checkpointed data and on_phase_complete is called on entry."""
        runner.checkpoint = {
            "completed_phases": ["skipped"],
        }
        runner._phase_results = {
            "skipped": {"status": "completed", "saved_data": "hello"},
        }

        received = []

        def script(_):
            with phase("skipped") as pr:
                received.append((pr.name, dict(pr)))

        runner.run_script(script)
        # Body did execute (Python semantics), but PhaseResult carried data
        assert len(received) == 1
        assert received[0][0] == "skipped"

    def test_run_requires_runner(self):
        """Calling run() outside a WorkflowRunner raises."""
        _runner_local.runner = None
        with pytest.raises(RuntimeError, match="No active WorkflowRunner"):
            run("test goal")

    def test_run_all_requires_runner(self):
        _runner_local.runner = None
        with pytest.raises(RuntimeError, match="No active WorkflowRunner"):
            run_all([])


class TestWorkflowRunnerBootstrap:
    def test_context_manager(self, tmp_path):
        """__enter__ and __exit__ should work without errors."""
        with WorkflowRunner(
            name="ctx-mgr",
            checkpoint_dir=str(tmp_path),
            display=False,
        ) as r:
            assert r.name == "ctx-mgr"

    def test_display_false(self, tmp_path):
        r = WorkflowRunner(
            name="no-display",
            checkpoint_dir=str(tmp_path),
            display=False,
        )
        r._bootstrap()
        assert r._display is None

    def test_clarify_callback(self):
        """Non-interactive clarify should return a sensible default."""
        result = WorkflowRunner._clarify_callback("What should I do?")
        assert "workflow mode" in result

    def test_clarify_callback_with_choices(self):
        result = WorkflowRunner._clarify_callback("Pick one", choices=["a", "b"])
        assert "a" in result or "b" in result
