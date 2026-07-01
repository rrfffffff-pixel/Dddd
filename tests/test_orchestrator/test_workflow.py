"""Tests for the workflow engine improvements."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from coding_agent.orchestrator.workflow import (
    StageNode,
    StageResult,
    StageType,
    WorkflowEngine,
    WorkflowTask,
)


class MockAgent:
    """Mock agent for testing workflow execution."""

    def __init__(self, result: str = "success", delay: float = 0.0, fail: bool = False):
        self.result = result
        self.delay = delay
        self.fail = fail
        self.call_count = 0

    def run(self, task: str) -> str:
        self.call_count += 1
        if self.delay > 0:
            time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("Agent failed")
        return self.result


class SlowAgent:
    """Mock agent that takes too long to complete."""

    def __init__(self, delay: float = 10.0):
        self.delay = delay

    def run(self, task: str) -> str:
        time.sleep(self.delay)
        return "done"


# ===== Dependency Validation Tests =====


class TestDependencyValidation:
    """Test cycle detection and missing dependency handling."""

    def test_validate_no_dependencies(self):
        engine = WorkflowEngine(agents={})
        stages = {
            "s1": StageNode(id="s1", name="Stage 1", stage_type=StageType.SEQUENTIAL),
            "s2": StageNode(id="s2", name="Stage 2", stage_type=StageType.SEQUENTIAL),
        }
        errors = engine.validate_dependencies(stages)
        assert errors == []

    def test_validate_linear_dependencies(self):
        engine = WorkflowEngine(agents={})
        stages = {
            "s1": StageNode(id="s1", name="Stage 1", stage_type=StageType.SEQUENTIAL, depends_on=[]),
            "s2": StageNode(id="s2", name="Stage 2", stage_type=StageType.SEQUENTIAL, depends_on=["s1"]),
            "s3": StageNode(id="s3", name="Stage 3", stage_type=StageType.SEQUENTIAL, depends_on=["s2"]),
        }
        errors = engine.validate_dependencies(stages)
        assert errors == []

    def test_validate_diamond_dependencies(self):
        engine = WorkflowEngine(agents={})
        stages = {
            "s1": StageNode(id="s1", name="Stage 1", stage_type=StageType.SEQUENTIAL, depends_on=[]),
            "s2": StageNode(id="s2", name="Stage 2", stage_type=StageType.SEQUENTIAL, depends_on=["s1"]),
            "s3": StageNode(id="s3", name="Stage 3", stage_type=StageType.SEQUENTIAL, depends_on=["s1"]),
            "s4": StageNode(id="s4", name="Stage 4", stage_type=StageType.SEQUENTIAL, depends_on=["s2", "s3"]),
        }
        errors = engine.validate_dependencies(stages)
        assert errors == []

    def test_validate_missing_dependency(self):
        engine = WorkflowEngine(agents={})
        stages = {
            "s1": StageNode(id="s1", name="Stage 1", stage_type=StageType.SEQUENTIAL, depends_on=["s2"]),
        }
        errors = engine.validate_dependencies(stages)
        assert len(errors) == 1
        assert "unknown stage" in errors[0]
        assert "s2" in errors[0]

    def test_validate_multiple_missing_dependencies(self):
        engine = WorkflowEngine(agents={})
        stages = {
            "s1": StageNode(id="s1", name="Stage 1", stage_type=StageType.SEQUENTIAL, depends_on=["s2", "s3"]),
            "s2": StageNode(id="s2", name="Stage 2", stage_type=StageType.SEQUENTIAL, depends_on=["s4"]),
        }
        errors = engine.validate_dependencies(stages)
        assert len(errors) == 2

    def test_validate_cycle_two_nodes(self):
        engine = WorkflowEngine(agents={})
        stages = {
            "s1": StageNode(id="s1", name="Stage 1", stage_type=StageType.SEQUENTIAL, depends_on=["s2"]),
            "s2": StageNode(id="s2", name="Stage 2", stage_type=StageType.SEQUENTIAL, depends_on=["s1"]),
        }
        errors = engine.validate_dependencies(stages)
        assert len(errors) == 1
        assert "cycle" in errors[0].lower()

    def test_validate_cycle_three_nodes(self):
        engine = WorkflowEngine(agents={})
        stages = {
            "s1": StageNode(id="s1", name="Stage 1", stage_type=StageType.SEQUENTIAL, depends_on=["s3"]),
            "s2": StageNode(id="s2", name="Stage 2", stage_type=StageType.SEQUENTIAL, depends_on=["s1"]),
            "s3": StageNode(id="s3", name="Stage 3", stage_type=StageType.SEQUENTIAL, depends_on=["s2"]),
        }
        errors = engine.validate_dependencies(stages)
        assert len(errors) == 1
        assert "cycle" in errors[0].lower()

    def test_validate_self_cycle(self):
        engine = WorkflowEngine(agents={})
        stages = {
            "s1": StageNode(id="s1", name="Stage 1", stage_type=StageType.SEQUENTIAL, depends_on=["s1"]),
        }
        errors = engine.validate_dependencies(stages)
        assert len(errors) == 1
        assert "cycle" in errors[0].lower()


# ===== Timeout Handling Tests =====


class TestTimeoutHandling:
    """Test task timeout handling."""

    def test_no_timeout(self):
        agent = MockAgent(result="done")
        engine = WorkflowEngine(agents={"agent": agent})
        stage = StageNode(
            id="s1",
            name="Stage 1",
            stage_type=StageType.SEQUENTIAL,
            tasks=[WorkflowTask(id="t1", description="test", agent_name="agent")],
        )
        results = engine.run({"s1": stage})
        assert len(results) == 1
        assert results[0].success

    def test_stage_task_timeout(self):
        agent = SlowAgent(delay=5.0)
        engine = WorkflowEngine(agents={"agent": agent}, default_task_timeout=0.1)
        stage = StageNode(
            id="s1",
            name="Stage 1",
            stage_type=StageType.PARALLEL,
            tasks=[WorkflowTask(id="t1", description="slow task", agent_name="agent")],
        )
        results = engine.run({"s1": stage})
        assert len(results) == 1
        assert not results[0].success
        assert results[0].failed_tasks == 1

    def test_stage_specific_timeout_overrides_default(self):
        agent = SlowAgent(delay=5.0)
        engine = WorkflowEngine(agents={"agent": agent}, default_task_timeout=10.0)
        stage = StageNode(
            id="s1",
            name="Stage 1",
            stage_type=StageType.PARALLEL,
            tasks=[WorkflowTask(id="t1", description="slow task", agent_name="agent")],
            task_timeout=0.1,
        )
        results = engine.run({"s1": stage})
        assert not results[0].success


# ===== Error Recovery Tests =====


class TestErrorRecovery:
    """Test error recovery with skip-failed and continue capabilities."""

    def test_agent_not_found(self):
        engine = WorkflowEngine(agents={})
        stage = StageNode(
            id="s1",
            name="Stage 1",
            stage_type=StageType.SEQUENTIAL,
            tasks=[WorkflowTask(id="t1", description="test", agent_name="nonexistent")],
        )
        results = engine.run({"s1": stage})
        assert len(results) == 1
        assert not results[0].success

    def test_agent_exception(self):
        agent = MockAgent(fail=True)
        engine = WorkflowEngine(agents={"agent": agent})
        stage = StageNode(
            id="s1",
            name="Stage 1",
            stage_type=StageType.SEQUENTIAL,
            tasks=[WorkflowTask(id="t1", description="test", agent_name="agent")],
        )
        results = engine.run({"s1": stage})
        assert not results[0].success
        assert results[0].errors["t1"]

    def test_partial_failure_in_parallel(self):
        agent_ok = MockAgent(result="ok")
        agent_fail = MockAgent(fail=True)
        engine = WorkflowEngine(agents={"ok": agent_ok, "fail": agent_fail})
        stage = StageNode(
            id="s1",
            name="Stage 1",
            stage_type=StageType.PARALLEL,
            tasks=[
                WorkflowTask(id="t1", description="ok task", agent_name="ok"),
                WorkflowTask(id="t2", description="fail task", agent_name="fail"),
            ],
        )
        results = engine.run({"s1": stage})
        assert len(results) == 1
        assert results[0].failed_tasks == 1
        assert results[0].succeeded_tasks == 1

    def test_error_in_result_string(self):
        agent = MockAgent(result="error: something went wrong")
        engine = WorkflowEngine(agents={"agent": agent})
        stage = StageNode(
            id="s1",
            name="Stage 1",
            stage_type=StageType.SEQUENTIAL,
            tasks=[WorkflowTask(id="t1", description="test", agent_name="agent")],
        )
        results = engine.run({"s1": stage})
        assert not results[0].success

    def test_downstream_skipped_on_failure(self):
        agent_fail = MockAgent(fail=True)
        agent_ok = MockAgent(result="ok")
        engine = WorkflowEngine(agents={"fail": agent_fail, "ok": agent_ok})
        stages = {
            "s1": StageNode(
                id="s1",
                name="Stage 1",
                stage_type=StageType.SEQUENTIAL,
                tasks=[WorkflowTask(id="t1", description="fail", agent_name="fail")],
            ),
            "s2": StageNode(
                id="s2",
                name="Stage 2",
                stage_type=StageType.SEQUENTIAL,
                depends_on=["s1"],
                tasks=[WorkflowTask(id="t2", description="ok", agent_name="ok")],
            ),
        }
        results = engine.run(stages)
        assert len(results) == 1
        assert not results[0].success


# ===== Cancel Capability Tests =====


class TestCancelCapability:
    """Test workflow cancellation."""

    def test_cancel_before_run(self):
        engine = WorkflowEngine(agents={})
        engine.cancel()
        assert engine.is_cancelled

    def test_cancel_clears_on_run(self):
        engine = WorkflowEngine(agents={})
        engine.cancel()
        engine.run({})
        assert not engine.is_cancelled

    def test_cancel_during_execution(self):
        call_count = 0

        def slow_agent(task: str):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                engine.cancel()
            return "done"

        mock_agent = MagicMock(side_effect=slow_agent)
        engine = WorkflowEngine(agents={"agent": mock_agent})
        stages = {
            "s1": StageNode(
                id="s1",
                name="Stage 1",
                stage_type=StageType.SEQUENTIAL,
                tasks=[
                    WorkflowTask(id="t1", description="task 1", agent_name="agent"),
                    WorkflowTask(id="t2", description="task 2", agent_name="agent"),
                ],
            ),
        }
        results = engine.run(stages)
        assert len(results) <= 1

    def test_cancel_parallel_tasks(self):
        cancel_event = threading.Event()

        def slow_agent(task: str):
            cancel_event.wait(timeout=5.0)
            return "done"

        mock_agent = MagicMock(side_effect=slow_agent)
        engine = WorkflowEngine(agents={"agent": mock_agent})
        stage = StageNode(
            id="s1",
            name="Stage 1",
            stage_type=StageType.PARALLEL,
            tasks=[
                WorkflowTask(id="t1", description="task 1", agent_name="agent"),
                WorkflowTask(id="t2", description="task 2", agent_name="agent"),
            ],
        )

        def cancel_after_delay():
            time.sleep(0.1)
            engine.cancel()
            cancel_event.set()

        thread = threading.Thread(target=cancel_after_delay)
        thread.start()
        results = engine.run({"s1": stage})
        thread.join()
        assert len(results) == 1


# ===== Result Aggregation Statistics Tests =====


class TestResultAggregation:
    """Test stage result aggregation and statistics."""

    def test_successful_stage_statistics(self):
        agent = MockAgent(result="ok")
        engine = WorkflowEngine(agents={"agent": agent})
        stage = StageNode(
            id="s1",
            name="Stage 1",
            stage_type=StageType.SEQUENTIAL,
            tasks=[
                WorkflowTask(id="t1", description="task 1", agent_name="agent"),
                WorkflowTask(id="t2", description="task 2", agent_name="agent"),
                WorkflowTask(id="t3", description="task 3", agent_name="agent"),
            ],
        )
        results = engine.run({"s1": stage})
        result = results[0]
        assert result.total_tasks == 3
        assert result.succeeded_tasks == 3
        assert result.failed_tasks == 0
        assert result.skipped_tasks == 0
        assert result.success is True

    def test_failed_stage_statistics(self):
        agent_fail = MockAgent(fail=True)
        agent_ok = MockAgent(result="ok")
        engine = WorkflowEngine(agents={"fail": agent_fail, "ok": agent_ok})
        stage = StageNode(
            id="s1",
            name="Stage 1",
            stage_type=StageType.SEQUENTIAL,
            tasks=[
                WorkflowTask(id="t1", description="fail", agent_name="fail"),
                WorkflowTask(id="t2", description="ok", agent_name="ok"),
            ],
        )
        results = engine.run({"s1": stage})
        result = results[0]
        assert result.total_tasks == 2
        assert result.failed_tasks >= 1
        assert result.success is False

    def test_duration_tracking(self):
        agent = MockAgent(result="ok")
        engine = WorkflowEngine(agents={"agent": agent})
        stage = StageNode(
            id="s1",
            name="Stage 1",
            stage_type=StageType.SEQUENTIAL,
            tasks=[WorkflowTask(id="t1", description="task", agent_name="agent")],
        )
        results = engine.run({"s1": stage})
        assert results[0].duration_seconds >= 0

    def test_task_result_collection(self):
        agent = MockAgent(result="output data")
        engine = WorkflowEngine(agents={"agent": agent})
        stage = StageNode(
            id="s1",
            name="Stage 1",
            stage_type=StageType.SEQUENTIAL,
            tasks=[WorkflowTask(id="t1", description="task", agent_name="agent")],
        )
        results = engine.run({"s1": stage})
        assert "t1" in results[0].task_results
        assert results[0].task_results["t1"] == "output data"

    def test_error_collection(self):
        agent = MockAgent(fail=True)
        engine = WorkflowEngine(agents={"agent": agent})
        stage = StageNode(
            id="s1",
            name="Stage 1",
            stage_type=StageType.SEQUENTIAL,
            tasks=[WorkflowTask(id="t1", description="task", agent_name="agent")],
        )
        results = engine.run({"s1": stage})
        assert "t1" in results[0].errors


# ===== Progress Callback Tests =====


class TestProgressCallback:
    """Test progress notification callbacks."""

    def test_progress_callback_invoked(self):
        callbacks = []
        engine = WorkflowEngine(
            agents={"agent": MockAgent(result="ok")},
            progress_callback=lambda sid, status, msg: callbacks.append((sid, status, msg)),
        )
        stage = StageNode(
            id="s1",
            name="Stage 1",
            stage_type=StageType.SEQUENTIAL,
            tasks=[WorkflowTask(id="t1", description="task", agent_name="agent")],
        )
        engine.run({"s1": stage})
        assert len(callbacks) > 0
        statuses = [c[1] for c in callbacks]
        assert "running" in statuses
        assert "done" in statuses


# ===== Workflow Run Integration Tests =====


class TestWorkflowRun:
    """Integration tests for workflow execution."""

    def test_empty_stages(self):
        engine = WorkflowEngine(agents={})
        results = engine.run({})
        assert results == []

    def test_single_stage_sequential(self):
        agent = MockAgent(result="done")
        engine = WorkflowEngine(agents={"agent": agent})
        stage = StageNode(
            id="s1",
            name="Stage 1",
            stage_type=StageType.SEQUENTIAL,
            tasks=[WorkflowTask(id="t1", description="task", agent_name="agent")],
        )
        results = engine.run({"s1": stage})
        assert len(results) == 1
        assert results[0].success

    def test_single_stage_parallel(self):
        agent = MockAgent(result="done")
        engine = WorkflowEngine(agents={"agent": agent})
        stage = StageNode(
            id="s1",
            name="Stage 1",
            stage_type=StageType.PARALLEL,
            tasks=[
                WorkflowTask(id="t1", description="task 1", agent_name="agent"),
                WorkflowTask(id="t2", description="task 2", agent_name="agent"),
            ],
        )
        results = engine.run({"s1": stage})
        assert len(results) == 1
        assert results[0].success

    def test_multi_stage_dependency_order(self):
        execution_order = []

        class TrackingAgent:
            def run(self, task: str) -> str:
                execution_order.append(task)
                return "done"

        engine = WorkflowEngine(agents={"agent": TrackingAgent()})
        stages = {
            "s1": StageNode(
                id="s1", name="Stage 1", stage_type=StageType.SEQUENTIAL,
                tasks=[WorkflowTask(id="t1", description="first", agent_name="agent")],
            ),
            "s2": StageNode(
                id="s2", name="Stage 2", stage_type=StageType.SEQUENTIAL,
                depends_on=["s1"],
                tasks=[WorkflowTask(id="t2", description="second", agent_name="agent")],
            ),
            "s3": StageNode(
                id="s3", name="Stage 3", stage_type=StageType.SEQUENTIAL,
                depends_on=["s2"],
                tasks=[WorkflowTask(id="t3", description="third", agent_name="agent")],
            ),
        }
        engine.run(stages)
        assert execution_order == ["first", "second", "third"]

    def test_invalid_dependencies_return_empty(self):
        engine = WorkflowEngine(agents={"agent": MockAgent(result="ok")})
        stages = {
            "s1": StageNode(
                id="s1", name="Stage 1", stage_type=StageType.SEQUENTIAL,
                depends_on=["s2"],
                tasks=[WorkflowTask(id="t1", description="task", agent_name="agent")],
            ),
        }
        results = engine.run(stages)
        assert results == []
