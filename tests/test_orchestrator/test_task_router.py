"""Tests for the task orchestrator improvements."""

import time
from unittest.mock import MagicMock, patch

import pytest

from coding_agent.core.agent import Agent, AgentConfig
from coding_agent.core.context import SharedContext
from coding_agent.core.message import MessageBus
from coding_agent.core.tool import ToolRegistry
from coding_agent.models.provider import MockProvider, LLMResponse
from coding_agent.orchestrator.task_router import Subtask, TaskOrchestrator


class MockAgent:
    """Mock agent for testing task orchestrator."""

    def __init__(self, name: str = "code", result: str = "success"):
        self.name = name
        self.result = result
        self.bus = None
        self.call_count = 0

    def run(self, task: str) -> str:
        self.call_count += 1
        return self.result

    def set_context(self, context: SharedContext) -> None:
        pass


# ===== Parallel Execution with Dependencies Tests =====


class TestParallelExecution:
    """Test parallel execution respecting dependency graph."""

    def test_linear_dependency_execution(self):
        provider = MockProvider(responses=['[{"id": "1", "description": "task1", "agent_name": "code", "depends_on": []}, {"id": "2", "description": "task2", "agent_name": "code", "depends_on": ["1"]}]', "result1", "result2"])
        agents = {"code": MockAgent(name="code", result="ok")}
        orch = TaskOrchestrator(provider, agents)

        subtasks = [
            Subtask(id="1", description="task1", agent_name="code", depends_on=[]),
            Subtask(id="2", description="task2", agent_name="code", depends_on=["1"]),
        ]

        ready = orch._get_ready_tasks(subtasks)
        assert len(ready) == 1
        assert ready[0].id == "1"

        subtasks[0].status = "done"
        ready = orch._get_ready_tasks(subtasks)
        assert len(ready) == 1
        assert ready[0].id == "2"

    def test_parallel_independent_tasks(self):
        provider = MockProvider()
        agents = {"code": MockAgent(name="code")}
        orch = TaskOrchestrator(provider, agents)

        subtasks = [
            Subtask(id="1", description="task1", agent_name="code", depends_on=[]),
            Subtask(id="2", description="task2", agent_name="code", depends_on=[]),
            Subtask(id="3", description="task3", agent_name="code", depends_on=[]),
        ]

        ready = orch._get_ready_tasks(subtasks)
        assert len(ready) == 3

    def test_diamond_dependency_execution(self):
        provider = MockProvider()
        agents = {"code": MockAgent(name="code")}
        orch = TaskOrchestrator(provider, agents)

        subtasks = [
            Subtask(id="1", description="task1", agent_name="code", depends_on=[]),
            Subtask(id="2", description="task2", agent_name="code", depends_on=["1"]),
            Subtask(id="3", description="task3", agent_name="code", depends_on=["1"]),
            Subtask(id="4", description="task4", agent_name="code", depends_on=["2", "3"]),
        ]

        ready = orch._get_ready_tasks(subtasks)
        assert len(ready) == 1
        assert ready[0].id == "1"

        subtasks[0].status = "done"
        ready = orch._get_ready_tasks(subtasks)
        assert len(ready) == 2
        ready_ids = {s.id for s in ready}
        assert ready_ids == {"2", "3"}


# ===== Progress Tracking Tests =====


class TestProgressTracking:
    """Test progress tracking functionality."""

    def test_progress_summary(self):
        provider = MockProvider()
        agents = {"code": MockAgent(name="code")}
        orch = TaskOrchestrator(provider, agents)

        subtasks = [
            Subtask(id="1", description="task1", agent_name="code", status="done"),
            Subtask(id="2", description="task2", agent_name="code", status="done"),
            Subtask(id="3", description="task3", agent_name="code", status="pending"),
            Subtask(id="4", description="task4", agent_name="code", status="pending"),
        ]

        summary = orch._get_progress_summary(subtasks)
        assert "2/4" in summary
        assert "50%" in summary

    def test_progress_summary_all_done(self):
        provider = MockProvider()
        agents = {"code": MockAgent(name="code")}
        orch = TaskOrchestrator(provider, agents)

        subtasks = [
            Subtask(id="1", description="task1", agent_name="code", status="done"),
            Subtask(id="2", description="task2", agent_name="code", status="done"),
        ]

        summary = orch._get_progress_summary(subtasks)
        assert "2/2" in summary
        assert "100%" in summary

    def test_progress_summary_empty(self):
        provider = MockProvider()
        agents = {"code": MockAgent(name="code")}
        orch = TaskOrchestrator(provider, agents)

        summary = orch._get_progress_summary([])
        assert "0/0" in summary

    def test_progress_callback_invoked(self):
        callbacks = []
        provider = MockProvider(responses=['[{"id": "1", "description": "task1", "agent_name": "code", "depends_on": []}]', "result"])
        agents = {"code": MockAgent(name="code", result="ok")}
        orch = TaskOrchestrator(
            provider, agents,
            progress_callback=lambda sid, status, msg: callbacks.append((sid, status, msg)),
        )

        orch.run("do something")
        assert len(callbacks) > 0
        statuses = [c[1] for c in callbacks]
        assert "running" in statuses
        assert "done" in statuses


# ===== Retry with Exponential Backoff Tests =====


class TestRetryBackoff:
    """Test retry logic with exponential backoff."""

    def test_should_retry_on_error(self):
        provider = MockProvider()
        agents = {"code": MockAgent(name="code")}
        orch = TaskOrchestrator(provider, agents)

        subtask = Subtask(id="1", description="test", agent_name="code")
        assert orch._should_retry(subtask, "Error: something failed") is True
        assert orch._should_retry(subtask, "exception occurred") is True
        assert orch._should_retry(subtask, "traceback most recent") is True

    def test_should_not_retry_on_success(self):
        provider = MockProvider()
        agents = {"code": MockAgent(name="code")}
        orch = TaskOrchestrator(provider, agents)

        subtask = Subtask(id="1", description="test", agent_name="code")
        assert orch._should_retry(subtask, "Success!") is False
        assert orch._should_retry(subtask, "Done") is False

    def test_should_not_retry_when_max_retries_reached(self):
        provider = MockProvider()
        agents = {"code": MockAgent(name="code")}
        orch = TaskOrchestrator(provider, agents)

        subtask = Subtask(id="1", description="test", agent_name="code", retry_count=3, max_retries=3)
        assert orch._should_retry(subtask, "Error: failed") is False

    def test_backoff_delay_exponential(self):
        provider = MockProvider()
        agents = {"code": MockAgent(name="code")}
        orch = TaskOrchestrator(provider, agents)

        assert orch._backoff_delay(0) == 1.0
        assert orch._backoff_delay(1) == 2.0
        assert orch._backoff_delay(2) == 4.0
        assert orch._backoff_delay(3) == 8.0
        assert orch._backoff_delay(4) == 16.0

    def test_backoff_delay_max_cap(self):
        provider = MockProvider()
        agents = {"code": MockAgent(name="code")}
        orch = TaskOrchestrator(provider, agents)

        assert orch._backoff_delay(10) == 30.0
        assert orch._backoff_delay(100) == 30.0


# ===== Execution Time Tracking Tests =====


class TestExecutionTimeTracking:
    """Test execution time tracking for subtasks."""

    def test_execution_time_recorded(self):
        provider = MockProvider(responses=['[{"id": "1", "description": "task1", "agent_name": "code", "depends_on": []}]', "result"])
        agents = {"code": MockAgent(name="code", result="ok")}
        orch = TaskOrchestrator(provider, agents)

        subtasks = orch.decompose_task("do something")
        assert len(subtasks) == 1

    def test_subtask_execution_time_on_success(self):
        provider = MockProvider()
        agents = {"code": MockAgent(name="code")}
        orch = TaskOrchestrator(provider, agents)

        subtask = Subtask(id="1", description="test", agent_name="code")
        subtask.started_at = time.monotonic()
        time.sleep(0.01)
        subtask.completed_at = time.monotonic()
        subtask.execution_time = subtask.completed_at - subtask.started_at

        assert subtask.execution_time > 0

    def test_print_summary_with_timing(self, capsys):
        provider = MockProvider()
        agents = {"code": MockAgent(name="code")}
        orch = TaskOrchestrator(provider, agents)

        subtasks = [
            Subtask(id="1", description="task1", agent_name="code", status="done", execution_time=1.5),
            Subtask(id="2", description="task2", agent_name="code", status="done", execution_time=2.5),
        ]

        orch._print_summary(subtasks, 4.0)
        captured = capsys.readouterr()
        assert "Execution Summary" in captured.out
        assert "4.0s" in captured.out


# ===== Task Decomposition Tests =====


class TestTaskDecomposition:
    """Test task decomposition logic."""

    def test_decompose_returns_subtasks(self):
        decompose_response = '[{"id": 1, "description": "read files", "agent_name": "code", "depends_on": []}]'
        provider = MockProvider(responses=[decompose_response])
        agents = {"code": MockAgent(name="code")}
        orch = TaskOrchestrator(provider, agents)

        subtasks = orch.decompose_task("read the codebase")
        assert len(subtasks) == 1
        assert subtasks[0].agent_name == "code"
        assert subtasks[0].depends_on == []

    def test_decompose_with_dependencies(self):
        decompose_response = '[{"id": "1", "description": "explore", "agent_name": "code", "depends_on": []}, {"id": "2", "description": "edit", "agent_name": "code", "depends_on": ["1"]}]'
        provider = MockProvider(responses=[decompose_response])
        agents = {"code": MockAgent(name="code")}
        orch = TaskOrchestrator(provider, agents)

        subtasks = orch.decompose_task("explore then edit")
        assert len(subtasks) == 2
        assert subtasks[1].depends_on == ["1"]

    def test_decompose_fallback_on_invalid_json(self):
        provider = MockProvider(responses=["not valid json"])
        agents = {"code": MockAgent(name="code")}
        orch = TaskOrchestrator(provider, agents)

        subtasks = orch.decompose_task("do something")
        assert len(subtasks) == 1
        assert subtasks[0].description == "do something"


# ===== Simple Routing Tests =====


class TestSimpleRouting:
    """Test simple task routing logic."""

    def test_route_test_task(self):
        provider = MockProvider(responses=["test result"])
        agents = {"test": MockAgent(name="test", result="test result")}
        orch = TaskOrchestrator(provider, agents)

        result = orch.run_simple("run the test suite")
        assert result == "test result"

    def test_route_build_task(self):
        provider = MockProvider(responses=["build result"])
        agents = {"shell": MockAgent(name="shell", result="build result")}
        orch = TaskOrchestrator(provider, agents)

        result = orch.run_simple("install dependencies")
        assert result == "build result"

    def test_route_review_task(self):
        provider = MockProvider(responses=["review result"])
        agents = {"review": MockAgent(name="review", result="review result")}
        orch = TaskOrchestrator(provider, agents)

        result = orch.run_simple("review the code for bugs")
        assert result == "review result"

    def test_route_default_to_code(self):
        provider = MockProvider(responses=["code result"])
        agents = {"code": MockAgent(name="code", result="code result")}
        orch = TaskOrchestrator(provider, agents)

        result = orch.run_simple("write a function")
        assert result == "code result"

    def test_route_fallback_when_agent_missing(self):
        provider = MockProvider(responses=["fallback"])
        agents = {"code": MockAgent(name="code", result="fallback")}
        orch = TaskOrchestrator(provider, agents)

        result = orch.run_simple("run tests")
        assert "not available" in result.lower()


# ===== Shared Context Integration Tests =====


class TestSharedContextIntegration:
    """Test integration with SharedContext."""

    def test_context_initialized(self):
        provider = MockProvider()
        agents = {"code": MockAgent(name="code")}
        orch = TaskOrchestrator(provider, agents)

        assert orch.context is not None
        assert isinstance(orch.context, SharedContext)

    def test_context_set_on_agents(self):
        provider = MockProvider()
        mock_agent = MockAgent(name="code")
        agents = {"code": mock_agent}
        orch = TaskOrchestrator(provider, agents)

        assert mock_agent.bus is not None
        assert isinstance(mock_agent.bus, MessageBus)
