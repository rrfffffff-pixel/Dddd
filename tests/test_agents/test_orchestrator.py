"""Tests for the task orchestrator."""

from coding_agent.agents.code_agent import create_code_agent
from coding_agent.core.tool import ToolRegistry
from coding_agent.models.provider import MockProvider
from coding_agent.orchestrator.task_router import TaskOrchestrator, Subtask


def test_orchestrator_creation():
    provider = MockProvider()
    tools = ToolRegistry()
    agents = {"code": create_code_agent(provider, tools)}
    orch = TaskOrchestrator(provider, agents)
    assert orch.context is not None


def test_orchestrator_decompose():
    decompose_response = '[{"id": 1, "description": "read files", "agent_name": "code", "depends_on": []}]'
    provider = MockProvider(responses=[decompose_response, "Done reading"])
    tools = ToolRegistry()
    agents = {"code": create_code_agent(provider, tools)}
    orch = TaskOrchestrator(provider, agents)

    subtasks = orch.decompose_task("read the codebase")
    assert len(subtasks) == 1
    assert subtasks[0].agent_name == "code"
    assert subtasks[0].depends_on == []


def test_orchestrator_run_simple():
    provider = MockProvider(responses=["Task complete"])
    tools = ToolRegistry()
    agents = {"code": create_code_agent(provider, tools)}
    orch = TaskOrchestrator(provider, agents)

    result = orch.run_simple("do something simple")
    assert result == "Task complete"


def test_orchestrator_run_simple_fallback():
    provider = MockProvider(responses=["Done"])
    tools = ToolRegistry()
    agents = {"code": create_code_agent(provider, tools)}
    orch = TaskOrchestrator(provider, agents)

    result = orch.run_simple("run the test suite")
    assert "not available" in result.lower()


def test_orchestrator_dependency_order():
    provider = MockProvider()
    tools = ToolRegistry()
    agents = {"code": create_code_agent(provider, tools)}
    orch = TaskOrchestrator(provider, agents)

    subtasks = [
        Subtask(id="1", description="explore", agent_name="code", depends_on=[]),
        Subtask(id="2", description="edit", agent_name="code", depends_on=["1"]),
        Subtask(id="3", description="test", agent_name="code", depends_on=["2"]),
    ]

    ready = orch._get_ready_tasks(subtasks)
    assert len(ready) == 1
    assert ready[0].id == "1"

    subtasks[0].status = "done"
    ready = orch._get_ready_tasks(subtasks)
    assert len(ready) == 1
    assert ready[0].id == "2"

    subtasks[1].status = "done"
    ready = orch._get_ready_tasks(subtasks)
    assert len(ready) == 1
    assert ready[0].id == "3"


def test_should_retry():
    provider = MockProvider()
    tools = ToolRegistry()
    agents = {"code": create_code_agent(provider, tools)}
    orch = TaskOrchestrator(provider, agents)

    subtask = Subtask(id="1", description="test", agent_name="code")
    assert orch._should_retry(subtask, "Error: file not found") is True
    assert orch._should_retry(subtask, "Success") is False

    subtask.retry_count = 1
    assert orch._should_retry(subtask, "Error: failed") is False
