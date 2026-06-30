"""Tests for the task orchestrator."""

from coding_agent.agents.code_agent import create_code_agent
from coding_agent.core.tool import ToolRegistry
from coding_agent.models.provider import MockProvider
from coding_agent.orchestrator.task_router import TaskOrchestrator


def test_orchestrator_creation():
    provider = MockProvider()
    tools = ToolRegistry()
    agents = {"code": create_code_agent(provider, tools)}
    orch = TaskOrchestrator(provider, agents)
    assert orch.context is not None


def test_orchestrator_decompose():
    # Mock returns a JSON array for decomposition
    decompose_response = '[{"id": 1, "description": "read files", "agent_name": "code"}]'
    provider = MockProvider(responses=[decompose_response, "Done reading"])
    tools = ToolRegistry()
    agents = {"code": create_code_agent(provider, tools)}
    orch = TaskOrchestrator(provider, agents)

    subtasks = orch.decompose_task("read the codebase")
    assert len(subtasks) == 1
    assert subtasks[0].agent_name == "code"


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

    # "run" triggers test agent selection, but since test agent isn't
    # available, it returns an error about missing agent
    result = orch.run_simple("run the test suite")
    assert "not available" in result.lower()
