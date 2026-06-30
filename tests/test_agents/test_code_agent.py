"""Tests for the code agent."""

import tempfile
from pathlib import Path

from coding_agent.agents.code_agent import create_code_agent
from coding_agent.core.context import SharedContext
from coding_agent.core.tool import ToolRegistry
from coding_agent.models.provider import MockProvider
from coding_agent.tools.file_ops import register_file_tools


def test_code_agent_creation():
    provider = MockProvider()
    tools = ToolRegistry()
    agent = create_code_agent(provider, tools)
    assert agent.name == "code"


def test_code_agent_has_tools():
    provider = MockProvider()
    tools = ToolRegistry()
    with tempfile.TemporaryDirectory() as tmpdir:
        register_file_tools(tools, tmpdir)
        agent = create_code_agent(provider, tools, tmpdir)
        tool_names = [t.name for t in agent.tools.list_tools()]
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "edit_file" in tool_names


def test_code_agent_runs():
    provider = MockProvider(responses=["I have read the file and it looks good."])
    tools = ToolRegistry()
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "test.py").write_text("x = 1")
        register_file_tools(tools, tmpdir)
        agent = create_code_agent(provider, tools, tmpdir)

        result = agent.run("Read test.py")
        assert "good" in result.lower() or "read" in result.lower()
        assert provider.call_count == 1


def test_code_agent_with_context():
    provider = MockProvider(responses=["Done"])
    tools = ToolRegistry()
    with tempfile.TemporaryDirectory() as tmpdir:
        register_file_tools(tools, tmpdir)
        agent = create_code_agent(provider, tools, tmpdir)

        ctx = SharedContext(task_description="test task")
        ctx.add_file_read("main.py")
        agent.set_context(ctx)

        result = agent.run("check main.py")
        assert result == "Done"
