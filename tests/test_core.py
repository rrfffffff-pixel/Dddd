"""Tests for core components."""

import json
import tempfile
from pathlib import Path

from coding_agent.core.agent import Agent, AgentConfig
from coding_agent.core.context import SharedContext, FileChange
from coding_agent.core.message import Message, MessageBus, MessageType
from coding_agent.core.tool import Tool, ToolParameter, ToolRegistry
from coding_agent.models.provider import MockProvider, LLMResponse


def test_tool_schema():
    tool = Tool(
        name="test_tool",
        description="A test tool",
        parameters=[
            ToolParameter(name="x", type="string", description="Input"),
            ToolParameter(name="y", type="integer", description="Count", required=False, default=5),
        ],
    )
    schema = tool.to_schema()
    assert schema["function"]["name"] == "test_tool"
    assert "x" in schema["function"]["parameters"]["properties"]
    assert "x" in schema["function"]["parameters"]["required"]
    assert "y" not in schema["function"]["parameters"]["required"]


def test_tool_execute():
    def add(a: int, b: int) -> int:
        return a + b

    tool = Tool(name="add", description="Add numbers", handler=add)
    assert tool.execute(a=2, b=3) == 5


def test_registry():
    registry = ToolRegistry()
    tool = Tool(name="echo", description="Echo input", handler=lambda text: text)
    registry.register(tool)

    assert registry.get("echo") is not None
    assert registry.execute("echo", text="hello") == "hello"
    assert len(registry.list_tools()) == 1


def test_registry_merge():
    r1 = ToolRegistry()
    r1.register(Tool(name="a", description="A"))
    r2 = ToolRegistry()
    r2.register(Tool(name="b", description="B"))

    r1.merge(r2)
    assert len(r1.list_tools()) == 2


def test_registry_unknown_tool():
    registry = ToolRegistry()
    try:
        registry.execute("nonexistent")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent" in str(e)


def test_message_bus():
    bus = MessageBus()
    msg = Message(sender="a", receiver="b", content="hello")
    bus.publish(msg)

    received = bus.consume("b")
    assert received is not None
    assert received.content == "hello"
    assert received.sender == "a"


def test_message_bus_empty():
    bus = MessageBus()
    assert bus.consume("nobody") is None


def test_shared_context():
    ctx = SharedContext(task_description="test task")
    ctx.add_file_read("a.py")
    ctx.add_file_read("a.py")  # duplicate
    ctx.add_file_written("b.py", content="x = 1")
    ctx.add_error("something went wrong")

    assert len(ctx.files_read) == 1
    assert len(ctx.files_written) == 1
    assert len(ctx.changes) == 1
    assert len(ctx.errors) == 1
    assert "test task" in ctx.summary()


def test_llm_response():
    resp = LLMResponse(content="hello", tool_calls=[{"function": {"name": "test"}}])
    assert resp.has_tool_calls is True
    assert resp.is_error is False

    err_resp = LLMResponse(content="Error: something failed")
    assert err_resp.is_error is True


def test_mock_provider():
    provider = MockProvider(responses=["hello", "world"])
    assert provider.chat_simple("hi") == "hello"
    assert provider.chat_simple("hi again") == "world"
    assert provider.call_count == 2
    assert len(provider.calls) == 2
