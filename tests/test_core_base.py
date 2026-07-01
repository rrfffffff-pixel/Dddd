"""Tests for core components."""

import json
from coding_agent.core.agent import Agent, AgentConfig
from coding_agent.core.context import SharedContext, FileChange, CodeSymbol
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
    assert "x" in schema["function"]["parameters"]["required"]
    assert "y" not in schema["function"]["parameters"]["required"]


def test_tool_execute():
    def add(a: int, b: int) -> int:
        return a + b
    tool = Tool(name="add", description="Add numbers", handler=add)
    assert tool.execute(a=2, b=3) == 5


def test_tool_validate():
    tool = Tool(
        name="test",
        description="test",
        parameters=[ToolParameter(name="x", type="string", description="x", required=True)],
    )
    errors = tool.validate({})
    assert len(errors) == 1
    assert "x" in errors[0]

    errors = tool.validate({"x": "hi"})
    assert len(errors) == 0


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
        assert False
    except ValueError:
        pass


def test_registry_cache():
    call_count = 0
    def counting(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * 2

    registry = ToolRegistry()
    registry.register(Tool(name="cached", description="cached", handler=counting, cacheable=True))
    registry.execute("cached", x=5)
    registry.execute("cached", x=5)
    assert call_count == 1


def test_message_bus():
    bus = MessageBus()
    msg = Message(sender="a", receiver="b", content="hello")
    bus.publish(msg)
    received = bus.consume("b")
    assert received is not None
    assert received.content == "hello"


def test_message_bus_empty():
    bus = MessageBus()
    assert bus.consume("nobody") is None


def test_shared_context():
    ctx = SharedContext(task_description="test task")
    ctx.add_file_read("a.py")
    ctx.add_file_read("a.py")
    ctx.add_file_written("b.py")
    ctx.add_error("something went wrong")
    ctx.add_symbol(CodeSymbol(name="main", kind="function", file="a.py"))
    ctx.add_decision("use asyncio")

    assert len(ctx.files_read) == 1
    assert len(ctx.symbols) == 1
    assert ctx.find_symbol("main") is not None
    assert ctx.find_symbol("missing") is None
    summary = ctx.summary()
    assert "test task" in summary
    assert "main" in summary


def test_llm_response():
    resp = LLMResponse(content="hello", tool_calls=[{"function": {"name": "test"}}], usage={"total_tokens": 100})
    assert resp.has_tool_calls is True
    assert resp.is_error is False
    assert resp.total_tokens == 100

    err_resp = LLMResponse(content="Error: failed")
    assert err_resp.is_error is True


def test_mock_provider():
    provider = MockProvider(responses=["hello", "world"])
    assert provider.chat_simple("hi") == "hello"
    assert provider.chat_simple("again") == "world"
    assert provider.call_count == 2
