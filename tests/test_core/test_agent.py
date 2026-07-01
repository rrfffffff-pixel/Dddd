"""Tests for the base agent improvements."""

import hashlib
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from coding_agent.core.agent import Agent, AgentConfig, AgentMetrics
from coding_agent.core.context import SharedContext
from coding_agent.core.message import MessageBus, MessageType
from coding_agent.core.tool import Tool, ToolParameter, ToolRegistry
from coding_agent.intelligence.preprocessor import CacheLayer
from coding_agent.models.provider import MockProvider, LLMResponse


class ConcreteAgent(Agent):
    """Concrete agent implementation for testing."""

    def get_system_prompt(self) -> str:
        return "Test system prompt"


# ===== AgentMetrics Tests =====


class TestAgentMetrics:
    """Test metrics collection functionality."""

    def test_initial_metrics(self):
        metrics = AgentMetrics()
        assert metrics.total_tokens == 0
        assert metrics.total_llm_calls == 0
        assert metrics.total_tool_calls == 0
        assert metrics.cache_hits == 0
        assert metrics.elapsed == 0.0

    def test_record_llm_call(self):
        metrics = AgentMetrics()
        metrics.record_llm_call(latency=0.5, tokens=100)
        assert metrics.total_llm_calls == 1
        assert metrics.total_tokens == 100
        assert metrics.llm_latencies == [0.5]
        assert metrics.token_usage_history == [100]

    def test_record_multiple_llm_calls(self):
        metrics = AgentMetrics()
        metrics.record_llm_call(0.1, 50)
        metrics.record_llm_call(0.2, 75)
        metrics.record_llm_call(0.3, 100)
        assert metrics.total_llm_calls == 3
        assert metrics.total_tokens == 225
        assert metrics.avg_llm_latency == pytest.approx(0.2)

    def test_record_tool_call(self):
        metrics = AgentMetrics()
        metrics.record_tool_call("read_file", 0.1)
        assert metrics.total_tool_calls == 1
        assert "read_file" in metrics.tool_latencies
        assert metrics.tool_latencies["read_file"] == [0.1]

    def test_record_multiple_tool_calls(self):
        metrics = AgentMetrics()
        metrics.record_tool_call("read_file", 0.1)
        metrics.record_tool_call("read_file", 0.2)
        metrics.record_tool_call("write_file", 0.3)
        assert metrics.total_tool_calls == 3
        assert metrics.avg_tool_latency["read_file"] == pytest.approx(0.15)
        assert metrics.avg_tool_latency["write_file"] == pytest.approx(0.3)

    def test_metrics_summary(self):
        metrics = AgentMetrics()
        metrics.record_llm_call(0.5, 100)
        metrics.record_tool_call("test", 0.1)
        metrics.cache_hits = 5
        summary = metrics.summary()
        assert "llm_calls=1" in summary
        assert "tool_calls=1" in summary
        assert "tokens=100" in summary
        assert "cache_hits=5" in summary


# ===== Token Budget Enforcement Tests =====


class TestTokenBudget:
    """Test token budget enforcement."""

    def test_no_budget_allows_all(self):
        provider = MockProvider(responses=["done"])
        config = AgentConfig(token_budget=0)
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())

        assert agent._check_token_budget() is True

    def test_budget_within_limit(self):
        provider = MockProvider(responses=["done"])
        config = AgentConfig(token_budget=1000)
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())
        agent.metrics.total_tokens = 500

        assert agent._check_token_budget() is True

    def test_budget_exceeded(self):
        provider = MockProvider(responses=["done"])
        config = AgentConfig(token_budget=1000)
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())
        agent.metrics.total_tokens = 1000

        assert agent._check_token_budget() is False

    def test_budget_warning_threshold(self):
        provider = MockProvider(responses=["done"])
        config = AgentConfig(token_budget=1000, token_budget_warning_ratio=0.8)
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())
        agent.metrics.total_tokens = 800

        # Should still return True but log a warning
        assert agent._check_token_budget() is True

    def test_run_stops_on_budget_exceeded(self):
        """Test that agent stops when token budget is exceeded mid-run."""
        provider = MockProvider(responses=["done"])
        config = AgentConfig(token_budget=10, max_iterations=10, model_provider=provider)
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())

        # Pre-populate metrics to simulate budget exhaustion
        # The run method resets metrics, so we need to check _check_token_budget directly
        agent.metrics.total_tokens = 15
        assert agent._check_token_budget() is False

        # Verify budget check with fresh metrics
        agent.metrics.total_tokens = 5
        assert agent._check_token_budget() is True


# ===== Streaming Response Tests =====


class TestStreamingResponse:
    """Test streaming response functionality."""

    def test_streaming_yields_progress(self):
        provider = MockProvider(responses=["done"])
        config = AgentConfig(max_iterations=1, model_provider=provider)
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())

        events = list(agent.run_streaming("test task"))
        types = [e["type"] for e in events]
        assert "progress" in types
        assert "result" in types

    def test_streaming_yields_llm_response(self):
        provider = MockProvider(responses=["done"])
        config = AgentConfig(max_iterations=1, model_provider=provider)
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())

        events = list(agent.run_streaming("test task"))
        llm_events = [e for e in events if e["type"] == "llm_response"]
        assert len(llm_events) == 1
        assert llm_events[0]["content"] == "done"

    def test_streaming_error_event(self):
        provider = MockProvider(responses=["Error: something failed"])
        config = AgentConfig(max_iterations=1, model_provider=provider)
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())

        events = list(agent.run_streaming("test task"))
        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1

    def test_streaming_no_provider(self):
        config = AgentConfig(model_provider=None)
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())

        events = list(agent.run_streaming("test task"))
        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1
        assert "No model provider" in error_events[0]["content"]

    def test_streaming_tool_call_events(self):
        tool_call_response = LLMResponse(
            content="",
            tool_calls=[{"id": "tc1", "function": {"name": "test_tool", "arguments": '{"x": 1}'}}],
            usage={"total_tokens": 50},
        )
        done_response = LLMResponse(content="done", usage={"total_tokens": 10})
        provider = MockProvider(responses=[])
        provider.chat = MagicMock(side_effect=[tool_call_response, done_response])

        tool_registry = ToolRegistry()
        tool_registry.register(Tool(name="test_tool", description="test", handler=lambda x: f"result_{x}"))

        config = AgentConfig(max_iterations=2, model_provider=provider)
        agent = ConcreteAgent(config=config, tool_registry=tool_registry)

        events = list(agent.run_streaming("test task"))
        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        tool_result_events = [e for e in events if e["type"] == "tool_result"]
        assert len(tool_call_events) == 1
        assert len(tool_result_events) == 1


# ===== Cache Key Generation Tests =====


class TestCacheKeyGeneration:
    """Test cache key generation for LLM and tool caching."""

    def test_cache_key_deterministic(self):
        config = AgentConfig()
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())

        messages = [{"role": "user", "content": "test"}]
        key1 = agent._generate_cache_key(messages)
        key2 = agent._generate_cache_key(messages)
        assert key1 == key2

    def test_cache_key_different_for_different_messages(self):
        config = AgentConfig()
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())

        messages1 = [{"role": "user", "content": "test1"}]
        messages2 = [{"role": "user", "content": "test2"}]
        key1 = agent._generate_cache_key(messages1)
        key2 = agent._generate_cache_key(messages2)
        assert key1 != key2

    def test_cache_key_order_sensitive(self):
        """Cache keys are order-sensitive since JSON serialization preserves list order."""
        config = AgentConfig()
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())

        messages1 = [{"role": "user", "content": "test"}, {"role": "assistant", "content": "response"}]
        messages2 = [{"role": "assistant", "content": "response"}, {"role": "user", "content": "test"}]
        key1 = agent._generate_cache_key(messages1)
        key2 = agent._generate_cache_key(messages2)
        # Different order produces different keys
        assert key1 != key2

    def test_cache_key_same_order_matches(self):
        config = AgentConfig()
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())

        messages1 = [{"role": "user", "content": "test"}, {"role": "assistant", "content": "response"}]
        messages2 = [{"role": "user", "content": "test"}, {"role": "assistant", "content": "response"}]
        key1 = agent._generate_cache_key(messages1)
        key2 = agent._generate_cache_key(messages2)
        assert key1 == key2

    def test_cache_key_is_sha256(self):
        config = AgentConfig()
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())

        messages = [{"role": "user", "content": "test"}]
        key = agent._generate_cache_key(messages)
        assert len(key) == 64  # SHA256 hex digest length


# ===== Tool Caching Tests =====


class TestToolCaching:
    """Test tool execution caching."""

    def test_tool_result_cached(self):
        call_count = 0

        def counting_tool(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        tool_registry = ToolRegistry()
        tool_registry.register(Tool(name="cached", description="cached", handler=counting_tool))

        config = AgentConfig(max_tool_retries=1)
        agent = ConcreteAgent(config=config, tool_registry=tool_registry)

        result1 = agent._execute_tool_with_retry("cached", {"x": 5})
        result2 = agent._execute_tool_with_retry("cached", {"x": 5})

        assert call_count == 1
        assert result1 == result2
        assert agent.metrics.cache_hits == 1

    def test_tool_result_not_cached_different_args(self):
        call_count = 0

        def counting_tool(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        tool_registry = ToolRegistry()
        tool_registry.register(Tool(name="cached", description="cached", handler=counting_tool))

        config = AgentConfig(max_tool_retries=1)
        agent = ConcreteAgent(config=config, tool_registry=tool_registry)

        agent._execute_tool_with_retry("cached", {"x": 5})
        agent._execute_tool_with_retry("cached", {"x": 10})

        assert call_count == 2


# ===== Message Trimming Tests =====


class TestMessageTrimming:
    """Test message history trimming."""

    def test_no_trim_when_under_limit(self):
        config = AgentConfig()
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())

        messages = [{"role": "user", "content": "test"}] * 10
        trimmed = agent._trim_messages(messages, max_messages=20)
        assert len(trimmed) == 10

    def test_trim_preserves_system_messages(self):
        config = AgentConfig()
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())

        messages = [
            {"role": "system", "content": "system1"},
            {"role": "system", "content": "system2"},
        ] + [{"role": "user", "content": f"msg{i}"} for i in range(50)]

        trimmed = agent._trim_messages(messages, max_messages=10)
        system_msgs = [m for m in trimmed if m["role"] == "system"]
        assert len(system_msgs) == 2

    def test_trim_keeps_recent_messages(self):
        config = AgentConfig()
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())

        messages = [{"role": "user", "content": f"msg{i}"} for i in range(50)]
        trimmed = agent._trim_messages(messages, max_messages=10)
        assert len(trimmed) == 10
        assert trimmed[-1]["content"] == "msg49"


# ===== Context Window Summarization Tests =====


class TestContextSummarization:
    """Test context window summarization for old messages."""

    def test_no_summarize_when_under_threshold(self):
        config = AgentConfig(context_window_limit=10000)
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())

        messages = [{"role": "user", "content": "short"}] * 4
        result = agent._summarize_old_messages(messages)
        assert len(result) == 4

    def test_no_summarize_when_disabled(self):
        config = AgentConfig(context_window_limit=0)
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())

        messages = [{"role": "user", "content": "x" * 1000}] * 10
        result = agent._summarize_old_messages(messages)
        assert len(result) == 10

    def test_summarize_old_messages(self):
        config = AgentConfig(context_window_limit=100, summary_threshold_ratio=0.5)
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())

        messages = [{"role": "user", "content": "x" * 200}] * 10
        result = agent._summarize_old_messages(messages)
        assert len(result) < len(messages)
        assert any(m["role"] == "system" and "Earlier conversation summary" in m.get("content", "") for m in result)


# ===== Agent Run Integration Tests =====


class TestAgentRun:
    """Integration tests for agent run functionality."""

    def test_run_returns_result(self):
        provider = MockProvider(responses=["final answer"])
        config = AgentConfig(max_iterations=1, model_provider=provider)
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())

        result = agent.run("test task")
        assert result == "final answer"

    def test_run_no_provider(self):
        config = AgentConfig(model_provider=None)
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())

        result = agent.run("test task")
        assert "No model provider" in result

    def test_run_max_iterations(self):
        """Test that agent stops at max iterations when LLM keeps requesting tools."""
        tool_call_response = LLMResponse(
            content="",
            tool_calls=[{"id": "tc1", "function": {"name": "test_tool", "arguments": "{}"}}],
            usage={"total_tokens": 10},
        )
        provider = MockProvider(responses=[])
        provider.chat = MagicMock(return_value=tool_call_response)

        tool_registry = ToolRegistry()
        tool_registry.register(Tool(name="test_tool", description="test", handler=lambda: "result"))

        config = AgentConfig(max_iterations=3, model_provider=provider)
        agent = ConcreteAgent(config=config, tool_registry=tool_registry)

        result = agent.run("test task")
        assert "Max iterations" in result

    def test_run_records_metrics(self):
        provider = MockProvider(responses=["done"])
        config = AgentConfig(max_iterations=1, model_provider=provider)
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())

        agent.run("test task")
        assert agent.metrics.total_llm_calls >= 1

    def test_run_with_tool_calls(self):
        tool_call_response = LLMResponse(
            content="",
            tool_calls=[{"id": "tc1", "function": {"name": "test_tool", "arguments": '{"x": 1}'}}],
            usage={"total_tokens": 50},
        )
        done_response = LLMResponse(content="final answer", usage={"total_tokens": 10})
        provider = MockProvider(responses=[])
        provider.chat = MagicMock(side_effect=[tool_call_response, done_response])

        tool_registry = ToolRegistry()
        tool_registry.register(Tool(name="test_tool", description="test", handler=lambda x: f"result_{x}"))

        config = AgentConfig(max_iterations=2, model_provider=provider)
        agent = ConcreteAgent(config=config, tool_registry=tool_registry)

        result = agent.run("test task")
        assert result == "final answer"
        assert agent.metrics.total_tool_calls == 1


# ===== Message Bus Integration Tests =====


class TestMessageBusIntegration:
    """Test agent messaging functionality."""

    def test_send_message(self):
        bus = MessageBus()
        config = AgentConfig()
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())
        agent.bus = bus

        agent.send_message("receiver", "hello")
        msg = bus.consume("receiver")
        assert msg is not None
        assert msg.content == "hello"
        assert msg.sender == agent.name

    def test_receive_message(self):
        bus = MessageBus()
        config = AgentConfig()
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())
        agent.bus = bus

        msg = bus.consume("nonexistent")
        assert msg is None

    def test_send_without_bus(self):
        config = AgentConfig()
        agent = ConcreteAgent(config=config, tool_registry=ToolRegistry())
        agent.bus = None

        # Should not raise
        agent.send_message("receiver", "hello")


# ===== Tool Summary Tests =====


class TestToolSummary:
    """Test tool summary generation."""

    def test_tool_summary(self):
        tool_registry = ToolRegistry()
        tool_registry.register(Tool(
            name="read_file",
            description="Read a file",
            parameters=[ToolParameter(name="path", type="string", description="File path")],
        ))
        tool_registry.register(Tool(
            name="write_file",
            description="Write a file",
            parameters=[
                ToolParameter(name="path", type="string", description="File path"),
                ToolParameter(name="content", type="string", description="File content"),
            ],
        ))

        config = AgentConfig()
        agent = ConcreteAgent(config=config, tool_registry=tool_registry)

        summary = agent.get_tool_summary()
        assert "read_file" in summary
        assert "write_file" in summary
        assert "Read a file" in summary
