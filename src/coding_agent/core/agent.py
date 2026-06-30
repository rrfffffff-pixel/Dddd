"""Base Agent class - all agents inherit from this."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from coding_agent.core.context import SharedContext
from coding_agent.core.message import Message, MessageBus, MessageType
from coding_agent.core.tool import ToolRegistry
from coding_agent.models.provider import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    name: str = "agent"
    system_prompt: str = ""
    max_iterations: int = 10
    model_provider: LLMProvider | None = None
    verbose: bool = False
    max_tool_retries: int = 2
    token_budget: int = 0  # 0 = unlimited


class Agent(ABC):
    """Base class for all coding agents."""

    def __init__(
        self,
        config: AgentConfig,
        tool_registry: ToolRegistry,
        message_bus: MessageBus | None = None,
    ) -> None:
        self.config = config
        self.tools = tool_registry
        self.bus = message_bus
        self.context: SharedContext | None = None
        self._total_tokens = 0
        self._tool_call_count = 0
        self._start_time = 0.0

    @property
    def name(self) -> str:
        return self.config.name

    def set_context(self, context: SharedContext) -> None:
        self.context = context

    @abstractmethod
    def get_system_prompt(self) -> str:
        ...

    def _build_messages(self, task: str) -> list[dict]:
        messages = [{"role": "system", "content": self.get_system_prompt()}]
        if self.context:
            ctx_text = self.context.summary()
            if ctx_text:
                messages.append({"role": "system", "content": f"Project Context:\n{ctx_text}"})
        messages.append({"role": "user", "content": task})
        return messages

    def _trim_messages(self, messages: list[dict], max_messages: int = 50) -> list[dict]:
        if len(messages) <= max_messages:
            return messages
        system_msgs = [m for m in messages if m["role"] == "system"]
        other_msgs = [m for m in messages if m["role"] != "system"]
        keep = other_msgs[-(max_messages - len(system_msgs)):]
        return system_msgs + keep

    def _execute_tool_with_retry(self, name: str, args: dict) -> str:
        for attempt in range(self.config.max_tool_retries):
            try:
                result = self.tools.execute(name, **args)
                if isinstance(result, dict | list):
                    return json.dumps(result, indent=2)
                return str(result)
            except Exception as e:
                if attempt == self.config.max_tool_retries - 1:
                    logger.error(f"Tool {name} failed after {self.config.max_tool_retries} attempts: {e}")
                    return f"Error: {e}"
                logger.warning(f"Tool {name} failed (attempt {attempt + 1}): {e}")
                time.sleep(0.5 * (attempt + 1))
        return f"Error: tool {name} failed"

    def run(self, task: str) -> str:
        provider = self.config.model_provider
        if provider is None:
            return "Error: No model provider configured"

        self._start_time = time.time()
        self._total_tokens = 0
        self._tool_call_count = 0

        messages = self._build_messages(task)
        tools = self.tools.to_schemas()

        for iteration in range(self.config.max_iterations):
            if self.config.verbose:
                elapsed = time.time() - self._start_time
                logger.info(f"[{self.name}] Iteration {iteration + 1}/{self.config.max_iterations} ({elapsed:.1f}s, {self._tool_call_count} tools)")

            messages = self._trim_messages(messages)
            response: LLMResponse = provider.chat(messages, tools if tools else None)

            if response.is_error:
                logger.error(f"[{self.name}] LLM error: {response.content}")
                return response.content

            self._total_tokens += response.usage.get("total_tokens", 0)

            assistant_msg: dict = {"role": "assistant", "content": response.content}
            if response.tool_calls:
                assistant_msg["tool_calls"] = response.tool_calls
            messages.append(assistant_msg)

            if response.tool_calls:
                for tool_call in response.tool_calls:
                    fn = tool_call.get("function", {})
                    tool_name = fn.get("name", "")
                    tool_args = fn.get("arguments", {})

                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}

                    self._tool_call_count += 1
                    logger.info(f"[{self.name}] Tool: {tool_name}")
                    result = self._execute_tool_with_retry(tool_name, tool_args)

                    if self.context and tool_name in ("write_file", "edit_file"):
                        path = tool_args.get("path", "")
                        if path and "Error" not in result:
                            self.context.add_file_written(path)

                    messages.append({
                        "role": "tool",
                        "content": result[:8000],
                    })
            else:
                elapsed = time.time() - self._start_time
                logger.info(f"[{self.name}] Done in {elapsed:.1f}s, {self._tool_call_count} tools, ~{self._total_tokens} tokens")
                return response.content

        logger.warning(f"[{self.name}] Max iterations reached")
        return f"Max iterations reached after {self._tool_call_count} tool calls. Last: {response.content[:200] if response else 'none'}"

    def send_message(self, receiver: str, content: str, msg_type: MessageType = MessageType.TASK) -> None:
        if self.bus is None:
            return
        msg = Message(sender=self.name, receiver=receiver, type=msg_type, content=content)
        self.bus.publish(msg)

    def receive_message(self) -> Message | None:
        if self.bus is None:
            return None
        return self.bus.consume(self.name)

    def get_tool_summary(self) -> str:
        tools = self.tools.list_tools()
        lines = []
        for tool in tools:
            params = ", ".join(f"{p.name}: {p.type}" for p in tool.parameters)
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines)
