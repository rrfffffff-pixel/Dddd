"""Base Agent class - all agents inherit from this."""

from __future__ import annotations

import json
import logging
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

    @property
    def name(self) -> str:
        return self.config.name

    def set_context(self, context: SharedContext) -> None:
        self.context = context

    @abstractmethod
    def get_system_prompt(self) -> str:
        ...

    def run(self, task: str) -> str:
        """Run the agent on a task. Returns the final result."""
        provider = self.config.model_provider
        if provider is None:
            return "Error: No model provider configured"

        messages = [
            {"role": "system", "content": self.get_system_prompt()},
        ]
        if self.context:
            messages.append({
                "role": "system",
                "content": f"Context:\n{self.context.summary()}",
            })
        messages.append({"role": "user", "content": task})

        tools = self.tools.to_schemas()
        tool_results = []

        for iteration in range(self.config.max_iterations):
            if self.config.verbose:
                logger.info(f"[{self.name}] Iteration {iteration + 1}/{self.config.max_iterations}")

            response: LLMResponse = provider.chat(messages, tools if tools else None)

            if response.is_error:
                logger.error(f"[{self.name}] LLM error: {response.content}")
                return response.content

            # Add assistant response to history
            assistant_msg: dict = {"role": "assistant", "content": response.content}
            if response.tool_calls:
                assistant_msg["tool_calls"] = response.tool_calls
            messages.append(assistant_msg)

            # Handle tool calls
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

                    logger.info(f"[{self.name}] Calling tool: {tool_name}")
                    result = self._execute_tool(tool_name, tool_args)
                    tool_results.append({"tool": tool_name, "result": result[:500]})

                    messages.append({
                        "role": "tool",
                        "content": json.dumps(result) if not isinstance(result, str) else result,
                    })
            else:
                # No tool calls - agent is done
                return response.content

        logger.warning(f"[{self.name}] Max iterations ({self.config.max_iterations}) reached")
        return f"Max iterations reached. Last response: {response.content if response else 'none'}"

    def _execute_tool(self, name: str, args: dict) -> str:
        """Execute a tool and return the result as a string."""
        try:
            result = self.tools.execute(name, **args)
            if isinstance(result, dict | list):
                return json.dumps(result, indent=2)
            return str(result)
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return f"Error executing {name}: {e}"

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
        """Get a summary of available tools for the system prompt."""
        tools = self.tools.list_tools()
        lines = []
        for tool in tools:
            params = ", ".join(f"{p.name}: {p.type}" for p in tool.parameters)
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines)
