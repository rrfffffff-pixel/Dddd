"""Base Agent class with optimization pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from coding_agent.core.context import SharedContext
from coding_agent.core.message import Message, MessageBus, MessageType
from coding_agent.core.tool import ToolRegistry
from coding_agent.intelligence.preprocessor import (
    CacheLayer,
    LexicalAnalyzer,
    PromptCompressor,
    TaskClassifier,
)
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
    token_budget: int = 0
    enable_preprocessing: bool = True


class Agent(ABC):
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
        self._llm_call_count = 0
        self._cache_hits = 0
        self._start_time = 0.0
        self.cache = CacheLayer()
        self.analyzer = LexicalAnalyzer()
        self.compressor = PromptCompressor()
        self.classifier = TaskClassifier()

    @property
    def name(self) -> str:
        return self.config.name

    def set_context(self, context: SharedContext) -> None:
        self.context = context

    @abstractmethod
    def get_system_prompt(self) -> str:
        ...

    def _preprocess_task(self, task: str) -> str | None:
        if not self.config.enable_preprocessing:
            return None
        classification = self.classifier.classify(task)
        if not classification.needs_llm and classification.confidence > 0.7:
            logger.info(f"[{self.name}] Static classification: {classification.suggested_agent} (confidence={classification.confidence})")
            return classification.static_answer
        return None

    def _build_messages(self, task: str) -> list[dict]:
        system_prompt = self.compressor.minify_system_prompt(self.get_system_prompt())
        messages = [{"role": "system", "content": system_prompt}]
        if self.context:
            ctx_text = self.context.summary()
            if ctx_text:
                messages.append({"role": "system", "content": f"Context:\n{ctx_text}"})
        messages.append({"role": "user", "content": task})
        return messages

    def _trim_messages(self, messages: list[dict], max_messages: int = 40) -> list[dict]:
        if len(messages) <= max_messages:
            return messages
        system_msgs = [m for m in messages if m["role"] == "system"]
        other_msgs = [m for m in messages if m["role"] != "system"]
        keep = other_msgs[-(max_messages - len(system_msgs)):]
        return system_msgs + keep

    def _execute_tool_with_retry(self, name: str, args: dict) -> str:
        cache_key = hashlib.md5(json.dumps({"tool": name, "args": args}, sort_keys=True).encode()).hexdigest()
        cached = self.cache.get_tool_cache(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        for attempt in range(self.config.max_tool_retries):
            try:
                result = self.tools.execute(name, **args)
                result_str = json.dumps(result, indent=2) if isinstance(result, dict | list) else str(result)
                result_str = self.compressor.compress_tool_result(result_str, name)
                self.cache.set_tool_cache(cache_key, result_str)
                return result_str
            except Exception as e:
                if attempt == self.config.max_tool_retries - 1:
                    return f"Error: {e}"
                time.sleep(0.3 * (attempt + 1))
        return "Error: tool failed"

    def run(self, task: str) -> str:
        provider = self.config.model_provider
        if provider is None:
            return "Error: No model provider configured"

        static_result = self._preprocess_task(task)
        if static_result:
            return static_result

        self._start_time = time.time()
        self._total_tokens = 0
        self._tool_call_count = 0
        self._llm_call_count = 0
        self._cache_hits = 0

        messages = self._build_messages(task)
        tools = self.tools.to_schemas()

        for iteration in range(self.config.max_iterations):
            messages = self._trim_messages(messages)

            prompt_hash = hashlib.md5(json.dumps(messages[-3:]).encode()).hexdigest()
            cached = self.cache.get_llm_cache(prompt_hash)
            if cached:
                self._cache_hits += 1
                response = LLMResponse(content=cached, model=provider.__class__.__name__)
            else:
                response = provider.chat(messages, tools if tools else None)
                self._llm_call_count += 1
                if response.content and not response.tool_calls:
                    self.cache.set_llm_cache(prompt_hash, response.content)

            if response.is_error:
                return response.content

            self._total_tokens += response.total_tokens

            assistant_msg: dict = {"role": "assistant", "content": response.content or ""}
            if response.tool_calls:
                assistant_msg["tool_calls"] = response.tool_calls
            messages.append(assistant_msg)

            if response.tool_calls:
                for tool_call in response.tool_calls:
                    fn = tool_call.get("function", {})
                    tool_name = fn.get("name", "")
                    raw_args = fn.get("arguments", "{}")
                    tool_call_id = tool_call.get("id", "")

                    # Parse arguments from string to dict
                    if isinstance(raw_args, str):
                        try:
                            tool_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            tool_args = {}
                    else:
                        tool_args = raw_args

                    self._tool_call_count += 1
                    result = self._execute_tool_with_retry(tool_name, tool_args)

                    if self.context and tool_name in ("write_file", "edit_file"):
                        path = tool_args.get("path", "")
                        if path and "Error" not in result:
                            self.context.add_file_written(path)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result[:6000],
                    })
            else:
                elapsed = time.time() - self._start_time
                logger.info(
                    f"[{self.name}] Done: {elapsed:.1f}s, "
                    f"{self._llm_call_count} LLM calls, "
                    f"{self._tool_call_count} tools, "
                    f"{self._cache_hits} cache hits, "
                    f"~{self._total_tokens} tokens"
                )
                return response.content

        elapsed = time.time() - self._start_time
        return f"Max iterations ({elapsed:.1f}s, {self._llm_call_count} LLM calls)"

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
        return "\n".join(
            f"- {t.name}({', '.join(p.name for p in t.parameters)}): {t.description}"
            for t in tools
        )
