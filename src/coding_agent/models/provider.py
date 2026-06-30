"""Unified LLM interface supporting multiple providers."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def is_error(self) -> bool:
        return self.content.startswith("Error:")

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0) or (
            self.usage.get("input_tokens", 0) + self.usage.get("output_tokens", 0)
        )


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        ...

    def chat_simple(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages).content


class OllamaProvider(LLMProvider):
    def __init__(
        self,
        model: str = "qwen3:1.7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_retries: int = 3,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if tools:
            payload["tools"] = tools

        last_error = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.json()

                message = data.get("message", {})
                content = message.get("content", "")
                tool_calls = message.get("tool_calls", [])

                return LLMResponse(
                    content=content,
                    tool_calls=tool_calls,
                    usage={"total_tokens": data.get("eval_count", 0)},
                    model=self.model,
                    finish_reason=data.get("done_reason", ""),
                )
            except requests.exceptions.ConnectionError as e:
                last_error = e
                logger.warning(f"Ollama connection failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
            except requests.exceptions.Timeout:
                logger.warning(f"Ollama timeout (attempt {attempt + 1})")
                last_error = Exception("Request timed out")
            except Exception as e:
                last_error = e
                logger.error(f"Ollama error: {e}")
                break

        return LLMResponse(content=f"Error: {last_error}", model=self.model)


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        max_retries: int = 3,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        payload: dict = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        last_error = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.json()

                choice = data["choices"][0]
                message = choice["message"]

                tool_calls = []
                for tc in message.get("tool_calls", []):
                    fn = tc["function"]
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    tool_calls.append({
                        "function": {
                            "name": fn["name"],
                            "arguments": args,
                        }
                    })

                return LLMResponse(
                    content=message.get("content", ""),
                    tool_calls=tool_calls,
                    usage=data.get("usage", {}),
                    model=self.model,
                    finish_reason=choice.get("finish_reason", ""),
                )
            except requests.exceptions.ConnectionError as e:
                last_error = e
                logger.warning(f"OpenAI connection failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
            except requests.exceptions.Timeout:
                logger.warning(f"OpenAI timeout (attempt {attempt + 1})")
                last_error = Exception("Request timed out")
            except Exception as e:
                last_error = e
                logger.error(f"OpenAI error: {e}")
                break

        return LLMResponse(content=f"Error: {last_error}", model=self.model)


class AnthropicProvider(LLMProvider):
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str = "",
        max_retries: int = 3,
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        system_msg = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg += m["content"] + "\n"
            else:
                user_messages.append(m)

        if not user_messages:
            user_messages = [{"role": "user", "content": "Hello"}]

        payload: dict = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": user_messages,
        }
        if system_msg.strip():
            payload["system"] = system_msg.strip()

        if tools:
            anthropic_tools = []
            for t in tools:
                fn = t.get("function", {})
                anthropic_tools.append({
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {}),
                })
            payload["tools"] = anthropic_tools

        last_error = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                r.raise_for_status()
                data = r.json()

                content = ""
                tool_calls = []
                for block in data.get("content", []):
                    if block["type"] == "text":
                        content += block["text"]
                    elif block["type"] == "tool_use":
                        tool_calls.append({
                            "function": {
                                "name": block["name"],
                                "arguments": block.get("input", {}),
                            }
                        })

                usage = data.get("usage", {})
                return LLMResponse(
                    content=content,
                    tool_calls=tool_calls,
                    usage={
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                    },
                    model=self.model,
                    finish_reason=data.get("stop_reason", ""),
                )
            except requests.exceptions.ConnectionError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
            except requests.exceptions.Timeout:
                last_error = Exception("Request timed out")
            except Exception as e:
                last_error = e
                logger.error(f"Anthropic error: {e}")
                break

        return LLMResponse(content=f"Error: {last_error}", model=self.model)


class MockProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or ["Done"]
        self.call_count = 0
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools})
        idx = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return LLMResponse(content=self.responses[idx], model="mock")


def create_provider(provider: str = "ollama", **kwargs) -> LLMProvider:
    if provider == "ollama":
        return OllamaProvider(
            model=kwargs.get("model", "qwen3:1.7b"),
            base_url=kwargs.get("base_url", "http://localhost:11434"),
            temperature=kwargs.get("temperature", 0.1),
        )
    elif provider == "openai":
        return OpenAIProvider(
            model=kwargs.get("model", "gpt-4o"),
            api_key=kwargs.get("api_key", ""),
            base_url=kwargs.get("base_url", "https://api.openai.com/v1"),
        )
    elif provider == "anthropic":
        return AnthropicProvider(
            model=kwargs.get("model", "claude-sonnet-4-20250514"),
            api_key=kwargs.get("api_key", ""),
        )
    elif provider == "mock":
        return MockProvider(responses=kwargs.get("responses"))
    else:
        raise ValueError(f"Unknown provider: {provider}")
