"""Base Agent class with optimization pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

from coding_agent.core.checkpoint import CheckpointManager
from coding_agent.core.context import SharedContext
from coding_agent.core.message import Message, MessageBus, MessageType
from coding_agent.core.tool import ToolRegistry
from coding_agent.intelligence.mentions import expand_mentions
from coding_agent.intelligence.preprocessor import (
    CacheLayer,
    LexicalAnalyzer,
    PromptCompressor,
    TaskClassifier,
)
from coding_agent.intelligence.repomap import RepoMap
from coding_agent.intelligence.rules import load_rules, rules_summary
from coding_agent.intelligence.validator import ShadowValidator
from coding_agent.models.provider import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    llm_latencies: list[float] = field(default_factory=list)
    tool_latencies: dict[str, list[float]] = field(default_factory=dict)
    token_usage_history: list[int] = field(default_factory=list)
    total_tokens: int = 0
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    cache_hits: int = 0
    start_time: float = 0.0

    @property
    def elapsed(self) -> float:
        if self.start_time <= 0:
            return 0.0
        return time.time() - self.start_time

    @property
    def avg_llm_latency(self) -> float:
        return sum(self.llm_latencies) / len(self.llm_latencies) if self.llm_latencies else 0.0

    @property
    def avg_tool_latency(self) -> dict[str, float]:
        return {
            name: sum(times) / len(times)
            for name, times in self.tool_latencies.items()
            if times
        }

    def record_llm_call(self, latency: float, tokens: int) -> None:
        self.llm_latencies.append(latency)
        self.token_usage_history.append(tokens)
        self.total_tokens += tokens
        self.total_llm_calls += 1

    def record_tool_call(self, tool_name: str, latency: float) -> None:
        self.tool_latencies.setdefault(tool_name, []).append(latency)
        self.total_tool_calls += 1

    def summary(self) -> str:
        return (
            f"elapsed={self.elapsed:.1f}s, "
            f"llm_calls={self.total_llm_calls}, "
            f"tool_calls={self.total_tool_calls}, "
            f"tokens={self.total_tokens}, "
            f"cache_hits={self.cache_hits}, "
            f"avg_llm_latency={self.avg_llm_latency:.2f}s"
        )


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
    token_budget_warning_ratio: float = 0.8
    context_window_limit: int = 0
    summary_threshold_ratio: float = 0.75


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
        self.metrics = AgentMetrics()
        self.cache = CacheLayer()
        self.analyzer = LexicalAnalyzer()
        self.compressor = PromptCompressor()
        self.classifier = TaskClassifier()
        self.checkpoints = CheckpointManager()
        self.validator = ShadowValidator()
        self._rules: list = []
        self._iteration_timeout: float = 120.0
        self._repo_map: RepoMap | None = None

    def load_project_rules(self, project_root: str) -> None:
        self._rules = load_rules(project_root)

    @property
    def name(self) -> str:
        return self.config.name

    def set_context(self, context: SharedContext) -> None:
        self.context = context

    def set_repo_map(self, repo_map: RepoMap) -> None:
        self._repo_map = repo_map

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

        # Inject project rules
        if self._rules:
            summary = rules_summary(self._rules)
            messages.append({"role": "system", "content": f"Project rules:\n{summary}"})

        # Inject shared context
        if self.context:
            ctx_text = self.context.summary()
            if ctx_text:
                messages.append({"role": "system", "content": f"Context:\n{ctx_text}"})

        # Inject repo map for code awareness
        if self._repo_map:
            project_root = self.context.project_root if self.context else "."
            code_files = self._find_source_files(project_root)
            repo_content = self._repo_map.get_repo_map(
                other_files=code_files,
            )
            if repo_content:
                messages.append({"role": "system", "content": f"Repository codebase map:\n{repo_content}"})

        # Expand @mentions (like Cursor's @file, @folder, @codebase)
        expanded = expand_mentions(task, self.context.project_root if self.context else ".")
        messages.append({"role": "user", "content": expanded})
        return messages

    def _find_source_files(self, root: str, max_files: int = 200) -> list[str]:
        files = []
        root_path = Path(root).resolve()
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirpath_obj = Path(dirpath)
            # Skip hidden dirs and common non-source dirs
            dirnames[:] = [d for d in dirnames
                          if not d.startswith(".") and d not in ("node_modules", "__pycache__",
                                                                 "venv", ".venv", "env", "dist", "build")]
            for f in filenames:
                ext = os.path.splitext(f)[1].lower()
                if ext in (".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java",
                          ".c", ".cpp", ".h", ".hpp", ".rb", ".php", ".swift", ".kt"):
                    files.append(str(dirpath_obj / f))
                    if len(files) >= max_files:
                        return files
        return files

    def _trim_messages(self, messages: list[dict], max_messages: int = 24) -> list[dict]:
        if len(messages) <= max_messages:
            return messages
        system_msgs = [m for m in messages if m["role"] == "system"]
        other_msgs = [m for m in messages if m["role"] != "system"]
        keep = other_msgs[-(max_messages - len(system_msgs)):]
        return system_msgs + keep

    def _generate_cache_key(self, messages: list[dict]) -> str:
        serialized = json.dumps(messages, sort_keys=False, ensure_ascii=False)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def _check_token_budget(self) -> bool:
        if self.config.token_budget <= 0:
            return True
        remaining = self.config.token_budget - self.metrics.total_tokens
        threshold = self.config.token_budget * self.config.token_budget_warning_ratio
        if remaining <= 0:
            logger.warning(f"[{self.name}] Token budget exhausted: {self.metrics.total_tokens}/{self.config.token_budget}")
            return False
        if self.metrics.total_tokens >= threshold:
            logger.warning(
                f"[{self.name}] Approaching token budget: "
                f"{self.metrics.total_tokens}/{self.config.token_budget} "
                f"({self.metrics.total_tokens / self.config.token_budget:.0%})"
            )
        return True

    def _summarize_old_messages(self, messages: list[dict]) -> list[dict]:
        if self.config.context_window_limit <= 0:
            return messages
        system_msgs = [m for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]
        estimated_tokens = sum(len(json.dumps(m)) // 4 for m in messages)
        threshold = int(self.config.context_window_limit * self.config.summary_threshold_ratio)
        if estimated_tokens < threshold or len(non_system) <= 6:
            return messages

        # Keep last 6 messages (2 full turns), summarize the rest
        keep_last = 6
        old_messages = non_system[:-keep_last]
        recent_messages = non_system[-keep_last:] if len(non_system) >= keep_last else non_system

        summary_parts = []
        for msg in old_messages:
            role = msg.get("role", "unknown")
            if role == "user":
                content = (msg.get("content") or "")[:150]
                summary_parts.append(f"User: {content}")
            elif role == "assistant":
                content = (msg.get("content") or "")[:150]
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    tool_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
                    summary_parts.append(f"Assistant: used tools [{', '.join(tool_names)}]")
                elif content:
                    summary_parts.append(f"Assistant: {content}")
            elif role == "tool":
                summary_parts.append("[Tool result omitted from summary]")

        summary_text = "Earlier conversation summary:\n" + "\n".join(summary_parts[:15])
        return system_msgs + [{"role": "system", "content": summary_text}] + recent_messages

    def _execute_tool_with_retry(self, name: str, args: dict) -> str:
        cache_key_input = json.dumps({"tool": name, "args": args}, sort_keys=True)
        cache_key = hashlib.sha256(cache_key_input.encode()).hexdigest()
        cached = self.cache.get_tool_cache(cache_key)
        if cached is not None:
            self.metrics.cache_hits += 1
            return cached

        for attempt in range(self.config.max_tool_retries):
            tool_start = time.monotonic()
            try:
                result = self.tools.execute(name, **args)
                tool_latency = time.monotonic() - tool_start
                self.metrics.record_tool_call(name, tool_latency)
                result_str = json.dumps(result, indent=2) if isinstance(result, dict | list) else str(result)
                result_str = self.compressor.compress_tool_result(result_str, name)
                self.cache.set_tool_cache(cache_key, result_str)
                return result_str
            except Exception as e:
                if attempt == self.config.max_tool_retries - 1:
                    tool_latency = time.monotonic() - tool_start
                    self.metrics.record_tool_call(name, tool_latency)
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

        self.metrics = AgentMetrics(start_time=time.monotonic())

        messages = self._build_messages(task)
        tools = self.tools.to_schemas()

        for iteration in range(self.config.max_iterations):
            iteration_start = time.monotonic()

            # Save checkpoint periodically
            if iteration > 0 and iteration % 3 == 0:
                ctx = self.context
                self.checkpoints.save(
                    task=task, iteration=iteration, messages=messages,
                    files_read=list(ctx.files_read) if ctx else [],
                    files_written=list(ctx.files_written) if ctx else [],
                    decisions=list(ctx.decisions) if ctx else [],
                )

            messages = self._trim_messages(messages)
            messages = self._summarize_old_messages(messages)

            if not self._check_token_budget():
                elapsed = self.metrics.elapsed
                return f"Token budget exceeded ({elapsed:.1f}s, {self.metrics.total_tokens} tokens used)"

            if time.monotonic() - iteration_start > self._iteration_timeout:
                logger.warning(f"[{self.name}] Iteration {iteration + 1} timed out")
                break

            prompt_hash = self._generate_cache_key(messages)
            cached = self.cache.get_llm_cache(prompt_hash)
            if cached:
                self.metrics.cache_hits += 1
                response = LLMResponse(content=cached, model=provider.__class__.__name__)
            else:
                llm_start = time.monotonic()
                response = provider.chat(messages, tools if tools else None)
                llm_latency = time.monotonic() - llm_start
                self.metrics.record_llm_call(llm_latency, response.total_tokens)
                if response.content and not response.tool_calls:
                    self.cache.set_llm_cache(prompt_hash, response.content)

            if time.monotonic() - iteration_start > self._iteration_timeout:
                logger.warning(f"[{self.name}] Iteration {iteration + 1} timed out after LLM call")
                break

            if response.is_error:
                return response.content

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

                    if isinstance(raw_args, str):
                        try:
                            tool_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            tool_args = {}
                    else:
                        tool_args = raw_args

                    result = self._execute_tool_with_retry(tool_name, tool_args)

                    if self.context:
                        path = tool_args.get("path", "")
                        if path and "Error" not in result:
                            if tool_name in ("write_file", "edit_file"):
                                self.context.add_file_written(path)
                                # Shadow validation on edited files
                                if tool_name == "edit_file":
                                    new_str = tool_args.get("new_string", "")
                                    validation = self.validator.validate(path, new_str)
                                    if validation:
                                        result += f"\n[Validation: {validation}]"
                            elif tool_name == "read_file":
                                self.context.add_file_read(path)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result[:6000],
                    })

                    if time.monotonic() - iteration_start > self._iteration_timeout:
                        logger.warning(f"[{self.name}] Iteration {iteration + 1} timed out during tool execution")
                        break

                # Reflection loop: auto-lint edited files
                if response.tool_calls and self.context:
                    lint_errors = self._auto_lint()
                    if lint_errors:
                        messages.append({
                            "role": "user",
                            "content": f"Linting found issues in edited files:\n{lint_errors}\n\nPlease fix them.",
                        })
                        continue
            else:
                logger.info(f"[{self.name}] Done: {self.metrics.summary()}")
                return response.content

        elapsed = self.metrics.elapsed
        return f"Max iterations ({elapsed:.1f}s, {self.metrics.total_llm_calls} LLM calls)"

    def run_streaming(self, task: str) -> Generator[dict, None, str]:
        provider = self.config.model_provider
        if provider is None:
            yield {"type": "error", "content": "Error: No model provider configured"}
            return "Error: No model provider configured"

        static_result = self._preprocess_task(task)
        if static_result:
            yield {"type": "result", "content": static_result}
            return static_result

        self.metrics = AgentMetrics(start_time=time.monotonic())

        messages = self._build_messages(task)
        tools = self.tools.to_schemas()

        for iteration in range(self.config.max_iterations):
            messages = self._trim_messages(messages)
            messages = self._summarize_old_messages(messages)

            if not self._check_token_budget():
                elapsed = self.metrics.elapsed
                error_msg = f"Token budget exceeded ({elapsed:.1f}s, {self.metrics.total_tokens} tokens used)"
                yield {"type": "error", "content": error_msg}
                return error_msg

            yield {"type": "progress", "content": f"Iteration {iteration + 1}/{self.config.max_iterations}"}

            prompt_hash = self._generate_cache_key(messages)
            cached = self.cache.get_llm_cache(prompt_hash)
            if cached:
                self.metrics.cache_hits += 1
                response = LLMResponse(content=cached, model=provider.__class__.__name__)
            else:
                llm_start = time.monotonic()
                response = provider.chat(messages, tools if tools else None)
                llm_latency = time.monotonic() - llm_start
                self.metrics.record_llm_call(llm_latency, response.total_tokens)
                if response.content and not response.tool_calls:
                    self.cache.set_llm_cache(prompt_hash, response.content)

            if response.is_error:
                yield {"type": "error", "content": response.content}
                return response.content

            yield {
                "type": "llm_response",
                "content": response.content or "",
                "tokens": response.total_tokens,
                "tool_calls": response.tool_calls,
            }

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

                    if isinstance(raw_args, str):
                        try:
                            tool_args = json.loads(raw_args)
                        except json.JSONDecodeError:
                            tool_args = {}
                    else:
                        tool_args = raw_args

                    yield {"type": "tool_call", "tool": tool_name, "args": tool_args}
                    result = self._execute_tool_with_retry(tool_name, tool_args)

                    if self.context and tool_name in ("write_file", "edit_file"):
                        path = tool_args.get("path", "")
                        if path and "Error" not in result:
                            self.context.add_file_written(path)

                    yield {"type": "tool_result", "tool": tool_name, "result": result[:6000]}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": result[:6000],
                    })

                # Reflection loop: auto-lint edited files
                if response.tool_calls and self.context:
                    lint_errors = self._auto_lint()
                    if lint_errors:
                        yield {"type": "progress", "content": f"Linting found issues:\n{lint_errors}"}
                        messages.append({
                            "role": "user",
                            "content": f"Linting found issues in edited files:\n{lint_errors}\n\nPlease fix them.",
                        })
                        continue
            else:
                logger.info(f"[{self.name}] Done streaming: {self.metrics.summary()}")
                yield {"type": "result", "content": response.content, "metrics": self.metrics.summary()}
                return response.content

        elapsed = self.metrics.elapsed
        result_msg = f"Max iterations ({elapsed:.1f}s, {self.metrics.total_llm_calls} LLM calls)"
        yield {"type": "result", "content": result_msg}
        return result_msg

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

    _reflection_linted: set[str] = set()

    def _auto_lint(self) -> str:
        if not self.context:
            return ""
        errors = []
        for fname in self.context.files_written:
            if fname in self._reflection_linted:
                continue
            self._reflection_linted.add(fname)
            if fname.endswith(".py"):
                full_path = Path(self.context.project_root) / fname
                if full_path.exists():
                    try:
                        import py_compile
                        py_compile.compile(str(full_path), doraise=True)
                    except py_compile.PyCompileError as e:
                        errors.append(f"{fname}: {e}")
                    except Exception as e:
                        errors.append(f"{fname}: compile error - {e}")
            elif fname.endswith((".js", ".jsx", ".ts", ".tsx")):
                pass
        return "\n".join(errors) if errors else ""
