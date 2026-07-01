"""Task Orchestrator - decomposes tasks and coordinates agents."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from coding_agent.core.agent import Agent
from coding_agent.core.context import SharedContext
from coding_agent.core.message import MessageBus
from coding_agent.intelligence.repomap import RepoMap
from coding_agent.models.provider import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class Subtask:
    id: str
    description: str
    agent_name: str
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"  # pending, running, done, failed
    result: str = ""
    retry_count: int = 0
    max_retries: int = 3
    started_at: float = 0.0
    completed_at: float = 0.0
    execution_time: float = 0.0


class TaskOrchestrator:
    """Decomposes user tasks and coordinates agent execution."""

    def __init__(
        self,
        provider: LLMProvider,
        agents: dict[str, Agent],
        project_root: str = ".",
        progress_callback: Callable[[str, str, str], None] | None = None,
    ) -> None:
        self.provider = provider
        self.agents = agents
        self.bus = MessageBus()
        self.context = SharedContext(project_root=project_root)
        self.progress_callback = progress_callback

        for agent in agents.values():
            agent.bus = self.bus
            agent.set_context(self.context)

        # Setup RepoMap for all agents
        self.repo_map = RepoMap(root=project_root, map_tokens=1024)
        for agent in agents.values():
            if hasattr(agent, "set_repo_map"):
                agent.set_repo_map(self.repo_map)

        # Auto-inject self-architecture when working on own codebase
        root = Path(project_root).resolve()
        agents_md = root / "AGENTS.md"
        markers = [root / "src" / "coding_agent" / "main.py", root / "src" / "coding_agent" / "core" / "agent.py"]
        if agents_md.exists() and all(m.exists() for m in markers):
            arch = agents_md.read_text(encoding="utf-8", errors="replace")
            self.context.project_type = "coding-agent"
            self.context.metadata["architecture"] = arch[:2000]
            self.context.add_decision("Working on coding agent's own codebase - architecture loaded from AGENTS.md")

    def _notify_progress(self, subtask_id: str, status: str, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(subtask_id, status, message)

    def _get_progress_summary(self, subtasks: list[Subtask]) -> str:
        done = sum(1 for s in subtasks if s.status == "done")
        total = len(subtasks)
        pct = (done / total * 100) if total > 0 else 0
        return f"{done}/{total} ({pct:.0f}%)"

    def decompose_task(self, task: str) -> list[Subtask]:
        agent_names = list(self.agents.keys())

        prompt = f"""Break this coding task into an ordered sequence of subtasks.

Available agents: {', '.join(agent_names)}

Task: {task}

Output a JSON array. Each item:
- id: string ("1", "2", ...)
- description: specific, actionable instruction (include file paths if known)
- agent_name: one of [{', '.join(agent_names)}]
- depends_on: list of subtask IDs that must complete first (empty if none)

Agent roles:
- "code": read, write, edit source files
- "test": run test suites, verify correctness
- "shell": install deps, run build commands, git operations
- "review": code review, security checks (use after code changes)

Rules:
- 2-6 subtasks
- Include a test step after code changes
- Dependencies must form a DAG (no cycles)
- Start with code reading/exploration, end with review or testing
- Make each subtask specific (include file names, function names)

Output ONLY the JSON array."""

        response = self.provider.chat_simple(prompt, system="You are a task planner. Output only valid JSON.")

        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                subtasks_data = json.loads(match.group())
                return [
                    Subtask(
                        id=str(s.get("id", i + 1)),
                        description=s.get("description", ""),
                        agent_name=s.get("agent_name", "code"),
                        depends_on=[str(d) for d in s.get("depends_on", [])],
                    )
                    for i, s in enumerate(subtasks_data)
                ]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse task decomposition: {e}")

        return [Subtask(id="1", description=task, agent_name="code")]

    def _get_ready_tasks(self, subtasks: list[Subtask]) -> list[Subtask]:
        done_ids = {s.id for s in subtasks if s.status == "done"}
        return [
            s for s in subtasks
            if s.status == "pending" and all(d in done_ids for d in s.depends_on)
        ]

    def _should_retry(self, subtask: Subtask, result: str) -> bool:
        if subtask.retry_count >= subtask.max_retries:
            return False
        failure_indicators = ["error", "failed", "exception", "traceback", "assert"]
        return any(ind in result.lower() for ind in failure_indicators)

    def _backoff_delay(self, retry_count: int) -> float:
        """Exponential backoff: 1s, 2s, 4s, 8s, ..."""
        return min(2 ** retry_count, 30)

    def _print_summary(self, subtasks: list[Subtask], total_time: float) -> None:
        succeeded = sum(1 for s in subtasks if s.status == "done")
        failed = sum(1 for s in subtasks if s.status == "failed")
        retries = sum(s.retry_count for s in subtasks)
        timed = [s for s in subtasks if s.execution_time > 0]
        avg_time = (sum(s.execution_time for s in timed) / len(timed)) if timed else 0

        print(f"\n{'='*60}")
        print("Execution Summary")
        print(f"{'='*60}")
        print(f"Total time:     {total_time:.1f}s")
        print(f"Tasks:          {succeeded} succeeded, {failed} failed, {len(subtasks)} total")
        print(f"Retries:        {retries}")
        print(f"Avg task time:  {avg_time:.1f}s")
        if timed:
            slowest = max(timed, key=lambda s: s.execution_time)
            print(f"Slowest task:   [{slowest.id}] {slowest.execution_time:.1f}s")
        print(f"Files modified: {', '.join(self.context.files_written) or 'none'}")
        print(f"Errors:         {len(self.context.errors)}")
        print(f"{'='*60}\n")

    def _inject_context(self, subtask: Subtask, completed: list[Subtask]) -> str:
        """Inject results from completed dependencies as context."""
        dep_results = []
        for dep_id in subtask.depends_on:
            dep = next((s for s in completed if s.id == dep_id), None)
            if dep and dep.result:
                summary = dep.result[:500].strip()
                dep_results.append(f"[{dep.id}] {dep.agent_name}: {summary}")
        if dep_results:
            return "\n".join(dep_results)
        return ""

    def run(self, task: str) -> str:
        self.context.task_description = task
        overall_start = time.monotonic()

        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}\n")

        print("Planning...")
        subtasks = self.decompose_task(task)
        print(f"Plan: {len(subtasks)} subtasks\n")

        results = []
        completed_subtasks: list[Subtask] = []
        max_rounds = len(subtasks) * 3
        for round_num in range(max_rounds):
            ready = self._get_ready_tasks(subtasks)
            if not ready:
                break

            for subtask in ready:
                agent = self.agents.get(subtask.agent_name)
                if agent is None:
                    agent = self.agents.get("code")
                if agent is None:
                    subtask.status = "failed"
                    subtask.result = f"No agent '{subtask.agent_name}' available"
                    continue

                # Inject context from completed dependencies
                context = self._inject_context(subtask, completed_subtasks)
                task_description = subtask.description
                if context:
                    task_description = f"{subtask.description}\n\nContext from previous steps:\n{context}"

                dep_info = f" (after: {', '.join(subtask.depends_on)})" if subtask.depends_on else ""
                retry_info = f" [retry {subtask.retry_count}]" if subtask.retry_count > 0 else ""
                progress = self._get_progress_summary(subtasks)
                print(f"[{subtask.id}] {subtask.agent_name}: {subtask.description}{dep_info}{retry_info} ({progress})")
                subtask.status = "running"
                self._notify_progress(subtask.id, "running", subtask.description)

                subtask.started_at = time.monotonic()
                result = agent.run(task_description)
                subtask.completed_at = time.monotonic()
                subtask.execution_time = subtask.completed_at - subtask.started_at
                subtask.result = result
                subtask.status = "done"

                if self._should_retry(subtask, result) and subtask.retry_count < subtask.max_retries:
                    subtask.retry_count += 1
                    subtask.status = "pending"
                    delay = self._backoff_delay(subtask.retry_count)
                    print(f"  Retry {subtask.retry_count}/{subtask.max_retries} after {delay:.0f}s backoff")
                    self._notify_progress(subtask.id, "retrying", f"Retry {subtask.retry_count}/{subtask.max_retries}")
                    time.sleep(delay)
                    continue

                print(f"  Done ({subtask.execution_time:.1f}s): {result[:150]}{'...' if len(result) > 150 else ''}\n")
                self._notify_progress(subtask.id, "done", result)
                results.append(f"[{subtask.id}] {result}")
                completed_subtasks.append(subtask)

        self._print_summary(subtasks, time.monotonic() - overall_start)
        return "\n\n".join(results)

    def run_parallel(self, task: str, max_workers: int = 4) -> str:
        """Run subtasks with parallel execution respecting the dependency graph."""
        self.context.task_description = task
        overall_start = time.monotonic()

        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}\n")

        print("Planning...")
        subtasks = self.decompose_task(task)
        print(f"Plan: {len(subtasks)} subtasks\n")

        results: dict[str, str] = {}
        completed_lock = threading.Lock()
        status_lock = threading.Lock()
        max_rounds = len(subtasks) * 3

        def execute_subtask(subtask: Subtask) -> str:
            agent = self.agents.get(subtask.agent_name)
            if agent is None:
                agent = self.agents.get("code")
            if agent is None:
                with status_lock:
                    subtask.status = "failed"
                    subtask.result = f"No agent '{subtask.agent_name}' available"
                self._notify_progress(subtask.id, "failed", subtask.result)
                return ""

            dep_info = f" (after: {', '.join(subtask.depends_on)})" if subtask.depends_on else ""
            retry_info = f" [retry {subtask.retry_count}]" if subtask.retry_count > 0 else ""
            progress = self._get_progress_summary(subtasks)
            print(f"[{subtask.id}] {subtask.agent_name}: {subtask.description}{dep_info}{retry_info} ({progress})")

            with status_lock:
                subtask.status = "running"
            self._notify_progress(subtask.id, "running", subtask.description)

            subtask.started_at = time.monotonic()
            result = agent.run(subtask.description)
            subtask.completed_at = time.monotonic()
            subtask.execution_time = subtask.completed_at - subtask.started_at

            if self._should_retry(subtask, result) and subtask.retry_count < subtask.max_retries:
                with status_lock:
                    subtask.retry_count += 1
                    subtask.status = "pending"
                    subtask.result = result
                delay = self._backoff_delay(subtask.retry_count)
                self._notify_progress(subtask.id, "retrying", f"Retry {subtask.retry_count}/{subtask.max_retries}")
                time.sleep(delay)
                return ""

            with status_lock:
                subtask.result = result
                subtask.status = "done"

            print(f"  Done ({subtask.execution_time:.1f}s): {result[:150]}{'...' if len(result) > 150 else ''}\n")
            self._notify_progress(subtask.id, "done", result)
            return result

        for round_num in range(max_rounds):
            ready = self._get_ready_tasks(subtasks)
            if not ready:
                break

            if len(ready) == 1:
                subtask = ready[0]
                result = execute_subtask(subtask)
                if result:
                    with completed_lock:
                        results[subtask.id] = result
                continue

            with ThreadPoolExecutor(max_workers=min(max_workers, len(ready))) as executor:
                futures: dict[Future, Subtask] = {}
                for subtask in ready:
                    future = executor.submit(execute_subtask, subtask)
                    futures[future] = subtask

                for future in futures:
                    subtask = futures[future]
                    try:
                        result = future.result()
                        if result:
                            with completed_lock:
                                results[subtask.id] = result
                    except Exception as e:
                        with status_lock:
                            subtask.status = "failed"
                            subtask.result = f"Error: {e}"
                        self._notify_progress(subtask.id, "failed", str(e))

        self._print_summary(subtasks, time.monotonic() - overall_start)

        ordered = [f"[{sid}] {results[sid]}" for sid in sorted(results.keys())]
        return "\n\n".join(ordered)

    def run_simple(self, task: str) -> str:
        self.context.task_description = task

        agent_name = "code"
        lower = task.lower()
        if "test" in lower or "run" in lower:
            agent_name = "test"
        elif "install" in lower or "build" in lower:
            agent_name = "shell"
        elif "review" in lower or "check" in lower:
            agent_name = "review"

        agent = self.agents.get(agent_name)
        if agent is None:
            return f"Error: Agent '{agent_name}' not available"

        return agent.run(task)
