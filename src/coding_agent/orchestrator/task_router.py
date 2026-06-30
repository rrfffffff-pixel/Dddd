"""Task Orchestrator - decomposes tasks and coordinates agents."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from coding_agent.core.agent import Agent
from coding_agent.core.context import SharedContext
from coding_agent.core.message import MessageBus, MessageType
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
    max_retries: int = 1


class TaskOrchestrator:
    """Decomposes user tasks and coordinates agent execution."""

    def __init__(
        self,
        provider: LLMProvider,
        agents: dict[str, Agent],
        project_root: str = ".",
    ) -> None:
        self.provider = provider
        self.agents = agents
        self.bus = MessageBus()
        self.context = SharedContext(project_root=project_root)

        for agent in agents.values():
            agent.bus = self.bus
            agent.set_context(self.context)

    def decompose_task(self, task: str) -> list[Subtask]:
        agent_names = list(self.agents.keys())

        prompt = f"""Break this coding task into an ordered sequence of subtasks.

Available agents: {', '.join(agent_names)}

Task: {task}

Output a JSON array. Each item:
- id: string ("1", "2", ...)
- description: specific, actionable instruction
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
- Start with exploration/reading, end with verification

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

    def run(self, task: str) -> str:
        self.context.task_description = task

        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}\n")

        print("Planning...")
        subtasks = self.decompose_task(task)
        print(f"Plan: {len(subtasks)} subtasks\n")

        results = []
        max_rounds = len(subtasks) * 3  # allow retries
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

                dep_info = f" (after: {', '.join(subtask.depends_on)})" if subtask.depends_on else ""
                retry_info = f" [retry {subtask.retry_count}]" if subtask.retry_count > 0 else ""
                print(f"[{subtask.id}] {subtask.agent_name}: {subtask.description}{dep_info}{retry_info}")
                subtask.status = "running"

                result = agent.run(subtask.description)
                subtask.result = result
                subtask.status = "done"

                if self._should_retry(subtask, result) and subtask.retry_count < subtask.max_retries:
                    subtask.retry_count += 1
                    subtask.status = "pending"
                    print(f"  Retrying ({subtask.retry_count}/{subtask.max_retries})...")
                    continue

                print(f"  Done: {result[:150]}{'...' if len(result) > 150 else ''}\n")
                results.append(f"[{subtask.id}] {result}")

        print(f"{'='*60}")
        print("Task Complete")
        print(f"Files modified: {', '.join(self.context.files_written) or 'none'}")
        print(f"Errors: {len(self.context.errors)}")
        print(f"{'='*60}")

        return "\n\n".join(results)

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
