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
    status: str = "pending"  # pending, running, done, failed
    result: str = ""


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
        """Use LLM to break a task into subtasks with agent assignments."""
        agent_names = list(self.agents.keys())

        prompt = f"""Break this coding task into subtasks. Assign each to an agent.

Available agents: {', '.join(agent_names)}

Task: {task}

Output a JSON array of subtasks. Each subtask has:
- id: sequential number (1, 2, 3...)
- description: what to do (be specific and actionable)
- agent_name: one of [{', '.join(agent_names)}]

Rules:
- Keep it to 2-5 subtasks for efficiency
- Each subtask should be self-contained
- Use "code" for reading/writing/editing files
- Use "test" for running tests
- Use "shell" for installing dependencies or running commands
- Use "review" only after code changes are made

Output ONLY the JSON array, no other text."""

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
                    )
                    for i, s in enumerate(subtasks_data)
                ]
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse task decomposition: {e}")

        # Fallback: single task with code agent
        return [Subtask(id="1", description=task, agent_name="code")]

    def run(self, task: str) -> str:
        """Execute a complete coding task."""
        self.context.task_description = task

        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}\n")

        # Decompose
        print("Planning...")
        subtasks = self.decompose_task(task)
        print(f"Plan: {len(subtasks)} subtasks\n")

        results = []
        for subtask in subtasks:
            agent = self.agents.get(subtask.agent_name)
            if agent is None:
                logger.warning(f"Unknown agent: {subtask.agent_name}, using code agent")
                agent = self.agents.get("code")
            if agent is None:
                results.append(f"[{subtask.id}] Skipped - no agent available")
                continue

            print(f"[{subtask.id}] {subtask.agent_name}: {subtask.description}")
            subtask.status = "running"

            result = agent.run(subtask.description)
            subtask.result = result
            subtask.status = "done"

            print(f"  Result: {result[:200]}{'...' if len(result) > 200 else ''}\n")
            results.append(f"[{subtask.id}] {result}")

        # Summary
        print(f"{'='*60}")
        print("Task Complete")
        print(f"Files modified: {', '.join(self.context.files_written) or 'none'}")
        print(f"Errors: {len(self.context.errors)}")
        print(f"{'='*60}")

        return "\n\n".join(results)

    def run_simple(self, task: str) -> str:
        """Run a task with a single agent (no decomposition)."""
        self.context.task_description = task

        # Pick the best agent for the task
        agent_name = "code"
        if "test" in task.lower() or "run" in task.lower():
            agent_name = "test"
        elif "install" in task.lower() or "build" in task.lower():
            agent_name = "shell"
        elif "review" in task.lower() or "check" in task.lower():
            agent_name = "review"

        agent = self.agents.get(agent_name)
        if agent is None:
            return f"Error: Agent '{agent_name}' not available"

        return agent.run(task)
