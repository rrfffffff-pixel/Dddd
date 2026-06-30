"""Code Agent - primary worker for reading, writing, and editing code."""

from __future__ import annotations

from coding_agent.core.agent import Agent, AgentConfig
from coding_agent.core.context import SharedContext
from coding_agent.core.tool import ToolRegistry
from coding_agent.models.provider import LLMProvider


def create_code_agent(
    provider: LLMProvider,
    tools: ToolRegistry,
    project_root: str = ".",
) -> Agent:
    config = AgentConfig(
        name="code",
        model_provider=provider,
        max_iterations=15,
    )

    class CodeAgent(Agent):
        def get_system_prompt(self) -> str:
            return """You are a coding agent. Your job is to read, write, and edit source code.

Rules:
1. Always read a file before editing it
2. Make precise, minimal changes - don't rewrite entire files unnecessarily
3. Use edit_file for small changes, write_file only for new files or complete rewrites
4. Verify your changes make sense by reading the file after editing
5. Report what files you changed and what you did
6. If you encounter an error, try to fix it before giving up
7. Never modify files outside the project root
8. Follow the existing code style of the project

When done, summarize all changes made."""

    return CodeAgent(config=config, tool_registry=tools)
