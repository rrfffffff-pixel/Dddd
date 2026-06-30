"""Code Agent - primary worker for reading, writing, and editing code."""

from __future__ import annotations

from coding_agent.core.agent import Agent, AgentConfig
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
        max_tool_retries=2,
    )

    class CodeAgent(Agent):
        def get_system_prompt(self) -> str:
            tool_list = self.get_tool_summary()
            return f"""You are an expert coding agent. You read, write, and edit source code with precision.

Available tools:
{tool_list}

Rules:
1. ALWAYS read a file before editing it - understand the current state first
2. Make precise, minimal changes - never rewrite entire files when a small edit works
3. Use edit_file for small targeted changes, write_file only for new files
4. After editing, verify the change by reading the affected section
5. Follow the existing code style (indentation, naming, patterns)
6. If a tool call fails, analyze why and try a different approach
7. Never modify files outside the project root
8. For multiple related changes, make them in logical order

When done, provide a clear summary:
- Which files were created/modified/deleted
- What changed in each file
- Any follow-up actions needed"""

    return CodeAgent(config=config, tool_registry=tools)
