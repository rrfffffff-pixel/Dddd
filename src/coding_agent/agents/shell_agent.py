"""Shell Agent - executes terminal commands and manages the environment."""

from __future__ import annotations

from coding_agent.core.agent import Agent, AgentConfig
from coding_agent.core.tool import ToolRegistry
from coding_agent.models.provider import LLMProvider


def create_shell_agent(
    provider: LLMProvider,
    tools: ToolRegistry,
) -> Agent:
    config = AgentConfig(
        name="shell",
        model_provider=provider,
        max_iterations=10,
        max_tool_retries=1,
    )

    class ShellAgent(Agent):
        def get_system_prompt(self) -> str:
            tool_list = self.get_tool_summary()
            return f"""You are a shell execution agent. You run terminal commands to build and manage projects.

Available tools:
{tool_list}

Rules:
1. Detect the environment first:
   - Check for package managers (pip, npm, yarn, cargo, go)
   - Check for Docker, docker-compose
   - Check for Makefiles
2. Run commands with appropriate timeouts
3. If a command fails, analyze the error and try to fix it
4. For installs, use the detected package manager
5. Never run destructive commands (rm -rf /, drop database, etc.)
6. Report: command run, output, exit code, and whether it succeeded

Common workflows:
- Setup: detect project type -> install deps -> verify
- Build: detect build tool -> run build -> report status
- Clean: find build artifacts -> remove -> confirm"""

    return ShellAgent(config=config, tool_registry=tools)
