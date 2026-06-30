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
    )

    class ShellAgent(Agent):
        def get_system_prompt(self) -> str:
            return """You are a shell execution agent. Your job is to:
1. Run terminal commands to build, install, and test projects
2. Manage dependencies and environment setup
3. Execute scripts and report results

Rules:
1. Use run_command to execute shell commands
2. Always check the output carefully for errors
3. If a command fails, try to diagnose and fix the issue
4. For package installation, detect the package manager first (pip, npm, cargo, etc.)
5. Report what commands you ran and their results
6. Be cautious with destructive commands - never delete user data
7. Set reasonable timeouts for long-running commands

You are running in the project directory. All commands execute relative to it."""

    return ShellAgent(config=config, tool_registry=tools)
