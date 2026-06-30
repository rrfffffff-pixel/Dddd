"""CLI entry point for the coding agent."""

from __future__ import annotations

import logging
import sys

import click

from coding_agent.core.config import Config
from coding_agent.core.tool import ToolRegistry
from coding_agent.agents.code_agent import create_code_agent
from coding_agent.agents.test_agent import create_test_agent
from coding_agent.agents.shell_agent import create_shell_agent
from coding_agent.agents.review_agent import create_review_agent
from coding_agent.models.provider import create_provider
from coding_agent.orchestrator.task_router import TaskOrchestrator
from coding_agent.tools.file_ops import register_file_tools
from coding_agent.tools.shell import register_shell_tools
from coding_agent.tools.search import register_search_tools


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


@cli.command()
@click.argument("task")
@click.option("--model", "-m", default=None, help="Model (e.g. qwen3:1.7b, deepseek-chat, gpt-4o)")
@click.option("--provider", "-p", default=None, help="Provider (ollama, deepseek, openai, anthropic)")
@click.option("--project", "-d", default=".", help="Project root directory")
@click.pass_context
def run(ctx: click.Context, task: str, model: str | None, provider: str | None, project: str) -> None:
    """Run a coding task."""
    config = Config.load()
    if model:
        config.model.model = model
    if provider:
        config.model.provider = provider
    config.project_root = project

    llm = create_provider(
        config.model.provider,
        model=config.model.model,
        base_url=config.model.base_url,
        api_key=config.model.api_key,
    )

    # Build tool registry
    tools = ToolRegistry()
    register_file_tools(tools, project)
    register_shell_tools(tools, project)
    register_search_tools(tools, project)

    # Create agents
    agents = {
        "code": create_code_agent(llm, tools, project),
        "test": create_test_agent(llm, tools),
        "shell": create_shell_agent(llm, tools),
        "review": create_review_agent(llm, tools),
    }

    # Run orchestrator
    orchestrator = TaskOrchestrator(llm, agents, project)
    result = orchestrator.run(task)
    print(f"\nFinal result:\n{result}")


@cli.command()
@click.argument("task")
@click.option("--model", "-m", default=None, help="Model to use")
@click.option("--project", "-d", default=".", help="Project root directory")
@click.pass_context
def code(ctx: click.Context, task: str, model: str | None, project: str) -> None:
    """Run a quick coding task with just the code agent."""
    config = Config.load()
    if model:
        config.model.model = model

    llm = create_provider(
        config.model.provider,
        model=config.model.model,
        base_url=config.model.base_url,
    )

    tools = ToolRegistry()
    register_file_tools(tools, project)
    register_shell_tools(tools, project)
    register_search_tools(tools, project)

    agent = create_code_agent(llm, tools, project)
    result = agent.run(task)
    print(result)


@cli.command()
@click.option("--model", "-m", default=None, help="Model to use")
@click.option("--project", "-d", default=".", help="Project root directory")
@click.pass_context
def test(ctx: click.Context, model: str | None, project: str) -> None:
    """Run the test suite."""
    config = Config.load()
    if model:
        config.model.model = model

    llm = create_provider(
        config.model.provider,
        model=config.model.model,
        base_url=config.model.base_url,
    )

    tools = ToolRegistry()
    register_file_tools(tools, project)
    register_shell_tools(tools, project)
    register_search_tools(tools, project)

    agent = create_test_agent(llm, tools)
    result = agent.run("Run the full test suite for this project and report results")
    print(result)


@cli.command()
@click.pass_context
def info(ctx: click.Context) -> None:
    """Show configuration info."""
    config = Config.load()
    print(f"Provider: {config.model.provider}")
    print(f"Model: {config.model.model}")
    print(f"Base URL: {config.model.base_url}")
    print(f"Project root: {config.project_root}")


if __name__ == "__main__":
    cli()
