"""CLI entry point for the coding agent."""

from __future__ import annotations

import logging
import time

import click

from coding_agent.agents.code_agent import create_code_agent
from coding_agent.agents.review_agent import create_review_agent
from coding_agent.agents.shell_agent import create_shell_agent
from coding_agent.agents.test_agent import create_test_agent
from coding_agent.benchmark.runner import DEFAULT_BENCHMARKS, BenchmarkResult, BenchmarkTask, run_benchmark
from coding_agent.core.checkpoint import CheckpointManager
from coding_agent.core.config import Config
from coding_agent.core.tool import Tool, ToolRegistry
from coding_agent.intelligence.rules import example_rules_dir, load_rules, rules_summary
from coding_agent.models.provider import create_provider
from coding_agent.orchestrator.task_router import TaskOrchestrator
from coding_agent.plugins.mcp import MCPManager
from coding_agent.tools.file_ops import register_file_tools
from coding_agent.tools.info import register_info_tools
from coding_agent.tools.search import register_search_tools
from coding_agent.tools.shell import register_shell_tools


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
@click.option("--sequential", is_flag=True, help="Run subtasks sequentially instead of parallel")
@click.pass_context
def run(ctx: click.Context, task: str, model: str | None, provider: str | None, project: str, sequential: bool) -> None:
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
    register_info_tools(tools, project)
    register_cursor_tools(tools, project)

    # Create agents
    agents = {
        "code": create_code_agent(llm, tools, project),
        "test": create_test_agent(llm, tools),
        "shell": create_shell_agent(llm, tools),
        "review": create_review_agent(llm, tools),
    }

    for agent in agents.values():
        agent.load_project_rules(project)

    # Run orchestrator (parallel by default for speed)
    orchestrator = TaskOrchestrator(llm, agents, project)
    if sequential:
        result = orchestrator.run(task)
    else:
        result = orchestrator.run_parallel(task)
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
    register_info_tools(tools, project)
    register_cursor_tools(tools, project)

    agent = create_code_agent(llm, tools, project)
    agent.load_project_rules(project)
    for event in agent.run_streaming(task):
        if event["type"] == "progress":
            print(f"  {event['content']}")
        elif event["type"] == "tool_call":
            print(f"  🔧 {event['tool']}({event['args']})")
        elif event["type"] == "tool_result":
            result_preview = event['result'][:100].replace('\n', ' ')
            print(f"  -> {result_preview}")
        elif event["type"] == "llm_response":
            if event.get('tool_calls'):
                names = [tc.get('function',{}).get('name','?') for tc in event['tool_calls']]
                print(f"  🤖 LLM -> tools: {', '.join(names)}")
        elif event["type"] == "result":
            print(f"\nResult:\n{event['content']}")
        elif event["type"] == "error":
            print(f"  ❌ {event['content']}")


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
    register_info_tools(tools, project)

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


@cli.command()
@click.option("--model", "-m", default=None, help="Model to use")
@click.option("--project", "-d", default=".", help="Project root directory")
@click.pass_context
def benchmark(ctx: click.Context, model: str | None, project: str) -> None:
    """Run benchmark tests on the coding agent."""
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

    def runner(task: BenchmarkTask) -> BenchmarkResult:
        start = time.monotonic()
        output = agent.run(task.description)
        elapsed = time.monotonic() - start
        success = task.expected_outcome.lower() in output.lower() if task.expected_outcome else bool(output)
        return BenchmarkResult(
            task_name=task.name,
            success=success,
            duration=elapsed,
            output=output[:200],
        )

    print(f"\n{'='*60}")
    print("Benchmark: Coding Agent")
    print(f"Model: {config.model.model}")
    print(f"{'='*60}\n")

    suite = run_benchmark("Coding Agent", DEFAULT_BENCHMARKS, runner)
    print(f"\n{suite.summary()}\n")


def register_cursor_tools(tools: ToolRegistry, project_root: str) -> None:
    """Register Cursor-like tools (rules, checkpoints, MCP)."""

    def list_rules_tool() -> str:
        rules = load_rules(project_root)
        return rules_summary(rules) if rules else "No project rules found in .coding-agent/rules/"

    def checkpoints_tool() -> str:
        cm = CheckpointManager(project_root)
        cps = cm.list_checkpoints()
        if not cps:
            return "No checkpoints saved yet"
        return "\n".join(f"{cp['id']}: {cp['task']} (iter {cp['iteration']})" for cp in cps)

    def mcp_tools_tool() -> str:
        mcp = MCPManager()
        mcp.load_config(project_root)
        tools = mcp.discover_tools()
        if not tools:
            return "No MCP tools configured (create .coding-agent/mcp.json to add some)"
        return "\n".join(f"- {t.name}: {t.description}" for t in tools)

    tools.register(Tool(
        name="list_rules",
        description="List project-specific rules from .coding-agent/rules/ directory",
        parameters=[],
        handler=list_rules_tool,
    ))
    tools.register(Tool(
        name="list_checkpoints",
        description="Show saved agent session checkpoints",
        parameters=[],
        handler=checkpoints_tool,
    ))
    tools.register(Tool(
        name="list_mcp_tools",
        description="List available MCP (Model Context Protocol) external tools",
        parameters=[],
        handler=mcp_tools_tool,
    ))

    example_rules_dir(project_root)


if __name__ == "__main__":
    cli()
