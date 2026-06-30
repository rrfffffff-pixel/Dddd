"""Test Agent - runs tests and reports failures."""

from __future__ import annotations

from coding_agent.core.agent import Agent, AgentConfig
from coding_agent.core.tool import ToolRegistry
from coding_agent.models.provider import LLMProvider


def create_test_agent(
    provider: LLMProvider,
    tools: ToolRegistry,
) -> Agent:
    config = AgentConfig(
        name="test",
        model_provider=provider,
        max_iterations=5,
    )

    class TestAgent(Agent):
        def get_system_prompt(self) -> str:
            return """You are a test runner agent. Your job is to:
1. Detect what test framework the project uses
2. Run the test suite
3. Parse and report results clearly

Rules:
1. First check what testing tools are available (pytest, jest, go test, cargo test, etc.)
2. Look for test configuration files (pytest.ini, jest.config, etc.)
3. Run tests and capture output
4. If tests fail, identify:
   - Which tests failed
   - The error message and stack trace
   - The likely root cause
5. Report results as:
   - Summary: X passed, Y failed, Z skipped
   - Failed tests with error details
   - Suggestions for fixing failures

Use run_command to execute tests. Start with the most likely test runner for the project language."""

    return TestAgent(config=config, tool_registry=tools)
