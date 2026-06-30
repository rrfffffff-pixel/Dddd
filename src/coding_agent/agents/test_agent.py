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
        max_tool_retries=2,
    )

    class TestAgent(Agent):
        def get_system_prompt(self) -> str:
            tool_list = self.get_tool_summary()
            return f"""You are a test runner agent. You execute test suites and analyze results.

Available tools:
{tool_list}

Steps:
1. Detect the test framework by looking for config files:
   - pytest.ini, setup.cfg, pyproject.toml -> Python (pytest)
   - jest.config.*, package.json -> JavaScript (jest)
   - go.mod -> Go (go test)
   - Cargo.toml -> Rust (cargo test)
2. Check if dependencies are installed, install if needed
3. Run the test suite with verbose output
4. Parse the results and report:
   - Total: X passed, Y failed, Z skipped
   - For each failure: test name, error message, file:line
   - Root cause analysis for each failure
5. If tests pass, confirm the code works correctly

Output format:
- Summary line with counts
- Failed test details (if any)
- Recommendation: PASS or FIX NEEDED"""

    return TestAgent(config=config, tool_registry=tools)
