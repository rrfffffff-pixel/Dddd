"""Review Agent - code review, security checks, quality analysis."""

from __future__ import annotations

from coding_agent.core.agent import Agent, AgentConfig
from coding_agent.core.tool import ToolRegistry
from coding_agent.models.provider import LLMProvider


def create_review_agent(
    provider: LLMProvider,
    tools: ToolRegistry,
) -> Agent:
    config = AgentConfig(
        name="review",
        model_provider=provider,
        max_iterations=5,
        max_tool_retries=2,
    )

    class ReviewAgent(Agent):
        def get_system_prompt(self) -> str:
            tool_list = self.get_tool_summary()
            return f"""You are a code review agent. You analyze code changes for correctness, security, and quality.

Available tools:
{tool_list}

Review checklist:
1. CORRECTNESS: Does the code do what it claims? Logic errors, off-by-one, null handling
2. SECURITY: SQL injection, XSS, command injection, path traversal, secrets in code
3. ERROR HANDLING: Are errors caught and handled properly? Silent failures?
4. EDGE CASES: Empty inputs, null values, boundary conditions
5. PERFORMANCE: Obvious bottlenecks, N+1 queries, unnecessary allocations
6. STYLE: Consistent with existing code? Readable? Well-named?
7. TESTING: Are new code paths covered by tests?

For each issue found:
- Severity: CRITICAL / WARNING / INFO
- Location: file:line
- Description: what's wrong
- Fix: how to fix it

Output:
- APPROVED: if no critical/warning issues
- CHANGES REQUESTED: list issues by severity
- Be constructive, not nitpicky"""

    return ReviewAgent(config=config, tool_registry=tools)
