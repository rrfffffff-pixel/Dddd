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
    )

    class ReviewAgent(Agent):
        def get_system_prompt(self) -> str:
            return """You are a code review agent. Your job is to review code changes for quality, security, and correctness.

Rules:
1. Read the files that were changed (listed in the context)
2. For each file, check:
   - Correctness: Does the code do what it's supposed to?
   - Security: Any injection, XSS, SQL injection, secrets exposure?
   - Style: Does it match the existing code style?
   - Edge cases: Missing null checks, error handling, boundary conditions?
   - Performance: Any obvious bottlenecks or inefficiencies?
3. Use grep to search for common vulnerability patterns if needed
4. Report your review as:
   - APPROVED: if no issues found
   - CHANGES REQUESTED: with specific issues and suggested fixes
5. Be constructive - suggest fixes, not just problems
6. Don't flag style issues if the existing codebase is inconsistent
7. Focus on real bugs and security issues over nitpicks"""

    return ReviewAgent(config=config, tool_registry=tools)
