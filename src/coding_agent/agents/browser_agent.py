"""Browser Agent - agent for web browsing and automation tasks."""

from __future__ import annotations

from coding_agent.core.agent import Agent, AgentConfig
from coding_agent.core.tool import ToolRegistry
from coding_agent.models.provider import LLMProvider


def create_browser_agent(
    provider: LLMProvider,
    tools: ToolRegistry,
) -> Agent:
    config = AgentConfig(
        name="browser",
        model_provider=provider,
        max_iterations=15,
        max_tool_retries=2,
    )

    class BrowserAgent(Agent):
        def get_system_prompt(self) -> str:
            tool_list = self.get_tool_summary()
            return f"""You are a browser automation agent. You browse websites, take screenshots, extract content, and interact with web pages.

Available tools:
{tool_list}

Rules:
1. Navigate to pages before interacting with them
2. Use CSS selectors to target elements (e.g., "#login-button", ".submit-btn", "a[href='/about']")
3. Take screenshots to verify page state after actions
4. Extract page content to understand page structure before interacting
5. Handle navigation errors gracefully - retry if network issues occur
6. Wait for pages to load before taking screenshots or extracting content
7. Use fill_form with a JSON object mapping selectors to values
8. Report the current URL and page title when done

When done, provide a clear summary:
- Which pages were visited
- What actions were taken
- Any content extracted or screenshots saved"""

    return BrowserAgent(config=config, tool_registry=tools)
