"""Tests for the browser agent."""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from coding_agent.agents.browser_agent import create_browser_agent
from coding_agent.core.tool import ToolRegistry
from coding_agent.models.provider import MockProvider
from coding_agent.tools.browser_ops import register_browser_tools, BrowserSession


def test_browser_agent_creation():
    provider = MockProvider()
    tools = ToolRegistry()
    agent = create_browser_agent(provider, tools)
    assert agent.name == "browser"


def test_browser_agent_has_browser_name():
    provider = MockProvider()
    tools = ToolRegistry()
    agent = create_browser_agent(provider, tools)
    assert agent.config.name == "browser"


def test_browser_agent_system_prompt():
    provider = MockProvider()
    tools = ToolRegistry()
    agent = create_browser_agent(provider, tools)
    prompt = agent.get_system_prompt()
    assert "browser" in prompt.lower()
    assert "navigate" in prompt.lower()


def test_browser_agent_with_mock_tools():
    provider = MockProvider()
    tools = ToolRegistry()

    tools.register_function(
        name="screenshot_page",
        description="Take a screenshot",
        parameters=[],
        handler=lambda url="", path="": f"Screenshot saved to {path}",
    )
    tools.register_function(
        name="navigate_to",
        description="Navigate to URL",
        parameters=[],
        handler=lambda url="": f"Navigated to {url}",
    )

    agent = create_browser_agent(provider, tools)
    tool_names = [t.name for t in agent.tools.list_tools()]
    assert "screenshot_page" in tool_names
    assert "navigate_to" in tool_names


def test_browser_agent_tool_summary():
    provider = MockProvider()
    tools = ToolRegistry()

    tools.register_function(
        name="screenshot_page",
        description="Take a screenshot",
        parameters=[],
        handler=lambda url="", path="": f"Screenshot saved to {path}",
    )

    agent = create_browser_agent(provider, tools)
    summary = agent.get_tool_summary()
    assert "screenshot_page" in summary


def test_browser_agent_runs():
    provider = MockProvider(responses=["I have navigated to the page and taken a screenshot."])
    tools = ToolRegistry()
    agent = create_browser_agent(provider, tools)

    result = agent.run("Navigate to example.com")
    assert "screenshot" in result.lower() or "navigate" in result.lower()
    assert provider.call_count == 1


def test_browser_agent_runs_with_tool_calls():
    tool_calls = [{
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "navigate_to",
            "arguments": '{"url": "https://example.com"}',
        },
    }]

    provider = MockProvider(responses=None)
    call_count = 0
    original_chat = provider.chat

    def mock_chat(messages, tools=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            from coding_agent.models.provider import LLMResponse
            return LLMResponse(
                content="",
                tool_calls=tool_calls,
                model="mock",
            )
        return original_chat(messages, tools)

    provider.chat = mock_chat

    tools = ToolRegistry()
    tools.register_function(
        name="navigate_to",
        description="Navigate to URL",
        parameters=[],
        handler=lambda url="": f"Navigated to {url}\nTitle: Example Domain",
    )

    agent = create_browser_agent(provider, tools)
    result = agent.run("Go to example.com")
    assert "done" in result.lower() or "navigat" in result.lower()


def test_browser_session_mock():
    with patch("coding_agent.tools.browser_ops.BrowserSession") as mock_session_cls:
        mock_session = MagicMock()
        mock_page = MagicMock()
        mock_page.title.return_value = "Test Page"
        mock_page.url = "https://example.com"
        mock_page.inner_text.return_value = "Hello World"
        mock_page.query_selector_all.return_value = []
        mock_session.page = mock_page
        mock_session_cls.return_value = mock_session

        tools = ToolRegistry()
        register_browser_tools(tools, mock_session)

        tool_names = [t.name for t in tools.list_tools()]
        assert "screenshot_page" in tool_names
        assert "get_page_content" in tool_names
        assert "click_element" in tool_names
        assert "fill_form" in tool_names
        assert "navigate_to" in tool_names


def test_browser_tool_screenshot():
    with patch("coding_agent.tools.browser_ops.BrowserSession") as mock_session_cls:
        mock_session = MagicMock()
        mock_page = MagicMock()
        mock_session.page = mock_page
        mock_session_cls.return_value = mock_session

        tools = ToolRegistry()
        register_browser_tools(tools, mock_session)

        result = tools.execute("screenshot_page", url="https://example.com", path="test.png")
        assert "test.png" in result
        mock_page.goto.assert_called_once()
        mock_page.screenshot.assert_called_once_with(path="test.png", full_page=True)


def test_browser_tool_navigate():
    with patch("coding_agent.tools.browser_ops.BrowserSession") as mock_session_cls:
        mock_session = MagicMock()
        mock_page = MagicMock()
        mock_page.title.return_value = "Example"
        mock_page.url = "https://example.com"
        mock_session.page = mock_page
        mock_session_cls.return_value = mock_session

        tools = ToolRegistry()
        register_browser_tools(tools, mock_session)

        result = tools.execute("navigate_to", url="https://example.com")
        assert "Example" in result
        assert "example.com" in result


def test_browser_tool_click():
    with patch("coding_agent.tools.browser_ops.BrowserSession") as mock_session_cls:
        mock_session = MagicMock()
        mock_page = MagicMock()
        mock_page.url = "https://example.com"
        mock_session.page = mock_page
        mock_session_cls.return_value = mock_session

        tools = ToolRegistry()
        register_browser_tools(tools, mock_session)

        result = tools.execute("click_element", selector="#submit-btn")
        assert "Clicked" in result
        mock_page.click.assert_called_once_with("#submit-btn")


def test_browser_tool_fill_form():
    with patch("coding_agent.tools.browser_ops.BrowserSession") as mock_session_cls:
        mock_session = MagicMock()
        mock_page = MagicMock()
        mock_session.page = mock_page
        mock_session_cls.return_value = mock_session

        tools = ToolRegistry()
        register_browser_tools(tools, mock_session)

        values = '{"#email": "test@example.com", "#password": "secret"}'
        result = tools.execute("fill_form", selector="form", values=values)
        assert "2" in result
        assert mock_page.fill.call_count == 2


def test_browser_tool_fill_form_invalid_json():
    with patch("coding_agent.tools.browser_ops.BrowserSession") as mock_session_cls:
        mock_session = MagicMock()
        mock_page = MagicMock()
        mock_session.page = mock_page
        mock_session_cls.return_value = mock_session

        tools = ToolRegistry()
        register_browser_tools(tools, mock_session)

        result = tools.execute("fill_form", selector="form", values="not json")
        assert "Error" in result


def test_browser_tool_get_content():
    with patch("coding_agent.tools.browser_ops.BrowserSession") as mock_session_cls:
        mock_session = MagicMock()
        mock_page = MagicMock()
        mock_page.title.return_value = "Test Page"
        mock_page.url = "https://example.com"
        mock_page.inner_text.return_value = "Hello World"
        mock_page.query_selector_all.return_value = []
        mock_session.page = mock_page
        mock_session_cls.return_value = mock_session

        tools = ToolRegistry()
        register_browser_tools(tools, mock_session)

        result = tools.execute("get_page_content", url="https://example.com")
        assert "Test Page" in result
        assert "Hello World" in result
