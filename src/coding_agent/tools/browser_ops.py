"""Browser operation tools for agents."""

from __future__ import annotations

import json
import logging
from typing import Any

from coding_agent.core.tool import Tool, ToolParameter, ToolRegistry

logger = logging.getLogger(__name__)


class BrowserSession:
    """Manages a Playwright browser session."""

    def __init__(self, headless: bool = True) -> None:
        self._headless = headless
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None

    def start(self) -> None:
        from playwright.sync_api import sync_playwright
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self._headless)
        self._page = self._browser.new_page()

    def stop(self) -> None:
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        self._page = None
        self._browser = None
        self._playwright = None

    @property
    def page(self) -> Any:
        if self._page is None:
            self.start()
        return self._page

    def __enter__(self) -> BrowserSession:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


def register_browser_tools(registry: ToolRegistry, session: BrowserSession | None = None) -> None:
    """Register browser automation tools."""

    if session is None:
        session = BrowserSession(headless=True)
        session.start()

    def screenshot_page(url: str = "", path: str = "") -> str:
        """Take a screenshot of the current page or navigate to URL first."""
        try:
            page = session.page
            if url:
                page.goto(url, wait_until="networkidle", timeout=30000)
            screenshot_path = path or "screenshot.png"
            page.screenshot(path=screenshot_path, full_page=True)
            return f"Screenshot saved to {screenshot_path}"
        except Exception as e:
            return f"Error taking screenshot: {e}"

    def get_page_content(url: str = "") -> str:
        """Extract text content and links from the current page."""
        try:
            page = session.page
            if url:
                page.goto(url, wait_until="networkidle", timeout=30000)

            title = page.title()
            text_content = page.inner_text("body")[:5000]

            links = []
            for link in page.query_selector_all("a[href]"):
                href = link.get_attribute("href") or ""
                text = link.inner_text().strip()
                if href and text:
                    links.append({"text": text[:100], "href": href[:200]})

            forms = []
            for form in page.query_selector_all("form"):
                action = form.get_attribute("action") or ""
                method = form.get_attribute("method") or "GET"
                inputs = []
                for inp in form.query_selector_all("input, textarea, select"):
                    inputs.append({
                        "type": inp.get_attribute("type") or "text",
                        "name": inp.get_attribute("name") or "",
                        "placeholder": inp.get_attribute("placeholder") or "",
                    })
                forms.append({"action": action, "method": method, "inputs": inputs[:20]})

            result = {
                "title": title,
                "text": text_content,
                "links": links[:50],
                "forms": forms[:10],
                "url": page.url,
            }
            return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error getting page content: {e}"

    def click_element(selector: str) -> str:
        """Click an element on the page using a CSS selector."""
        try:
            page = session.page
            page.wait_for_selector(selector, timeout=10000)
            page.click(selector)
            page.wait_for_load_state("networkidle", timeout=15000)
            return f"Clicked element: {selector}. Current URL: {page.url}"
        except Exception as e:
            return f"Error clicking element: {e}"

    def fill_form(selector: str, values: str) -> str:
        """Fill form fields. values is a JSON string mapping selectors to values."""
        try:
            page = session.page
            parsed = json.loads(values)
            for field_selector, value in parsed.items():
                page.wait_for_selector(field_selector, timeout=5000)
                page.fill(field_selector, str(value))
            return f"Filled {len(parsed)} form fields"
        except json.JSONDecodeError:
            return "Error: values must be a JSON object mapping selectors to values"
        except Exception as e:
            return f"Error filling form: {e}"

    def navigate_to(url: str) -> str:
        """Navigate to a URL and return the page title."""
        try:
            page = session.page
            page.goto(url, wait_until="networkidle", timeout=30000)
            return f"Navigated to: {page.url}\nTitle: {page.title()}"
        except Exception as e:
            return f"Error navigating: {e}"

    registry.register(Tool(
        name="screenshot_page",
        description="Take a screenshot of a web page. Optionally navigate to a URL first.",
        parameters=[
            ToolParameter(name="url", type="string", description="URL to navigate to before screenshot", required=False, default=""),
            ToolParameter(name="path", type="string", description="Path to save screenshot", required=False, default="screenshot.png"),
        ],
        handler=screenshot_page,
    ))

    registry.register(Tool(
        name="get_page_content",
        description="Extract text content, links, and form info from a web page",
        parameters=[
            ToolParameter(name="url", type="string", description="URL to fetch content from", required=False, default=""),
        ],
        handler=get_page_content,
    ))

    registry.register(Tool(
        name="click_element",
        description="Click an element on the page using CSS selector",
        parameters=[
            ToolParameter(name="selector", type="string", description="CSS selector for the element to click"),
        ],
        handler=click_element,
    ))

    registry.register(Tool(
        name="fill_form",
        description="Fill form fields with values. Values is a JSON object mapping CSS selectors to values.",
        parameters=[
            ToolParameter(name="selector", type="string", description="CSS selector for the form"),
            ToolParameter(name="values", type="string", description='JSON object mapping selectors to values, e.g. \'{"#email": "test@example.com"}\''),
        ],
        handler=fill_form,
    ))

    registry.register(Tool(
        name="navigate_to",
        description="Navigate to a URL and wait for the page to load",
        parameters=[
            ToolParameter(name="url", type="string", description="URL to navigate to"),
        ],
        handler=navigate_to,
    ))
