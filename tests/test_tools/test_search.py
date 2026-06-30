"""Tests for search tools."""

import tempfile
from pathlib import Path

from coding_agent.core.tool import ToolRegistry
from coding_agent.tools.search import register_search_tools


def test_grep():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "app.py").write_text("def hello():\n    pass\n\ndef world():\n    pass")
        tools = ToolRegistry()
        register_search_tools(tools, tmpdir)

        result = tools.execute("grep", pattern="def \\w+")
        assert "hello" in result
        assert "world" in result


def test_grep_no_matches():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "app.py").write_text("x = 1")
        tools = ToolRegistry()
        register_search_tools(tools, tmpdir)

        result = tools.execute("grep", pattern="nonexistent_pattern_xyz")
        assert "No matches" in result
