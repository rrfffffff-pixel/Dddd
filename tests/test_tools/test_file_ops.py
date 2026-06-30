"""Tests for file operation tools."""

import os
import tempfile
from pathlib import Path

from coding_agent.core.tool import ToolRegistry
from coding_agent.tools.file_ops import register_file_tools


def test_read_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        (Path(tmpdir) / "test.py").write_text("print('hello')")
        tools = ToolRegistry()
        register_file_tools(tools, tmpdir)

        result = tools.execute("read_file", path="test.py")
        assert result == "print('hello')"


def test_write_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        tools = ToolRegistry()
        register_file_tools(tools, tmpdir)

        result = tools.execute("write_file", path="new.py", content="x = 1")
        assert "Successfully" in result

        content = (Path(tmpdir) / "new.py").read_text()
        assert content == "x = 1"


def test_edit_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "edit.py").write_text("x = 1\ny = 2")
        tools = ToolRegistry()
        register_file_tools(tools, tmpdir)

        result = tools.execute("edit_file", path="edit.py", old_string="x = 1", new_string="x = 10")
        assert "Successfully" in result

        content = (Path(tmpdir) / "edit.py").read_text()
        assert "x = 10" in content
        assert "y = 2" in content


def test_list_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "a.py").touch()
        (Path(tmpdir) / "b.py").touch()
        tools = ToolRegistry()
        register_file_tools(tools, tmpdir)

        result = tools.execute("list_files", directory=".")
        assert "a.py" in result
        assert "b.py" in result


def test_search_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "test.py").touch()
        (Path(tmpdir) / "app.js").touch()
        tools = ToolRegistry()
        register_file_tools(tools, tmpdir)

        result = tools.execute("search_files", pattern="*.py")
        assert "test.py" in result
        assert "app.js" not in result


def test_read_file_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        tools = ToolRegistry()
        register_file_tools(tools, tmpdir)

        result = tools.execute("read_file", path="nonexistent.py")
        assert "Error" in result


def test_edit_file_not_found():
    with tempfile.TemporaryDirectory() as tmpdir:
        tools = ToolRegistry()
        register_file_tools(tools, tmpdir)

        result = tools.execute("edit_file", path="missing.py", old_string="a", new_string="b")
        assert "Error" in result
