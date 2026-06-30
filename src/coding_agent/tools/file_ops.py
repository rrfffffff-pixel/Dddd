"""File operation tools for agents."""

from __future__ import annotations

import os
from pathlib import Path

from coding_agent.core.tool import Tool, ToolParameter, ToolRegistry


def register_file_tools(registry: ToolRegistry, project_root: str = ".") -> None:
    root = Path(project_root).resolve()

    def read_file(path: str) -> str:
        full = (root / path).resolve()
        if not full.exists():
            return f"Error: File not found: {path}"
        if not str(full).startswith(str(root)):
            return "Error: Access denied - path outside project root"
        try:
            return full.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading {path}: {e}"

    def write_file(path: str, content: str) -> str:
        full = (root / path).resolve()
        if not str(full).startswith(str(root)):
            return "Error: Access denied - path outside project root"
        try:
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(content, encoding="utf-8")
            return f"Successfully wrote {path} ({len(content)} bytes)"
        except Exception as e:
            return f"Error writing {path}: {e}"

    def edit_file(path: str, old_string: str, new_string: str) -> str:
        full = (root / path).resolve()
        if not full.exists():
            return f"Error: File not found: {path}"
        if not str(full).startswith(str(root)):
            return "Error: Access denied - path outside project root"
        try:
            content = full.read_text(encoding="utf-8")
            if old_string not in content:
                return f"Error: old_string not found in {path}"
            count = content.count(old_string)
            if count > 1:
                return f"Error: old_string found {count} times - provide more context"
            new_content = content.replace(old_string, new_string, 1)
            full.write_text(new_content, encoding="utf-8")
            return f"Successfully edited {path}"
        except Exception as e:
            return f"Error editing {path}: {e}"

    def list_files(directory: str = ".") -> str:
        full = (root / directory).resolve()
        if not str(full).startswith(str(root)):
            return "Error: Access denied - path outside project root"
        try:
            entries = []
            for item in sorted(full.iterdir()):
                if item.name.startswith(".") and item.name not in (".gitignore",):
                    continue
                if item.name in ("node_modules", "__pycache__", ".git", "venv", ".venv"):
                    continue
                prefix = "  " if item.is_file() else "d "
                entries.append(f"{prefix}{item.relative_to(root)}")
            return "\n".join(entries) if entries else "Empty directory"
        except Exception as e:
            return f"Error listing {directory}: {e}"

    def search_files(pattern: str, directory: str = ".") -> str:
        """Search for files matching a glob pattern."""
        full = (root / directory).resolve()
        if not str(full).startswith(str(root)):
            return "Error: Access denied"
        try:
            matches = []
            for match in full.rglob(pattern):
                if match.is_file():
                    rel = match.relative_to(root)
                    if not any(skip in str(rel) for skip in [".git", "node_modules", "__pycache__"]):
                        matches.append(str(rel))
            return "\n".join(matches[:50]) if matches else "No matches found"
        except Exception as e:
            return f"Error: {e}"

    registry.register(Tool(
        name="read_file",
        description="Read the contents of a file",
        parameters=[
            ToolParameter(name="path", type="string", description="File path relative to project root"),
        ],
        handler=read_file,
    ))

    registry.register(Tool(
        name="write_file",
        description="Write content to a file (creates parent directories)",
        parameters=[
            ToolParameter(name="path", type="string", description="File path relative to project root"),
            ToolParameter(name="content", type="string", description="Content to write"),
        ],
        handler=write_file,
    ))

    registry.register(Tool(
        name="edit_file",
        description="Edit a file by replacing an exact string match",
        parameters=[
            ToolParameter(name="path", type="string", description="File path"),
            ToolParameter(name="old_string", type="string", description="Exact string to find"),
            ToolParameter(name="new_string", type="string", description="Replacement string"),
        ],
        handler=edit_file,
    ))

    registry.register(Tool(
        name="list_files",
        description="List files in a directory",
        parameters=[
            ToolParameter(name="directory", type="string", description="Directory path", required=False, default="."),
        ],
        handler=list_files,
    ))

    registry.register(Tool(
        name="search_files",
        description="Search for files by glob pattern (e.g. *.py, **/*.js)",
        parameters=[
            ToolParameter(name="pattern", type="string", description="Glob pattern"),
            ToolParameter(name="directory", type="string", description="Directory to search in", required=False, default="."),
        ],
        handler=search_files,
    ))
