"""Code search tools - grep and content search."""

from __future__ import annotations

import re
from pathlib import Path

from coding_agent.core.tool import Tool, ToolParameter, ToolRegistry


def register_search_tools(registry: ToolRegistry, project_root: str = ".") -> None:
    root = Path(project_root).resolve()

    def grep(pattern: str, directory: str = ".", include: str = "") -> str:
        """Search file contents for a regex pattern."""
        full = (root / directory).resolve()
        if not str(full).startswith(str(root)):
            return "Error: Access denied"
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Invalid regex: {e}"

        results = []
        skip = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mimocode"}
        file_pattern = re.compile(include) if include else None

        for path in full.rglob("*"):
            if not path.is_file():
                continue
            if any(s in str(path) for s in skip):
                continue
            if file_pattern and not file_pattern.search(path.name):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        rel = path.relative_to(root)
                        results.append(f"{rel}:{i}: {line.strip()[:120]}")
                        if len(results) >= 100:
                            return "\n".join(results) + "\n... (truncated)"
            except Exception:
                continue
        return "\n".join(results) if results else "No matches found"

    registry.register(Tool(
        name="grep",
        description="Search file contents using regex pattern",
        parameters=[
            ToolParameter(name="pattern", type="string", description="Regex pattern to search for"),
            ToolParameter(name="directory", type="string", description="Directory to search in", required=False, default="."),
            ToolParameter(name="include", type="string", description="File pattern filter (e.g. *.py)", required=False, default=""),
        ],
        handler=grep,
    ))
