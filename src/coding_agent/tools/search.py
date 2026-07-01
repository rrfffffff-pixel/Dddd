"""Code search tools - grep and content search."""

from __future__ import annotations

import re
from pathlib import Path

from coding_agent.core.tool import Tool, ToolParameter, ToolRegistry

BINARY_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf",
                     ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
                     ".so", ".dll", ".dylib", ".exe", ".bin",
                     ".pyc", ".pyo", ".whl", ".egg",
                     ".woff", ".woff2", ".ttf", ".eot",
                     ".mp3", ".mp4", ".avi", ".mov", ".wav",
                     ".o", ".a", ".lib", ".obj"}


def register_search_tools(registry: ToolRegistry, project_root: str = ".") -> None:
    root = Path(project_root).resolve()

    def grep(pattern: str, directory: str = ".", include: str = "", context: int = 0) -> str:
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
            if path.suffix in BINARY_EXTENSIONS:
                continue
            if any(s in str(path) for s in skip):
                continue
            if file_pattern and not file_pattern.search(path.name):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines()
                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        rel = path.relative_to(root)
                        if context > 0:
                            ctx_start = max(0, i - 1 - context)
                            ctx_end = min(len(lines), i + context)
                            ctx_lines = []
                            for ci in range(ctx_start, ctx_end):
                                marker = ">" if ci == i - 1 else " "
                                ctx_lines.append(f"{rel}:{ci + 1}:{marker} {lines[ci][:120]}")
                            results.extend(ctx_lines)
                            results.append("---")
                        else:
                            results.append(f"{rel}:{i}: {line.strip()[:120]}")
                        if len(results) >= 100:
                            return "\n".join(results) + "\n... (truncated)"
            except (UnicodeDecodeError, Exception):
                continue
        return "\n".join(results) if results else "No matches found"

    registry.register(Tool(
        name="grep",
        description="Search file contents using regex pattern",
        parameters=[
            ToolParameter(name="pattern", type="string", description="Regex pattern to search for"),
            ToolParameter(name="directory", type="string", description="Directory to search in", required=False, default="."),
            ToolParameter(name="include", type="string", description="File pattern filter (e.g. *.py)", required=False, default=""),
            ToolParameter(name="context", type="integer", description="Number of context lines around matches", required=False, default=0),
        ],
        handler=grep,
    ))
