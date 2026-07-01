"""File operation tools for agents."""

from __future__ import annotations

import difflib
import re
from pathlib import Path

from coding_agent.core.tool import Tool, ToolParameter, ToolRegistry


def _compute_diff(path: str, old_content: str, new_content: str) -> str:
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{path}", tofile=f"b/{path}",
        n=3,
    )
    return "".join(diff)


def apply_search_replace(content: str, search: str, replace: str) -> tuple[str, str] | None:
    """Apply a SEARCH/REPLACE edit with fuzzy matching. Returns (new_content, match_type) or None."""
    search_stripped = search.strip()
    replace_stripped = replace.strip()

    # Strategy 1: Exact match
    idx = content.find(search_stripped)
    if idx != -1:
        new_content = content[:idx] + replace_stripped + content[idx + len(search_stripped):]
        return new_content, "exact match"

    # Strategy 2: Whitespace-flexible match
    search_normalized = re.sub(r'\s+', ' ', search_stripped)
    content_normalized = re.sub(r'\s+', ' ', content)
    idx = content_normalized.find(search_normalized)
    if idx != -1:
        orig_map = _map_normalized_to_original(content, content_normalized)
        if idx in orig_map:
            orig_start = orig_map[idx]
            orig_end = orig_map[idx + len(search_normalized)]
            new_content = content[:orig_start] + replace_stripped + content[orig_end:]
            return new_content, "whitespace-flexible match"

    # Strategy 3: SequenceMatcher similarity
    matcher = difflib.SequenceMatcher(None, search_stripped, content)
    ratio = matcher.ratio()
    if ratio > 0.85:
        for match_found in matcher.get_matching_blocks():
            if match_found.size == len(search_stripped):
                new_content = (content[:match_found.b] +
                               replace_stripped +
                               content[match_found.b + match_found.size:])
                return new_content, f"fuzzy match (similarity={ratio:.2f})"

    # Strategy 4: Line-level matching
    search_lines = search_stripped.splitlines()
    content_lines = content.splitlines(keepends=False)
    for i in range(len(content_lines) - len(search_lines) + 1):
        match = True
        for j, sline in enumerate(search_lines):
            if content_lines[i + j].strip() != sline.strip():
                match = False
                break
        if match:
            start_char = sum(len(line) + 1 for line in content_lines[:i])
            end_char = sum(len(line) + 1 for line in content_lines[:i + len(search_lines)])
            new_content = (content[:start_char] +
                           replace_stripped +
                           content[end_char:])
            return new_content, "line-level match"

    return None


def _map_normalized_to_original(original: str, normalized: str) -> dict[int, int]:
    mapping = {}
    orig_pos = 0
    norm_pos = 0
    while norm_pos < len(normalized) and orig_pos < len(original):
        if original[orig_pos] == normalized[norm_pos]:
            mapping[norm_pos] = orig_pos
            orig_pos += 1
            norm_pos += 1
        else:
            orig_pos += 1
    mapping[len(normalized)] = len(original)
    return mapping


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
            diff = _compute_diff(path, content, new_content)
            full.write_text(new_content, encoding="utf-8")
            return f"Successfully edited {path}\nDiff:\n{diff}"
        except Exception as e:
            return f"Error editing {path}: {e}"

    def delete_file(path: str) -> str:
        full = (root / path).resolve()
        if not full.exists():
            return f"Error: File not found: {path}"
        if not str(full).startswith(str(root)):
            return "Error: Access denied - path outside project root"
        try:
            full.unlink()
            return f"Successfully deleted {path}"
        except Exception as e:
            return f"Error deleting {path}: {e}"

    def move_file(source: str, destination: str) -> str:
        src = (root / source).resolve()
        dst = (root / destination).resolve()
        if not str(src).startswith(str(root)) or not str(dst).startswith(str(root)):
            return "Error: Access denied - path outside project root"
        if not src.exists():
            return f"Error: Source not found: {source}"
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            return f"Successfully moved {source} -> {destination}"
        except Exception as e:
            return f"Error moving {source}: {e}"

    def list_files(directory: str = ".") -> str:
        full = (root / directory).resolve()
        if not str(full).startswith(str(root)):
            return "Error: Access denied - path outside project root"
        try:
            entries = []
            for item in sorted(full.iterdir()):
                if item.name.startswith(".") and item.name not in (".gitignore", ".env"):
                    continue
                if item.name in ("node_modules", "__pycache__", ".git", "venv", ".venv"):
                    continue
                size = item.stat().st_size if item.is_file() else 0
                prefix = f"  {size:>8d} " if item.is_file() else "d         "
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

    def search_replace(path: str, search: str, replace: str) -> str:
        """Apply a SEARCH/REPLACE edit with fuzzy matching."""
        full = (root / path).resolve()
        if not full.exists():
            return f"Error: File not found: {path}"
        if not str(full).startswith(str(root)):
            return "Error: Access denied - path outside project root"
        try:
            content = full.read_text(encoding="utf-8")

            result = apply_search_replace(content, search, replace)
            if result is None:
                return f"Error: Could not match search text in {path}"

            new_content, match_desc = result
            diff = _compute_diff(path, content, new_content)
            full.write_text(new_content, encoding="utf-8")
            return f"Successfully edited {path} ({match_desc})\nDiff:\n{diff}"
        except Exception as e:
            return f"Error editing {path}: {e}"


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
        name="delete_file",
        description="Delete a file permanently",
        parameters=[
            ToolParameter(name="path", type="string", description="File path relative to project root"),
        ],
        handler=delete_file,
    ))

    registry.register(Tool(
        name="move_file",
        description="Move or rename a file",
        parameters=[
            ToolParameter(name="source", type="string", description="Source path"),
            ToolParameter(name="destination", type="string", description="Destination path"),
        ],
        handler=move_file,
    ))

    registry.register(Tool(
        name="list_files",
        description="List files and directories with sizes",
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

    registry.register(Tool(
        name="search_replace",
        description="Edit a file using SEARCH/REPLACE with fuzzy matching. Provide the SEARCH text to find and the REPLACE text to replace it with. Supports exact, whitespace-flexible, and fuzzy matching.",
        parameters=[
            ToolParameter(name="path", type="string", description="File path relative to project root"),
            ToolParameter(name="search", type="string", description="Text to search for (SEARCH block)"),
            ToolParameter(name="replace", type="string", description="Text to replace with (REPLACE block)"),
        ],
        handler=search_replace,
    ))
