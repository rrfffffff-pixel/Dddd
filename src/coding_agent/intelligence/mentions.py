"""@mentions context parser - like Cursor's @file, @folder, @codebase.

Parses @mentions from task descriptions and resolves them to file contents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Mention:
    type: str  # file, folder, codebase, web
    target: str
    resolved: str = ""
    content: str = ""


def parse_mentions(text: str, project_root: str = ".") -> list[Mention]:
    root = Path(project_root).resolve()
    mentions = []

    patterns = [
        (r"@codebase\b", "codebase", ""),
        (r"@web\b", "web", ""),
        (r"@folder\(([^)]+)\)", "folder", ""),
        (r"@file\(([^)]+)\)", "file", ""),
        (r"@([\w/.-]+\.\w+)", "file", ""),
    ]

    for pattern, mtype, _ in patterns:
        for match in re.finditer(pattern, text):
            target = match.group(1) if match.lastindex else ""
            mention = Mention(type=mtype, target=target)

            if mtype == "codebase":
                mention.resolved = "entire codebase"
                files = list(root.rglob("*"))
                src_files = [f for f in files if f.is_file() and f.suffix in (".py", ".js", ".ts", ".rs", ".go")]
                mention.content = f"Codebase has {len(src_files)} source files"
            elif mtype == "file" and target:
                fp = root / target
                if fp.exists() and fp.is_file():
                    mention.resolved = str(fp)
                    try:
                        mention.content = fp.read_text(encoding="utf-8", errors="replace")[:3000]
                    except Exception:
                        mention.content = f"Error reading {target}"
                else:
                    mention.content = f"File not found: {target}"
            elif mtype == "folder" and target:
                fp = root / target
                if fp.is_dir():
                    mention.resolved = str(fp)
                    entries = [str(p.relative_to(root)) for p in sorted(fp.iterdir()) if not p.name.startswith(".")]
                    mention.content = "Folder contents:\n" + "\n".join(entries[:30])
                else:
                    mention.content = f"Folder not found: {target}"

            mentions.append(mention)

    return mentions


def expand_mentions(text: str, project_root: str = ".") -> str:
    mentions = parse_mentions(text, project_root)
    if not mentions:
        return text

    parts = [text, "\n\nReferenced context:"]
    for m in mentions:
        if m.type == "codebase":
            parts.append(f"\n[Codebase: {m.content}]")
        elif m.type == "file" and m.content:
            parts.append(f"\n--- {m.target} ---\n{m.content}")
        elif m.type == "folder":
            parts.append(f"\n--- {m.target}/ ---\n{m.content}")

    return "\n".join(parts)
