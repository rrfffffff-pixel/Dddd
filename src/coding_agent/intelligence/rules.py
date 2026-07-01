"""Rules system - like Cursor's .cursor/rules/ directory.

Loads rule files from .coding-agent/rules/*.mdc.
Each rule file has YAML frontmatter and markdown body.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Rule:
    name: str
    description: str = ""
    globs: list[str] = field(default_factory=lambda: ["**/*"])
    body: str = ""
    source: str = ""


RULES_DIR = ".coding-agent/rules"


def _parse_mdc(path: Path) -> Rule | None:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    name = path.stem
    description = ""
    globs = ["**/*"]
    body = content

    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if frontmatter_match:
        raw = frontmatter_match.group(1)
        body = content[frontmatter_match.end():].strip()
        for line in raw.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                val = val.strip()
                if key.strip() == "description":
                    description = val
                elif key.strip() == "globs":
                    globs = [g.strip().strip('"') for g in val.split(",")]

    return Rule(name=name, description=description, globs=globs, body=body, source=str(path))


def load_rules(project_root: str = ".") -> list[Rule]:
    rules_dir = Path(project_root).resolve() / RULES_DIR
    if not rules_dir.is_dir():
        return []
    rules = []
    for path in sorted(rules_dir.glob("*.mdc")):
        rule = _parse_mdc(path)
        if rule:
            rules.append(rule)
    return rules


def get_rules_for_file(rules: list[Rule], file_path: str) -> list[Rule]:
    from fnmatch import fnmatch
    matched = []
    for rule in rules:
        for g in rule.globs:
            if fnmatch(file_path, g):
                matched.append(rule)
                break
    return matched


def rules_summary(rules: list[Rule]) -> str:
    if not rules:
        return ""
    lines = ["Project Rules:"]
    for r in rules:
        lines.append(f"  - {r.name}: {r.description or r.body[:80]}")
    return "\n".join(lines)


def example_rules_dir(project_root: str = ".") -> None:
    rules_dir = Path(project_root).resolve() / RULES_DIR
    rules_dir.mkdir(parents=True, exist_ok=True)
    example = rules_dir / "example.mdc"
    if not example.exists():
        example.write_text("""---
description: Example rule for coding conventions
globs: **/*.py
---

# Python Coding Conventions

- Use 4-space indentation
- Use type hints for all function signatures
- Follow PEP 8 style guide
- Write docstrings for all public functions
""")
