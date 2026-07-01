"""Code Agent - primary worker for reading, writing, and editing code."""

from __future__ import annotations

from pathlib import Path

from coding_agent.core.agent import Agent, AgentConfig
from coding_agent.core.tool import ToolRegistry
from coding_agent.intelligence.repomap import RepoMap
from coding_agent.models.provider import LLMProvider


def _is_own_codebase(root: Path) -> bool:
    markers = [
        root / "src" / "coding_agent" / "main.py",
        root / "src" / "coding_agent" / "core" / "agent.py",
    ]
    return all(m.exists() for m in markers)


def _load_architecture(root: Path) -> str:
    agents_md = root / "AGENTS.md"
    if agents_md.exists():
        content = agents_md.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        return "\n".join(lines[:60])
    return ""


def create_code_agent(
    provider: LLMProvider,
    tools: ToolRegistry,
    project_root: str = ".",
) -> Agent:
    config = AgentConfig(
        name="code",
        model_provider=provider,
        max_iterations=10,
        max_tool_retries=2,
        token_budget=8192,
        enable_preprocessing=True,
    )

    root = Path(project_root).resolve()
    arch = _load_architecture(root) if _is_own_codebase(root) else ""
    repo_map = RepoMap(root=str(root), map_tokens=1024)

    class CodeAgent(Agent):
        def get_system_prompt(self) -> str:
            tool_list = self.get_tool_summary()
            parts = [f"""You are an expert coding agent. You read, write, and edit source code with precision.

Available tools:
{tool_list}"""]

            if arch:
                parts.append(f"""You are running on your OWN codebase. Architecture reference:
{arch}
When modifying yourself:
- Run `pytest tests/` after changes to verify
- Run `ruff check src/coding_agent/` for lint
- Follow existing patterns (dataclasses, type hints)
- Read the file first, then edit""" )

            parts.append("""Rules:
1. ALWAYS read a file before editing it - understand the current state first
2. Make precise, minimal changes - never rewrite entire files when a small edit works
3. Use edit_file for small targeted changes, write_file only for new files
4. After editing, verify the change by reading the affected section
5. Follow the existing code style (indentation, naming, patterns)
6. If a tool call fails, analyze the error and try a different approach
7. Never modify files outside the project root
8. For multiple related changes, make them in logical dependency order
9. Use grep/search_files to find relevant code before making changes
10. When edit_file returns a diff, verify it looks correct
11. Use @file(path) to reference files, @folder(path) for directories, @codebase for full context
12. Use list_rules to check project-specific conventions before editing

When done, provide a clear summary:
- Which files were created/modified/deleted
- What changed in each file
- Any follow-up actions needed""")

            return "\n\n".join(parts)

    agent = CodeAgent(config=config, tool_registry=tools)
    agent.set_repo_map(repo_map)
    return agent
