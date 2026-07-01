"""Self-awareness tools for the coding agent."""

from __future__ import annotations

from pathlib import Path

from coding_agent.core.tool import Tool, ToolRegistry


def _is_own_codebase(root: Path) -> bool:
    """Detect if the project is the coding agent itself."""
    markers = [
        root / "src" / "coding_agent" / "main.py",
        root / "src" / "coding_agent" / "core" / "agent.py",
        root / "pyproject.toml",
    ]
    return all(m.exists() for m in markers)


def register_info_tools(registry: ToolRegistry, project_root: str = ".") -> None:
    root = Path(project_root).resolve()

    def self_info() -> str:
        """Return the coding agent's own architecture documentation."""
        agents_md = root / "AGENTS.md"
        arch_md = root / "ARCHITECTURE.md"
        if agents_md.exists():
            return agents_md.read_text(encoding="utf-8", errors="replace")
        if arch_md.exists():
            return arch_md.read_text(encoding="utf-8", errors="replace")
        return "No architecture documentation found (AGENTS.md or ARCHITECTURE.md)"

    def project_type() -> str:
        """Auto-detect the project type."""
        if _is_own_codebase(root):
            return "coding-agent"
        if (root / "pyproject.toml").exists() and (root / "src").exists():
            return "python-package"
        if (root / "setup.py").exists() or (root / "setup.cfg").exists():
            return "python-package"
        if (root / "requirements.txt").exists():
            return "python"
        if (root / "package.json").exists():
            return "node"
        if (root / "Cargo.toml").exists():
            return "rust"
        if (root / "go.mod").exists():
            return "go"
        return "unknown"

    registry.register(Tool(
        name="self_info",
        description="Get the coding agent's own architecture documentation - useful when modifying this codebase",
        parameters=[],
        handler=self_info,
    ))

    registry.register(Tool(
        name="project_type",
        description="Auto-detect the current project type (coding-agent, python-package, node, rust, go, etc.)",
        parameters=[],
        handler=project_type,
    ))
