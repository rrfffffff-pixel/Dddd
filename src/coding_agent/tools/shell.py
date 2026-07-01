"""Shell execution tools."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from coding_agent.core.tool import Tool, ToolParameter, ToolRegistry

DESTRUCTIVE_PATTERNS = [
    r"(^|\s)rm\s+(-rf?\s+)?\/\s",
    r"(^|\s)rm\s+(-rf?\s+)?/\s",
    r"(^|\s)mkfs\.",
    r"(^|\s)dd\s+if=",
    r"(^|\s)format\s+",
    r"(^|\s):\(\)\s*\{",
    r"(^|\s)>\s+/dev/",
    r"(^|\s)chmod\s+000\s+/\s",
    r"(^|\s)mv\s+/\s+",
    r"(^|\s)shutdown",
    r"(^|\s)reboot",
    r"(^|\s)poweroff",
    r"(^|\s)init\s+0",
    r"(^|\s)init\s+6",
]


def register_shell_tools(registry: ToolRegistry, project_root: str = ".") -> None:
    root = Path(project_root).resolve()

    def run_command(command: str, timeout: int = 30) -> str:
        for pattern in DESTRUCTIVE_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return f"Error: Command blocked for safety: {command[:100]}"
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output_parts = []
            if result.stdout:
                output_parts.append(result.stdout.rstrip())
            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr.rstrip()}")
            if result.returncode != 0:
                output_parts.append(f"Exit code: {result.returncode}")
            output = "\n".join(output_parts)
            return output.strip()[:10000] if output.strip() else "Command completed (no output)"
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"

    registry.register(Tool(
        name="run_command",
        description="Execute a shell command in the project directory",
        parameters=[
            ToolParameter(name="command", type="string", description="Shell command to execute"),
            ToolParameter(name="timeout", type="integer", description="Timeout in seconds", required=False, default=30),
        ],
        handler=run_command,
    ))
