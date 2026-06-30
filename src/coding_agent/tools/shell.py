"""Shell execution tools."""

from __future__ import annotations

import subprocess
from pathlib import Path

from coding_agent.core.tool import Tool, ToolParameter, ToolRegistry


def register_shell_tools(registry: ToolRegistry, project_root: str = ".") -> None:
    root = Path(project_root).resolve()

    def run_command(command: str, timeout: int = 30) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}" if output else result.stderr
            if result.returncode != 0:
                output += f"\n[Exit code: {result.returncode}]"
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
