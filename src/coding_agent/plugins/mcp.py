"""Basic MCP (Model Context Protocol) plugin support.

Allows agents to use external tools via MCP-compatible servers.

MCP spec: https://modelcontextprotocol.io
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path



@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict = field(default_factory=dict)
    server_name: str = ""


@dataclass
class MCPServer:
    name: str
    command: str
    args: list[str] = field(default_factory=list)


class MCPManager:
    """Manages MCP tool connections."""

    def __init__(self) -> None:
        self._servers: dict[str, MCPServer] = {}
        self._tools: dict[str, MCPTool] = {}

    def load_config(self, project_root: str = ".") -> None:
        config_path = Path(project_root).resolve() / ".coding-agent" / "mcp.json"
        if not config_path.exists():
            return
        try:
            data = json.loads(config_path.read_text())
            for name, cfg in data.get("servers", {}).items():
                server = MCPServer(
                    name=name,
                    command=cfg.get("command", ""),
                    args=cfg.get("args", []),
                )
                self._servers[name] = server
        except Exception:
            pass

    def discover_tools(self) -> list[MCPTool]:
        for server in self._servers.values():
            try:
                result = subprocess.run(
                    [server.command, *server.args, "list-tools"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    for t in data.get("tools", []):
                        tool = MCPTool(
                            name=t["name"],
                            description=t.get("description", ""),
                            input_schema=t.get("inputSchema", {}),
                            server_name=server.name,
                        )
                        self._tools[tool.name] = tool
            except Exception:
                pass
        return list(self._tools.values())

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        tool = self._tools.get(tool_name)
        if not tool:
            return f"Error: MCP tool '{tool_name}' not found"
        server = self._servers.get(tool.server_name)
        if not server:
            return f"Error: Server '{tool.server_name}' not found"
        try:
            payload = json.dumps({"name": tool_name, "arguments": arguments})
            result = subprocess.run(
                [server.command, *server.args, "call-tool", payload],
                capture_output=True, text=True, timeout=30,
            )
            return result.stdout or result.stderr or "No output"
        except subprocess.TimeoutExpired:
            return f"Error: MCP tool '{tool_name}' timed out"
        except Exception as e:
            return f"Error calling MCP tool '{tool_name}': {e}"

    def has_tools(self) -> bool:
        return len(self._tools) > 0
