"""Tool registration and execution framework for agents."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class Tool:
    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    handler: Callable[..., Any] | None = None
    cacheable: bool = False

    def to_schema(self) -> dict:
        props = {}
        required = []
        for p in self.parameters:
            props[p.name] = {
                "type": p.type,
                "description": p.description,
            }
            if p.default is not None:
                props[p.name]["default"] = p.default
            elif p.required:
                required.append(p.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        }

    def validate(self, args: dict) -> list[str]:
        errors = []
        for p in self.parameters:
            if p.required and p.default is None and p.name not in args:
                errors.append(f"Missing required parameter: {p.name}")
        return errors

    def execute(self, **kwargs: Any) -> Any:
        if self.handler is None:
            raise NotImplementedError(f"Tool {self.name} has no handler")
        return self.handler(**kwargs)


class ToolRegistry:
    """Registry of available tools for agents."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._cache: dict[str, Any] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def register_function(
        self,
        name: str,
        description: str,
        parameters: list[ToolParameter],
        handler: Callable[..., Any],
    ) -> None:
        self.register(Tool(name=name, description=description, parameters=parameters, handler=handler))

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def to_schemas(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]

    def _cache_key(self, name: str, kwargs: dict) -> str:
        raw = json.dumps({"name": name, "args": kwargs}, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    def execute(self, name: str, **kwargs: Any) -> Any:
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")

        errors = tool.validate(kwargs)
        if errors:
            return f"Validation error: {'; '.join(errors)}"

        if tool.cacheable:
            key = self._cache_key(name, kwargs)
            if key in self._cache:
                return self._cache[key]

        result = tool.execute(**kwargs)

        if tool.cacheable:
            self._cache[self._cache_key(name, kwargs)] = result

        return result

    def clear_cache(self) -> None:
        self._cache.clear()

    def merge(self, other: ToolRegistry) -> None:
        for tool in other.list_tools():
            self.register(tool)
