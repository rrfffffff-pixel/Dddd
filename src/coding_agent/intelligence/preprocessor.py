"""Pre-processing pipeline to reduce LLM calls - lexical analysis, token optimization."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileAnalysis:
    path: str
    language: str = ""
    size_bytes: int = 0
    line_count: int = 0
    imports: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    has_tests: bool = False
    complexity: int = 0  # rough estimate


@dataclass
class TaskAnalysis:
    needs_llm: bool = True
    confidence: float = 0.0
    suggested_agent: str = "code"
    relevant_files: list[str] = field(default_factory=list)
    estimated_tokens: int = 0
    static_answer: str = ""


class LexicalAnalyzer:
    """Analyze code without LLM - pure static analysis."""

    EXTENSION_MAP = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "javascript", ".tsx": "typescript", ".go": "go",
        ".rs": "rust", ".java": "java", ".rb": "ruby",
        ".sh": "shell", ".yaml": "yaml", ".yml": "yaml",
        ".json": "json", ".toml": "toml", ".md": "markdown",
    }

    def analyze_file(self, path: str) -> FileAnalysis:
        analysis = FileAnalysis(path=path)
        try:
            p = Path(path)
            if not p.exists():
                return analysis

            analysis.size_bytes = p.stat().st_size
            content = p.read_text(encoding="utf-8", errors="replace")
            analysis.line_count = len(content.splitlines())
            analysis.language = self.EXTENSION_MAP.get(p.suffix, "unknown")

            if analysis.language == "python":
                self._analyze_python(content, analysis)
            elif analysis.language in ("javascript", "typescript"):
                self._analyze_js(content, analysis)

        except Exception:
            pass
        return analysis

    def _analyze_python(self, content: str, analysis: FileAnalysis) -> None:
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis.imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        analysis.imports.append(node.module)
                elif isinstance(node, ast.FunctionDef):
                    analysis.functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    analysis.classes.append(node.name)
            analysis.complexity = len(analysis.functions) + len(analysis.classes) * 2
        except SyntaxError:
            pass

    def _analyze_js(self, content: str, analysis: FileAnalysis) -> None:
        analysis.imports = re.findall(r"""(?:import|require)\s*\(?['"]([^'"]+)['"]""", content)
        analysis.functions = re.findall(r"""(?:function|const|let|var)\s+(\w+)\s*[=(]""", content)
        analysis.classes = re.findall(r'class\s+(\w+)', content)
        analysis.complexity = len(analysis.functions) + len(analysis.classes) * 2

    def analyze_project(self, root: str) -> dict[str, FileAnalysis]:
        analyses = {}
        skip = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mimocode", "dist", "build"}
        for path in Path(root).rglob("*"):
            if not path.is_file():
                continue
            if any(s in str(path) for s in skip):
                continue
            if path.suffix in self.EXTENSION_MAP:
                analyses[str(path.relative_to(root))] = self.analyze_file(str(path))
        return analyses


class PromptCompressor:
    """Compress prompts to reduce token usage."""

    def compress_file_content(self, content: str, path: str, max_lines: int = 100) -> str:
        lines = content.splitlines()
        if len(lines) <= max_lines:
            return content

        # Keep first 20 lines, last 10, and evenly spaced middle
        head = lines[:20]
        tail = lines[-10:]
        middle_start = 20
        middle_end = len(lines) - 10
        step = max(1, (middle_end - middle_start) // (max_lines - 30))
        middle = lines[middle_start:middle_end:step]

        return "\n".join(
            head + [f"... ({len(lines) - 30} lines omitted) ..."] + middle + tail
        )

    def compress_tool_result(self, result: str, tool_name: str, max_chars: int = 4000) -> str:
        if len(result) <= max_chars:
            return result

        if tool_name == "read_file":
            lines = result.splitlines()
            if len(lines) > 80:
                return "\n".join(lines[:40]) + f"\n... ({len(lines) - 60} lines) ...\n" + "\n".join(lines[-20:])
        elif tool_name == "grep":
            lines = result.splitlines()
            if len(lines) > 50:
                return "\n".join(lines[:30]) + f"\n... ({len(lines) - 40} matches) ...\n" + "\n".join(lines[-10:])

        return result[:max_chars] + f"\n... (truncated from {len(result)} chars)"

    def minify_system_prompt(self, prompt: str) -> str:
        lines = prompt.strip().splitlines()
        result = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("- ") or stripped.startswith("* "):
                result.append(stripped)
            elif stripped[0].isdigit() and ". " in stripped[:4]:
                result.append(stripped)
            else:
                result.append(stripped)
        return "\n".join(result)


class CacheLayer:
    """Cache LLM responses and tool results to avoid redundant calls."""

    def __init__(self) -> None:
        self._llm_cache: dict[str, str] = {}
        self._tool_cache: dict[str, str] = {}
        self._file_hashes: dict[str, str] = {}

    def file_changed(self, path: str, content: str) -> bool:
        new_hash = hashlib.md5(content.encode()).hexdigest()
        old_hash = self._file_hashes.get(path)
        self._file_hashes[path] = new_hash
        return old_hash != new_hash

    def get_llm_cache(self, prompt_hash: str) -> str | None:
        return self._llm_cache.get(prompt_hash)

    def set_llm_cache(self, prompt_hash: str, response: str) -> None:
        self._llm_cache[prompt_hash] = response

    def get_tool_cache(self, key: str) -> str | None:
        return self._tool_cache.get(key)

    def set_tool_cache(self, key: str, result: str) -> None:
        self._tool_cache[key] = result

    def clear(self) -> None:
        self._llm_cache.clear()
        self._tool_cache.clear()
        self._file_hashes.clear()


class TaskClassifier:
    """Classify tasks without LLM to skip unnecessary LLM calls."""

    STATIC_PATTERNS = {
        "list_files": [
            (r"list\s+(all\s+)?files", "code"),
            (r"show\s+(me\s+)?(the\s+)?files", "code"),
            (r"what\s+files", "code"),
        ],
        "search": [
            (r"find\s+(all\s+)?", "code"),
            (r"search\s+(for\s+)?", "code"),
            (r"grep\s+", "code"),
        ],
        "test": [
            (r"run\s+(the\s+)?tests?", "test"),
            (r"test\s+(the\s+)?code", "test"),
            (r"check\s+if.*works", "test"),
        ],
        "install": [
            (r"install\s+", "shell"),
            (r"setup\s+", "shell"),
            (r"pip\s+install", "shell"),
        ],
        "review": [
            (r"review\s+", "review"),
            (r"check\s+for\s+(bugs|issues)", "review"),
            (r"audit\s+", "review"),
        ],
    }

    def classify(self, task: str) -> TaskAnalysis:
        lower = task.lower().strip()
        for category, patterns in self.STATIC_PATTERNS.items():
            for pattern, agent in patterns:
                if re.search(pattern, lower):
                    return TaskAnalysis(
                        needs_llm=True,  # Still needs LLM, just suggests agent
                        confidence=0.8,
                        suggested_agent=agent,
                    )

        return TaskAnalysis(needs_llm=True, confidence=0.0, suggested_agent="code")
