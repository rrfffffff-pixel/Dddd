"""Pre-processing pipeline to reduce LLM calls - lexical analysis, token optimization."""

from __future__ import annotations

import ast
import hashlib
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


class LRUCache:
    """Simple LRU cache with max size eviction."""

    def __init__(self, max_size: int = 500) -> None:
        self._cache: dict[str, str] = {}
        self._max_size = max_size
        self._order: list[str] = []

    def get(self, key: str) -> str | None:
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: str) -> None:
        if key in self._cache:
            self._order.remove(key)
        elif len(self._cache) >= self._max_size:
            oldest = self._order.pop(0)
            del self._cache[oldest]
        self._cache[key] = value
        self._order.append(key)

    def clear(self) -> None:
        self._cache.clear()
        self._order.clear()


class CacheLayer:
    """Cache LLM responses and tool results to avoid redundant calls."""

    def __init__(self) -> None:
        self._llm_cache: LRUCache = LRUCache(max_size=200)
        self._tool_cache: LRUCache = LRUCache(max_size=500)
        self._file_hashes: dict[str, str] = {}

    def file_changed(self, path: str, content: str) -> bool:
        new_hash = hashlib.md5(content.encode()).hexdigest()
        old_hash = self._file_hashes.get(path)
        self._file_hashes[path] = new_hash
        return old_hash is not None and old_hash != new_hash

    def get_llm_cache(self, prompt_hash: str) -> str | None:
        return self._llm_cache.get(prompt_hash)

    def set_llm_cache(self, prompt_hash: str, response: str) -> None:
        self._llm_cache.set(prompt_hash, response)

    def get_tool_cache(self, key: str) -> str | None:
        return self._tool_cache.get(key)

    def set_tool_cache(self, key: str, result: str) -> None:
        self._tool_cache.set(key, result)

    def clear(self) -> None:
        self._llm_cache.clear()
        self._tool_cache.clear()
        self._file_hashes.clear()


class TaskClassifier:
    """Classify tasks without LLM to skip unnecessary LLM calls."""

    STATIC_PATTERNS = {
        "list_files": [
            (r"^(?:list|show)\s+(?:all\s+)?(?:the\s+)?files?", "code"),
            (r"what\s+files?\s+(?:are|exist|in)", "code"),
            (r"list\s+directory", "code"),
        ],
        "search": [
            (r"^(?:find|search)\s+(?:all\s+)?", "code"),
            (r"^(?:grep|search)\s+for\s+", "code"),
        ],
        "test": [
            (r"^run\s+(?:the\s+)?tests?", "test"),
            (r"^test\s+(?:the\s+)?code", "test"),
        ],
        "install": [
            (r"^install\s+", "shell"),
            (r"^(?:pip|npm|cargo|go)\s+install", "shell"),
        ],
        "review": [
            (r"^review\s+", "review"),
            (r"^check\s+(?:the\s+)?code\s+for\s+", "review"),
        ],
    }

    STATIC_ANSWERS = {
        "hello": "Hello! I'm Coding Agent. How can I help you with your code today?",
        "help": "Available commands via CLI: `coding-agent run <task>`, `coding-agent code <task>`, `coding-agent test`, `coding-agent info`.\n\nOr describe what you'd like me to do with your codebase.",
        "hi": "Hi there! I'm ready to help with coding tasks.",
        "thanks": "You're welcome! Let me know if you need anything else.",
    }

    def classify(self, task: str) -> TaskAnalysis:
        lower = task.lower().strip()

        # Check for static answer patterns (no LLM needed)
        for key, answer in self.STATIC_ANSWERS.items():
            if lower == key or lower.startswith(key):
                return TaskAnalysis(
                    needs_llm=False,
                    confidence=0.95,
                    suggested_agent="code",
                    static_answer=answer,
                )

        # Check for simple file listing (can be done without LLM)
        if re.match(r"^(?:list|show)\s+(?:all\s+)?files?\s*$", lower):
            return TaskAnalysis(
                needs_llm=False,
                confidence=0.9,
                suggested_agent="code",
                static_answer="Please use the list_files tool to browse the project structure.",
            )

        # Route to appropriate agent
        for category, patterns in self.STATIC_PATTERNS.items():
            for pattern, agent in patterns:
                if re.search(pattern, lower):
                    return TaskAnalysis(
                        needs_llm=True,
                        confidence=0.8,
                        suggested_agent=agent,
                    )

        return TaskAnalysis(needs_llm=True, confidence=0.0, suggested_agent="code")
