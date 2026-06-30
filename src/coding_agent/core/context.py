"""Shared context management for agent collaboration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class FileChange:
    path: str
    action: str  # created, modified, deleted
    diff: str = ""
    content: str = ""


@dataclass
class CodeSymbol:
    name: str
    kind: str  # function, class, variable, import
    file: str
    line: int = 0
    signature: str = ""


@dataclass
class SharedContext:
    """Context shared between agents during a task."""

    task_description: str = ""
    project_root: str = "."
    files_read: list[str] = field(default_factory=list)
    files_written: list[str] = field(default_factory=list)
    changes: list[FileChange] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    test_results: str = ""
    decisions: list[str] = field(default_factory=list)
    symbols: list[CodeSymbol] = field(default_factory=list)
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    project_type: str = ""
    code_style: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def add_file_read(self, path: str) -> None:
        if path not in self.files_read:
            self.files_read.append(path)

    def add_file_written(self, path: str, diff: str = "", content: str = "") -> None:
        self.files_written.append(path)
        self.changes.append(FileChange(path=path, action="modified", diff=diff, content=content))

    def add_error(self, error: str) -> None:
        self.errors.append(error)

    def add_symbol(self, symbol: CodeSymbol) -> None:
        self.symbols.append(symbol)

    def add_decision(self, decision: str) -> None:
        self.decisions.append(decision)

    def get_symbols_by_file(self, file_path: str) -> list[CodeSymbol]:
        return [s for s in self.symbols if s.file == file_path]

    def get_symbols_by_kind(self, kind: str) -> list[CodeSymbol]:
        return [s for s in self.symbols if s.kind == kind]

    def find_symbol(self, name: str) -> CodeSymbol | None:
        for s in self.symbols:
            if s.name == name:
                return s
        return None

    def file_hash(self, path: str, content: str) -> str:
        return hashlib.md5(f"{path}:{content}".encode()).hexdigest()

    def summary(self) -> str:
        lines = []
        if self.task_description:
            lines.append(f"Task: {self.task_description}")
        if self.project_type:
            lines.append(f"Project type: {self.project_type}")
        if self.files_read:
            lines.append(f"Files read: {', '.join(self.files_read[-10:])}")
        if self.files_written:
            lines.append(f"Files modified: {', '.join(self.files_written)}")
        if self.symbols:
            funcs = [s.name for s in self.symbols if s.kind == "function"][:5]
            classes = [s.name for s in self.symbols if s.kind == "class"][:5]
            if funcs:
                lines.append(f"Functions: {', '.join(funcs)}")
            if classes:
                lines.append(f"Classes: {', '.join(classes)}")
        if self.errors:
            lines.append(f"Errors: {len(self.errors)} - {self.errors[-1]}")
        if self.test_results:
            lines.append(f"Tests: {self.test_results[:300]}")
        if self.decisions:
            lines.append(f"Decisions: {'; '.join(self.decisions[-3:])}")
        return "\n".join(lines)
