"""Shared context management for agent collaboration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FileChange:
    path: str
    action: str  # created, modified, deleted
    diff: str = ""
    content: str = ""


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
    metadata: dict = field(default_factory=dict)

    def add_file_read(self, path: str) -> None:
        if path not in self.files_read:
            self.files_read.append(path)

    def add_file_written(self, path: str, diff: str = "", content: str = "") -> None:
        self.files_written.append(path)
        self.changes.append(FileChange(path=path, action="modified", diff=diff, content=content))

    def add_error(self, error: str) -> None:
        self.errors.append(error)

    def summary(self) -> str:
        lines = [f"Task: {self.task_description}"]
        if self.files_read:
            lines.append(f"Files read: {', '.join(self.files_read)}")
        if self.files_written:
            lines.append(f"Files modified: {', '.join(self.files_written)}")
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
        if self.test_results:
            lines.append(f"Test results: {self.test_results[:200]}")
        return "\n".join(lines)
