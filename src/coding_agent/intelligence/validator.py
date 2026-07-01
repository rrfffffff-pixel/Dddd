"""Shadow validation - post-edit verification of code changes.

Like Cursor's Shadow Workspace: validates edits before finalizing.
"""

from __future__ import annotations

import ast
from pathlib import Path


class ShadowValidator:
    """Validates code changes after editing."""

    def validate_python(self, file_path: str, content: str) -> list[str]:
        issues = []
        try:
            ast.parse(content)
        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
        return issues

    def validate_indentation(self, content: str, language: str = "python") -> list[str]:
        issues = []
        lines = content.splitlines()
        for i, line in enumerate(lines, 1):
            stripped = line.rstrip()
            if stripped and line != stripped:
                issues.append(f"Line {i}: trailing whitespace")
        if language == "python":
            for i, line in enumerate(lines, 1):
                if line.startswith("\t"):
                    issues.append(f"Line {i}: tabs used instead of spaces")
                    break
        return issues

    def validate(self, file_path: str, content: str) -> dict[str, list[str]]:
        results = {}
        ext = Path(file_path).suffix
        if ext == ".py":
            syntax = self.validate_python(file_path, content)
            if syntax:
                results["syntax"] = syntax
            indent = self.validate_indentation(content, "python")
            if indent:
                results["formatting"] = indent[:5]
        return results


def check_edit(path: str, old_content: str, new_content: str) -> list[str]:
    validator = ShadowValidator()
    return validator.validate(path, new_content)
