"""AST-based code intelligence for Python files."""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Symbol:
    name: str
    kind: str  # function | class | import | variable
    file: str
    line: int
    end_line: int
    signature: str = ""


def _extract_imports(tree: ast.Module, source: str) -> list[str]:
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def _build_function_signature(node: ast.FunctionDef) -> str:
    args = []
    for arg in node.args.args:
        args.append(arg.arg)
    params = ", ".join(args)
    return f"def {node.name}({params})"


def _build_class_signature(node: ast.ClassDef) -> str:
    bases = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
    if bases:
        return f"class {node.name}({', '.join(bases)})"
    return f"class {node.name}"


def _extract_symbols(tree: ast.Module, file_path: str, source: str) -> list[Symbol]:
    symbols: list[Symbol] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            symbols.append(Symbol(
                name=node.name,
                kind="function",
                file=file_path,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                signature=_build_function_signature(node),
            ))
        elif isinstance(node, ast.AsyncFunctionDef):
            symbols.append(Symbol(
                name=node.name,
                kind="function",
                file=file_path,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                signature=f"async {_build_function_signature(node)}",
            ))
        elif isinstance(node, ast.ClassDef):
            symbols.append(Symbol(
                name=node.name,
                kind="class",
                file=file_path,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                signature=_build_class_signature(node),
            ))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                symbols.append(Symbol(
                    name=alias.name,
                    kind="import",
                    file=file_path,
                    line=node.lineno,
                    end_line=node.lineno,
                    signature=f"import {alias.name}",
                ))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                symbols.append(Symbol(
                    name=f"{module}.{alias.name}",
                    kind="import",
                    file=file_path,
                    line=node.lineno,
                    end_line=node.lineno,
                    signature=f"from {module} import {alias.name}",
                ))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.append(Symbol(
                        name=target.id,
                        kind="variable",
                        file=file_path,
                        line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        signature=f"{target.id} = ...",
                    ))

    return symbols


class CodeIndex:
    def __init__(self) -> None:
        self._symbols: dict[str, list[Symbol]] = {}
        self._dependencies: dict[str, list[str]] = {}

    def index_file(self, path: str) -> list[Symbol]:
        p = Path(path)
        if not p.exists() or p.suffix != ".py":
            return []

        try:
            source = p.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            return []

        symbols = _extract_symbols(tree, str(p), source)
        self._symbols[str(p)] = symbols
        self._dependencies[str(p)] = _extract_imports(tree, source)
        return symbols

    def index_project(self, root: str) -> dict[str, list[Symbol]]:
        skip = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mimocode", "dist", "build"}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in skip]
            for fname in filenames:
                if fname.endswith(".py"):
                    self.index_file(os.path.join(dirpath, fname))
        return self._symbols

    def search_symbols(self, query: str) -> list[Symbol]:
        results: list[Symbol] = []
        lower = query.lower()
        for symbols in self._symbols.values():
            for sym in symbols:
                if lower in sym.name.lower():
                    results.append(sym)
        return results

    def get_dependencies(self, file: str) -> list[str]:
        return self._dependencies.get(file, [])

    def get_file_summary(self, path: str) -> str:
        symbols = self._symbols.get(path, [])
        if not symbols:
            return f"No symbols indexed for {path}"

        by_kind: dict[str, list[Symbol]] = {}
        for sym in symbols:
            by_kind.setdefault(sym.kind, []).append(sym)

        parts = [f"File: {path}"]
        for kind, syms in sorted(by_kind.items()):
            names = ", ".join(s.name for s in syms)
            parts.append(f"  {kind}: {names}")
        return "\n".join(parts)
