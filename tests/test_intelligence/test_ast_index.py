"""Tests for AST code intelligence index."""

import tempfile
from pathlib import Path

from coding_agent.intelligence.ast_index import CodeIndex, Symbol


SAMPLE_CODE = '''\
import os
from pathlib import Path

GLOBAL_VAR = 42

def greet(name: str) -> str:
    return f"Hello, {name}"

def add(a: int, b: int) -> int:
    return a + b

class Dog:
    def __init__(self, name: str):
        self.name = name

    def bark(self) -> str:
        return "Woof!"
'''


def test_index_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sample.py"
        path.write_text(SAMPLE_CODE)

        index = CodeIndex()
        symbols = index.index_file(str(path))

        assert len(symbols) > 0
        assert all(isinstance(s, Symbol) for s in symbols)

        names = [s.name for s in symbols]
        assert "greet" in names
        assert "add" in names
        assert "Dog" in names
        assert "GLOBAL_VAR" in names


def test_symbol_kinds():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sample.py"
        path.write_text(SAMPLE_CODE)

        index = CodeIndex()
        symbols = index.index_file(str(path))

        functions = [s for s in symbols if s.kind == "function"]
        classes = [s for s in symbols if s.kind == "class"]
        imports = [s for s in symbols if s.kind == "import"]
        variables = [s for s in symbols if s.kind == "variable"]

        assert len(functions) >= 2
        assert len(classes) >= 1
        assert len(imports) >= 2
        assert len(variables) >= 1


def test_line_numbers():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sample.py"
        path.write_text(SAMPLE_CODE)

        index = CodeIndex()
        symbols = index.index_file(str(path))

        greet = next(s for s in symbols if s.name == "greet")
        assert greet.line > 0
        assert greet.end_line >= greet.line


def test_search_symbols():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sample.py"
        path.write_text(SAMPLE_CODE)

        index = CodeIndex()
        index.index_file(str(path))

        results = index.search_symbols("greet")
        assert any(s.name == "greet" for s in results)

        results = index.search_symbols("dog")
        assert any(s.name == "Dog" for s in results)


def test_get_dependencies():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sample.py"
        path.write_text(SAMPLE_CODE)

        index = CodeIndex()
        index.index_file(str(path))

        deps = index.get_dependencies(str(path))
        assert "os" in deps
        assert "pathlib" in deps


def test_get_file_summary():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sample.py"
        path.write_text(SAMPLE_CODE)

        index = CodeIndex()
        index.index_file(str(path))

        summary = index.get_file_summary(str(path))
        assert "function" in summary
        assert "class" in summary
        assert "greet" in summary
        assert "Dog" in summary


def test_index_project():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "a.py").write_text("def a(): pass")
        (Path(tmpdir) / "b.py").write_text("class B: pass")
        sub = Path(tmpdir) / "sub"
        sub.mkdir()
        (sub / "c.py").write_text("x = 1")

        index = CodeIndex()
        result = index.index_project(tmpdir)

        assert len(result) == 3
        all_names = []
        for symbols in result.values():
            all_names.extend(s.name for s in symbols)
        assert "a" in all_names
        assert "B" in all_names
        assert "x" in all_names


def test_index_non_python():
    index = CodeIndex()
    symbols = index.index_file("/nonexistent.py")
    assert symbols == []


def test_signature():
    with tempfile.TemporaryDirectory() as tmpdir:
        code = "def my_func(a, b): pass\nclass MyClass(Base):\n    pass"
        path = Path(tmpdir) / "sig.py"
        path.write_text(code)

        index = CodeIndex()
        symbols = index.index_file(str(path))

        func = next(s for s in symbols if s.name == "my_func")
        assert "def" in func.signature
        assert "my_func" in func.signature

        cls = next(s for s in symbols if s.name == "MyClass")
        assert "class" in cls.signature
        assert "MyClass" in cls.signature
