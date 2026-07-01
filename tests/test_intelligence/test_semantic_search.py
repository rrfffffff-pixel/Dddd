"""Tests for src.coding_agent.intelligence.semantic_search."""

import pytest
from coding_agent.intelligence.semantic_search import (
    SemanticIndex,
    SearchResult,
    _tokenize,
    _cosine_sim,
    HAS_SKLEARN,
)


# ── Helpers ──────────────────────────────────────────────────────────

SAMPLE_PYTHON = '''\
def hello_world():
    """Print hello world."""
    print("hello world")

class Greeter:
    def greet(self, name: str) -> str:
        return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b
'''

SAMPLE_JS = '''\
function fetchData(url) {
    return fetch(url).then(res => res.json());
}

class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }
    async get(path) {
        return fetchData(this.baseUrl + path);
    }
}
'''


# ── Tokenizer / cosine helpers ──────────────────────────────────────

class TestHelpers:
    def test_tokenize_basic(self):
        assert _tokenize("hello world") == ["hello", "world"]

    def test_tokenize_camel_case(self):
        tokens = _tokenize("fetchData")
        assert "fetch" in tokens
        assert "data" in tokens

    def test_tokenize_underscore(self):
        tokens = _tokenize("hello_world")
        assert "hello_world" in tokens

    def test_cosine_sim_identical(self):
        v = {"a": 1.0, "b": 2.0}
        assert _cosine_sim(v, v) == pytest.approx(1.0)

    def test_cosine_sim_disjoint(self):
        assert _cosine_sim({"a": 1.0}, {"b": 1.0}) == 0.0

    def test_cosine_sim_partial(self):
        a = {"x": 1.0, "y": 2.0}
        b = {"y": 2.0, "z": 3.0}
        score = _cosine_sim(a, b)
        assert 0.0 < score < 1.0


# ── SemanticIndex: basic lifecycle ──────────────────────────────────

class TestSemanticIndex:
    def test_empty_search(self):
        idx = SemanticIndex()
        assert idx.search("anything") == []

    def test_add_and_search(self):
        idx = SemanticIndex()
        idx.add_document("hello.py", SAMPLE_PYTHON)
        results = idx.search("hello world")
        assert len(results) > 0
        assert results[0].path == "hello.py"
        assert isinstance(results[0], SearchResult)

    def test_search_returns_snippet(self):
        idx = SemanticIndex()
        idx.add_document("greeter.py", SAMPLE_PYTHON)
        results = idx.search("hello world")
        assert len(results) > 0
        assert results[0].path == "greeter.py"
        assert len(results[0].snippet) > 0
        assert results[0].line_start >= 1

    def test_search_respects_top_k(self):
        idx = SemanticIndex()
        idx.add_document("a.py", SAMPLE_PYTHON)
        idx.add_document("b.js", SAMPLE_JS)
        results = idx.search("function", top_k=1)
        assert len(results) <= 1

    def test_remove_document(self):
        idx = SemanticIndex()
        idx.add_document("tmp.py", "x = 1")
        idx.remove_document("tmp.py")
        assert idx.search("x") == []

    def test_remove_nonexistent(self):
        idx = SemanticIndex()
        idx.remove_document("nope.py")  # should not raise

    def test_get_stats(self):
        idx = SemanticIndex()
        idx.add_document("a.py", SAMPLE_PYTHON)
        stats = idx.get_stats()
        assert stats["documents"] == 1
        assert stats["total_chunks"] >= 1
        assert stats["engine"] in ("sklearn", "bow")


# ── SemanticIndex: dedup / caching ─────────────────────────────────

class TestCaching:
    def test_add_same_content_twice_no_duplicate(self):
        idx = SemanticIndex()
        idx.add_document("a.py", SAMPLE_PYTHON)
        idx.add_document("a.py", SAMPLE_PYTHON)
        stats = idx.get_stats()
        assert stats["documents"] == 1

    def test_add_updated_content(self):
        idx = SemanticIndex()
        idx.add_document("a.py", "version 1")
        idx.add_document("a.py", "version 2 completely different")
        results = idx.search("completely different")
        assert len(results) > 0
        assert results[0].path == "a.py"


# ── SemanticIndex: multi-file ranking ───────────────────────────────

class TestRanking:
    def test_specific_file_ranks_higher(self):
        idx = SemanticIndex()
        idx.add_document("auth.py", "def authenticate(user, password): ...")
        idx.add_document("math.py", "def add(a, b): return a + b")
        results = idx.search("authenticate user password")
        assert len(results) > 0
        assert results[0].path == "auth.py"

    def test_javascript_file_found(self):
        idx = SemanticIndex()
        idx.add_document("client.js", SAMPLE_JS)
        results = idx.search("fetch data api")
        assert len(results) > 0
        assert results[0].path == "client.js"


# ── Line numbers ────────────────────────────────────────────────────

class TestLineNumbers:
    def test_line_start_end_reasonable(self):
        idx = SemanticIndex()
        idx.add_document("demo.py", SAMPLE_PYTHON)
        results = idx.search("add function")
        for r in results:
            assert r.line_start >= 1
            assert r.line_end >= r.line_start


# ── Sklearn availability flag ───────────────────────────────────────

class TestEngine:
    def test_engine_flag_matches_available(self):
        idx = SemanticIndex()
        stats = idx.get_stats()
        if HAS_SKLEARN:
            assert stats["engine"] == "sklearn"
        else:
            assert stats["engine"] == "bow"
