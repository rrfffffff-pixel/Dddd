"""In-memory semantic search using TF-IDF embeddings."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class SearchResult:
    path: str
    score: float
    snippet: str
    line_start: int
    line_end: int


@dataclass
class _DocMeta:
    path: str
    content_hash: str
    chunks: list[str] = field(default_factory=list)
    chunk_starts: list[int] = field(default_factory=list)
    chunk_ends: list[int] = field(default_factory=list)


def _tokenize(text: str) -> list[str]:
    # Split camelCase and snake_case, then lowercase
    raw = re.findall(r"[a-zA-Z_]\w*", text)
    tokens: list[str] = []
    for word in raw:
        # Insert space before uppercase letters that follow lowercase
        split = re.sub(r"([a-z])([A-Z])", r"\1 \2", word)
        tokens.extend(t.lower() for t in split.split())
    return tokens


def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SemanticIndex:
    """TF-IDF based in-memory vector index for code search."""

    CHUNK_LINES = 40
    OVERLAP_LINES = 10
    SNIPPET_CONTEXT = 5

    def __init__(self) -> None:
        self._docs: dict[str, _DocMeta] = {}
        self._hash_cache: dict[str, dict[str, float]] = {}

        # sklearn state
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._doc_paths: list[str] = []
        self._dirty = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_document(self, path: str, content: str) -> None:
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if path in self._docs and self._docs[path].content_hash == content_hash:
            return  # unchanged

        lines = content.splitlines()
        chunks, starts, ends = self._chunk_lines(lines)

        self._docs[path] = _DocMeta(
            path=path,
            content_hash=content_hash,
            chunks=chunks,
            chunk_starts=starts,
            chunk_ends=ends,
        )
        self._dirty = True

    def remove_document(self, path: str) -> None:
        if path in self._docs:
            del self._docs[path]
            self._dirty = True

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not self._docs:
            return []

        self._rebuild_if_needed()
        query_lower = query.lower()

        if HAS_SKLEARN and self._vectorizer is not None:
            return self._search_sklearn(query_lower, top_k)
        return self._search_bow(query_lower, top_k)

    def get_stats(self) -> dict:
        total_chunks = sum(len(d.chunks) for d in self._docs.values())
        return {
            "documents": len(self._docs),
            "total_chunks": total_chunks,
            "total_lines": sum(len(d.chunks) * self.CHUNK_LINES for d in self._docs.values()),
            "engine": "sklearn" if HAS_SKLEARN else "bow",
        }

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk_lines(self, lines: list[str]) -> tuple[list[str], list[int], list[int]]:
        if not lines:
            return [], [], []
        chunks: list[str] = []
        starts: list[int] = []
        ends: list[int] = []
        step = self.CHUNK_LINES - self.OVERLAP_LINES
        for i in range(0, len(lines), step):
            end = min(i + self.CHUNK_LINES, len(lines))
            chunk_text = "\n".join(lines[i:end])
            if chunk_text.strip():
                chunks.append(chunk_text)
                starts.append(i + 1)  # 1-indexed
                ends.append(end)
            if end >= len(lines):
                break
        return chunks, starts, ends

    # ------------------------------------------------------------------
    # Rebuild index
    # ------------------------------------------------------------------

    def _rebuild_if_needed(self) -> None:
        if not self._dirty:
            return
        self._dirty = False

        if HAS_SKLEARN:
            self._rebuild_sklearn()
        # BOW: no pre-build needed; we compute per-search.

    def _rebuild_sklearn(self) -> None:
        all_chunks: list[str] = []
        self._doc_paths = []
        for path, meta in self._docs.items():
            for chunk in meta.chunks:
                all_chunks.append(chunk)
                self._doc_paths.append(path)

        if not all_chunks:
            self._vectorizer = None
            self._tfidf_matrix = None
            return

        self._vectorizer = TfidfVectorizer(
            token_pattern=r"[a-zA-Z_]\w*",
            max_features=5000,
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(all_chunks)

    # ------------------------------------------------------------------
    # Search backends
    # ------------------------------------------------------------------

    def _search_sklearn(self, query: str, top_k: int) -> list[SearchResult]:
        if self._vectorizer is None or self._tfidf_matrix is None:
            return []

        q_vec = self._vectorizer.transform([query])
        scores = sk_cosine(q_vec, self._tfidf_matrix).flatten()

        # Map chunk index -> path and line info
        chunk_idx = 0
        results: list[SearchResult] = []
        for path, meta in self._docs.items():
            for j in range(len(meta.chunks)):
                score = float(scores[chunk_idx]) if chunk_idx < len(scores) else 0.0
                if score > 0:
                    snippet = self._make_snippet(meta.chunks[j])
                    results.append(SearchResult(
                        path=path,
                        score=round(score, 4),
                        snippet=snippet,
                        line_start=meta.chunk_starts[j],
                        line_end=meta.chunk_ends[j],
                    ))
                chunk_idx += 1

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _search_bow(self, query: str, top_k: int) -> list[SearchResult]:
        query_tf = self._compute_tf(query)
        idf = self._compute_idf()

        results: list[SearchResult] = []
        for path, meta in self._docs.items():
            for j, chunk in enumerate(meta.chunks):
                chunk_tf = self._compute_tf(chunk)
                # Apply IDF weighting
                weighted = {t: tf * idf.get(t, 1.0) for t, tf in chunk_tf.items()}
                q_weighted = {t: tf * idf.get(t, 1.0) for t, tf in query_tf.items()}
                score = _cosine_sim(q_weighted, weighted)
                if score > 0:
                    snippet = self._make_snippet(chunk)
                    results.append(SearchResult(
                        path=path,
                        score=round(score, 4),
                        snippet=snippet,
                        line_start=meta.chunk_starts[j],
                        line_end=meta.chunk_ends[j],
                    ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_tf(self, text: str) -> dict[str, float]:
        tokens = _tokenize(text)
        if not tokens:
            return {}
        counts = Counter(tokens)
        max_count = max(counts.values())
        return {t: c / max_count for t, c in counts.items()}

    def _compute_idf(self) -> dict[str, float]:
        n = len(self._docs)
        if n == 0:
            return {}
        df: Counter[str] = Counter()
        for meta in self._docs.values():
            unique = set()
            for chunk in meta.chunks:
                unique.update(_tokenize(chunk))
            df.update(unique)
        return {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}

    @staticmethod
    def _make_snippet(chunk: str, max_lines: int = 6) -> str:
        lines = chunk.splitlines()
        if len(lines) <= max_lines:
            return chunk
        half = max_lines // 2
        return "\n".join(lines[:half] + ["..."] + lines[-half:])
