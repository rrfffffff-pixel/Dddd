"""RepoMap: codebase indexing and ranking using AST analysis.

Adapted from Aider's RepoMap (PageRank-based file ranking).
Replaces tree-sitter with Python's built-in `ast` module for portability.
"""

from __future__ import annotations

import ast
import os
import re
from collections import defaultdict

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


Tag = tuple[str, str, int, str, str]  # (rel_fname, abs_fname, line, name, kind)


class RepoMap:
    """Builds a ranked map of the codebase for LLM context."""

    def __init__(
        self,
        root: str = ".",
        map_tokens: int = 1024,
        refresh: str = "auto",
        verbose: bool = False,
    ):
        self.root = root
        self.max_map_tokens = map_tokens
        self.refresh = refresh
        self.verbose = verbose
        self._tag_cache: dict[str, dict] = {}
        self._map_cache: dict[str, str] = {}

    def get_repo_map(
        self,
        chat_files: list[str] | None = None,
        other_files: list[str] | None = None,
        mentioned_fnames: set[str] | None = None,
        mentioned_idents: set[str] | None = None,
    ) -> str | None:
        if self.max_map_tokens <= 0:
            return None
        if not other_files and not chat_files:
            return None
        chat_files = chat_files or []
        other_files = other_files or []
        mentioned_fnames = mentioned_fnames or set()
        mentioned_idents = mentioned_idents or set()

        result = self._get_ranked_tags_map(
            chat_files, other_files, self.max_map_tokens,
            mentioned_fnames, mentioned_idents,
        )
        if not result:
            return None
        return result

    def _get_ranked_tags_map(self, chat_files, other_files, max_map_tokens,
                              mentioned_fnames, mentioned_idents):
        fnames = sorted(set(chat_files) | set(other_files))
        tags = self._get_tags_for_files(fnames)

        idents = defaultdict(list)
        defines = defaultdict(set)
        references = defaultdict(list)
        definitions = defaultdict(set)
        file_tags: dict[str, list[Tag]] = defaultdict(list)

        for tag in tags:
            rel_fname, abs_fname, line, name, kind = tag
            file_tags[abs_fname].append(tag)
            tags_key = (rel_fname, abs_fname)

            if kind == "def":
                defines[tags_key].add(name)
                definitions[name].add(tags_key)
            elif kind == "ref":
                references[name].append(tags_key)
            idents[name].append(tags_key)

        if not defines and not references:
            return self._build_flat_list(fnames, max_map_tokens)

        if not HAS_NETWORKX:
            return self._build_flat_list(fnames, max_map_tokens)

        G = nx.MultiDiGraph()
        all_tags_keys = set(defines.keys()) | {k for v in references.values() for k in v}

        for ident, refs in references.items():
            defs = definitions.get(ident, set())
            for ref in refs:
                for d in defs:
                    if ref != d:
                        G.add_edge(ref, d, weight=1.0, ident=ident)

        for tag_key in all_tags_keys:
            G.add_node(tag_key)

        if not G.nodes:
            return self._build_flat_list(fnames, max_map_tokens)

        personalize = {}
        num_nodes = len(G.nodes)
        default_val = 100.0 / num_nodes if num_nodes > 0 else 1.0

        for tag_key in G.nodes:
            personalize[tag_key] = default_val
            rel_fname = tag_key[0]
            for f in chat_files:
                if os.path.relpath(f, self.root) == rel_fname:
                    personalize[tag_key] = default_val * 50
            for mf in mentioned_fnames:
                if os.path.relpath(mf, self.root) == rel_fname:
                    personalize[tag_key] = default_val * 100

        for ident in mentioned_idents:
            for tag_key in idents.get(ident, []):
                if tag_key in personalize:
                    personalize[tag_key] = default_val * 100

        try:
            ranked = nx.pagerank(G, weight="weight", personalization=personalize)
        except ZeroDivisionError:
            ranked = nx.pagerank(G, weight="weight")

        files_ranked: dict[str, float] = defaultdict(float)
        for (rel_fname, abs_fname), score in ranked.items():
            files_ranked[abs_fname] += score

        ranked_files = sorted(files_ranked, key=files_ranked.get, reverse=True)

        return self._render_map(ranked_files, file_tags, chat_files, max_map_tokens)

    def _render_map(self, ranked_files, file_tags, chat_files, max_map_tokens):
        chat_abs = {os.path.abspath(f) for f in chat_files}
        lines = []
        used_tokens = 0

        chat_fnames_output = set()

        for abs_fname in ranked_files:
            if abs_fname in chat_abs:
                rel = os.path.relpath(abs_fname, self.root)
                chat_fnames_output.add(rel)
                continue

            tags = file_tags.get(abs_fname, [])
            if not tags:
                continue

            rel_fname = tags[0][0]
            defs = sorted(set(t[3] if isinstance(t, tuple) else t.name for t in tags if (t[4] if isinstance(t, tuple) else t.kind) == "def"))
            if not defs:
                continue

            entry = f"{rel_fname}:\n"
            for d in defs[:30]:
                entry += f"    def {d[:80]}\n"

            estimated = len(entry) // 3
            if used_tokens + estimated > max_map_tokens:
                break

            lines.append(entry)
            used_tokens += estimated

        chat_headers = []
        for rel in sorted(chat_fnames_output):
            tags = file_tags.get(os.path.abspath(os.path.join(self.root, rel)), [])
            defs = sorted(set(t[3] if isinstance(t, tuple) else t.name for t in tags if (t[4] if isinstance(t, tuple) else t.kind) == "def"))
            if defs:
                entry = f"{rel}:\n"
                for d in defs[:30]:
                    entry += f"    def {d[:80]}\n"
                chat_headers.append(entry)

        final = ""
        if chat_headers:
            final += "".join(chat_headers)
        final += "".join(lines)
        return final

    def _build_flat_list(self, fnames, max_map_tokens):
        lines = []
        used = 0
        for fname in fnames[:50]:
            rel = os.path.relpath(fname, self.root)
            entry = f"{rel}\n"
            est = len(entry) // 3
            if used + est > max_map_tokens:
                break
            lines.append(entry)
            used += est
        return "".join(lines)

    def _get_tags_for_files(self, fnames: list[str]) -> list[Tag]:
        tags = []
        for fname in fnames:
            if not os.path.isfile(fname):
                continue
            cached = self._get_cached(fname)
            if cached:
                tags.extend(cached)
            else:
                extracted = list(self._get_tags_for_file(fname))
                self._set_cached(fname, extracted)
                tags.extend(extracted)
        return tags

    def _get_cached(self, fname):
        if fname in self._tag_cache:
            entry = self._tag_cache[fname]
            if entry.get("mtime") == self._get_mtime(fname):
                return entry["data"]
        return None

    def _set_cached(self, fname, data):
        self._tag_cache[fname] = {
            "mtime": self._get_mtime(fname),
            "data": data,
        }

    def _get_mtime(self, fname):
        try:
            return os.path.getmtime(fname)
        except OSError:
            return 0

    def _get_tags_for_file(self, fname: str) -> list[Tag]:
        rel_fname = os.path.relpath(fname, self.root)
        ext = os.path.splitext(fname)[1].lower()

        try:
            with open(fname, encoding="utf-8", errors="replace") as f:
                code = f.read()
        except OSError:
            return []

        if ext == ".py":
            yield from self._ast_tags(rel_fname, fname, code)
        else:
            yield from self._regex_tags(rel_fname, fname, code, ext)

    def _ast_tags(self, rel_fname, abs_fname, code):
        try:
            tree = ast.parse(code, filename=abs_fname)
        except SyntaxError:
            return

        finder = _TagFinder(rel_fname, abs_fname)
        finder.visit(tree)
        yield from finder.tags

    def _regex_tags(self, rel_fname, abs_fname, code, ext):
        patterns = _get_regex_patterns(ext)
        seen = set()

        for pattern, kind in patterns:
            for m in pattern.finditer(code):
                name = m.group(1) or m.group(0)
                if not name or name in seen:
                    continue
                seen.add(name)
                line = code[:m.start()].count("\n")
                yield (rel_fname, abs_fname, line, name.strip(), kind)


class _TagFinder(ast.NodeVisitor):
    def __init__(self, rel_fname, abs_fname):
        self.rel_fname = rel_fname
        self.abs_fname = abs_fname
        self.tags: list[Tag] = []
        self._refs: set[str] = set()
        self._defs: set[str] = set()

    def visit_FunctionDef(self, node):
        self.tags.append((self.rel_fname, self.abs_fname, node.lineno or 0, node.name, "def"))
        self._defs.add(node.name)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.tags.append((self.rel_fname, self.abs_fname, node.lineno or 0, node.name, "def"))
        self._defs.add(node.name)
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.tags.append((self.rel_fname, self.abs_fname, node.lineno or 0, target.id, "def"))
                self._defs.add(target.id)
        self.generic_visit(node)

    def visit_Call(self, node):
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name and name not in self._refs:
            self._refs.add(name)
            self.tags.append((self.rel_fname, self.abs_fname, node.lineno or 0, name, "ref"))
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id not in self._defs:
            self.tags.append((self.rel_fname, self.abs_fname, node.lineno or 0, node.id, "ref"))
            self._refs.add(node.id)


def _get_regex_patterns(ext):
    patterns = []
    if ext in (".js", ".jsx", ".ts", ".tsx"):
        patterns = [
            (re.compile(r'(?:function|const|let|var)\s+(\w+)\s*[=(]'), "def"),
            (re.compile(r'(?:class)\s+(\w+)'), "def"),
            (re.compile(r'(\w+)\.(\w+)\s*\('), "ref"),
        ]
    elif ext in (".go",):
        patterns = [
            (re.compile(r'func\s+(\w+)'), "def"),
            (re.compile(r'type\s+(\w+)\s'), "def"),
        ]
    elif ext in (".rs",):
        patterns = [
            (re.compile(r'fn\s+(\w+)'), "def"),
            (re.compile(r'struct\s+(\w+)'), "def"),
            (re.compile(r'enum\s+(\w+)'), "def"),
            (re.compile(r'trait\s+(\w+)'), "def"),
        ]
    elif ext in (".java", ".kt"):
        patterns = [
            (re.compile(r'(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*(?:\w+\s+)?(\w+)\s*\('), "def"),
            (re.compile(r'class\s+(\w+)'), "def"),
            (re.compile(r'interface\s+(\w+)'), "def"),
        ]
    elif ext in (".md", ".txt", ".rst"):
        patterns = []
    else:
        patterns = [
            (re.compile(r'def\s+(\w+)'), "def"),
            (re.compile(r'class\s+(\w+)'), "def"),
        ]
    return patterns
