"""Hybrid code search: BM25 over code-aware tokens + dense vectors, fused, optionally re-ranked.

    query --+-- BM25 (identifiers split: getUserById -> get user by id) --+
            +-- dense (bge-small, cosine) ----------------------------------+-- fusion (RRF or weighted)
                                                                                 -> cross-encoder rerank
                                                                                 -> exact-symbol boost
Which fusion and whether to rerank were chosen by evals/retrieval_eval.py, not by taste (README has the table):
weighted fusion is the default, and the generic MS MARCO reranker is opt-in because it made code search worse.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from . import embeddings, store
from .chunking import Chunk
from .config import get_settings

_STOP = {"the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "for", "on", "it", "this", "that", "how",
         "what", "where", "which", "does", "do", "with", "be", "by", "as", "at", "from", "self", "return", "def",
         "class", "import", "if", "else", "none", "true", "false", "function", "const", "let", "var", "new",
         "when", "why", "who", "can", "i", "we", "you", "code", "file", "used", "use"}


def tokenize(text: str) -> list[str]:
    """Identifiers are kept whole AND split: `HTTPAdapter.send_request` -> httpadapter, http, adapter, send, ..."""
    out: list[str] = []
    for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        low = word.lower()
        parts = [p.lower() for p in re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", word) if p]
        parts += [p for p in low.split("_") if p]
        for t in {low, *parts}:
            if len(t) > 1 and t not in _STOP:
                out.append(t)
    return out


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank_bm25: int | None = None
    rank_dense: int | None = None

    def to_dict(self, with_code: bool = True, max_code: int = 4000) -> dict:
        c = self.chunk
        d = {"file_path": c.file_path, "symbol_name": c.symbol_name, "qualified_name": c.qualified_name,
             "symbol_type": c.symbol_type, "start_line": c.start_line, "end_line": c.end_line,
             "language": c.language, "signature": c.signature, "docstring": c.docstring[:500],
             "score": round(self.score, 4)}
        if with_code:
            d["code"] = c.code[:max_code]
        return d


_bm25_cache: dict[tuple[str, int], BM25Okapi] = {}
_bm25_lock = threading.Lock()


def _bm25(index: store.RepoIndex) -> BM25Okapi:
    key = (index.namespace, id(index))
    with _bm25_lock:
        if key not in _bm25_cache:
            docs = [tokenize(f"{c.file_path} {c.qualified_name} {c.signature} {c.docstring} {c.code}")
                    for c in index.chunks]
            _bm25_cache[key] = BM25Okapi(docs)
        return _bm25_cache[key]


def bm25_search(index: store.RepoIndex, query: str, k: int) -> list[tuple[int, float]]:
    toks = tokenize(query)
    if not toks or not index.chunks:
        return []
    scores = _bm25(index).get_scores(toks)
    top = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top if scores[i] > 0]


def dense_search(index: store.RepoIndex, query: str, k: int, file: str | None = None,
                 kind: str | None = None) -> list[tuple[int, float]]:
    q = embeddings.embed_query(query, index.meta.get("embed_model"))
    return store.backend().search(index, q, k, file=file, kind=kind)


def _rrf(lists: list[list[tuple[int, float]]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for lst in lists:
        for rank, (i, _) in enumerate(lst):
            fused[i] = fused.get(i, 0.0) + 1.0 / (k + rank + 1)
    return fused


def _weighted(dense: list[tuple[int, float]], sparse: list[tuple[int, float]], w: float = 0.6) -> dict[int, float]:
    def norm(lst):
        if not lst:
            return {}
        vals = [s for _, s in lst]
        lo, hi = min(vals), max(vals)
        return {i: (1.0 if hi == lo else (s - lo) / (hi - lo)) for i, s in lst}
    d, s = norm(dense), norm(sparse)
    return {i: w * d.get(i, 0.0) + (1 - w) * s.get(i, 0.0) for i in set(d) | set(s)}


def search(index: store.RepoIndex, query: str, k: int = 8, *, mode: str = "hybrid", fusion: str | None = None,
           rerank: bool | None = None, weight: float | None = None, file: str | None = None,
           kind: str | None = None) -> list[Hit]:
    """mode: hybrid | bm25 | dense. Filters `file` (exact repo path) and `kind` (function, method, class ...)."""
    settings = get_settings()
    fusion = fusion or settings.fusion
    rerank = bool(settings.rerank_model) if rerank is None else rerank
    n = max(settings.candidates, k)
    sparse = bm25_search(index, query, n * 2) if mode in ("hybrid", "bm25") else []
    if file or kind:
        sparse = [(i, s) for i, s in sparse if (not file or index.chunks[i].file_path == file)
                  and (not kind or index.chunks[i].symbol_type == kind)]
    sparse = sparse[:n]
    dense = dense_search(index, query, n, file, kind) if mode in ("hybrid", "dense") else []

    if mode == "bm25":
        scored = dict(sparse)
    elif mode == "dense":
        scored = dict(dense)
    else:
        w = settings.dense_weight if weight is None else weight
        scored = _rrf([dense, sparse]) if fusion == "rrf" else _weighted(dense, sparse, w)
    ranked = sorted(scored.items(), key=lambda x: -x[1])[: n]
    r_sparse = {i: r for r, (i, _) in enumerate(sparse)}
    r_dense = {i: r for r, (i, _) in enumerate(dense)}
    hits = [Hit(index.chunks[i], s, r_sparse.get(i), r_dense.get(i)) for i, s in ranked]

    if rerank and len(hits) > 1:
        scores = embeddings.rerank(query, [h.chunk.embed_text()[:1500] for h in hits])
        for h, s in zip(hits, scores, strict=False):
            h.score = s
        hits.sort(key=lambda h: -h.score)

    # a query that names a symbol in code form (snake_case, camelCase, Dotted.name, `ticked`, call()) gets it first
    names: set[str] = set()
    for m in re.finditer(r"`?([A-Za-z_][A-Za-z0-9_.]*)`?(\()?", query):
        n = m.group(1).rstrip(".")
        if "_" in n or "." in n or re.search(r"[a-z][A-Z]", n) or m.group(0).startswith("`") or m.group(2):
            names.add(n)
    exact = [h for h in hits if h.chunk.qualified_name in names or h.chunk.symbol_name in names]
    if exact:
        hits = exact + [h for h in hits if h not in exact]
    return hits[:k]


def find_symbol(index: store.RepoIndex, name: str) -> list[Chunk]:
    """Go-to-definition: exact qualified name, then exact short name, then case-insensitive."""
    name = name.strip().strip("`").removesuffix("()")
    for test in (lambda c: c.qualified_name == name, lambda c: c.symbol_name == name,
                 lambda c: c.qualified_name.lower() == name.lower() or c.symbol_name.lower() == name.lower()):
        found = [c for c in index.chunks if c.symbol_type not in ("module", "block") and test(c)]
        if found:
            return sorted(found, key=lambda c: (c.symbol_type == "class", c.file_path))
    return []


def context_block(hits: list[Hit], max_tokens: int | None = None) -> str:
    """The prompt context: numbered sources with exact locations, cut to a token budget (~4 chars a token)."""
    budget = (max_tokens or get_settings().max_context_tokens) * 4
    parts, used = [], 0
    for n, h in enumerate(hits, 1):
        c = h.chunk
        head = f"[{n}] {c.file_path}:{c.start_line}-{c.end_line} `{c.qualified_name}` ({c.symbol_type})"
        body = c.code
        room = budget - used - len(head) - 20
        if room <= 200:
            break
        if len(body) > room:
            body = body[:room] + "\n# ... (truncated)"
        block = f"{head}\n```{c.language}\n{body}\n```"
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)
