"""
retrieval.py

Hybrid retrieval combining dense vector search (Pinecone) and sparse BM25
keyword search, fused with weighted score combination.

Why hybrid?
- Vector search captures semantic similarity (what does X conceptually relate to?)
- BM25 captures exact identifier matches (find 'authenticate_user' specifically)
- Score fusion gives us the best of both worlds: 0.6 * vector + 0.4 * BM25

Score normalization: both lists are independently min-max normalized to [0,1]
before fusion so that different score scales don't dominate.
"""

import os
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi

import indexing_state
from utils import tokenize_code, truncate_to_token_budget

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "codeograph")

genai.configure(api_key=GOOGLE_API_KEY)

_pc: Optional[Pinecone] = None


def _get_index():
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    return _pc.Index(PINECONE_INDEX_NAME)


def _normalize_scores(items: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
    """Min-max normalize scores to [0, 1]."""
    if not items:
        return []
    scores = [s for _, s in items]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [(doc, 1.0) for doc, _ in items]
    return [(doc, (s - lo) / (hi - lo)) for doc, s in items]


# ---------------------------------------------------------------------------
# Dense vector search
# ---------------------------------------------------------------------------

def vector_search(
    query: str,
    namespace: str,
    top_k: int = 10,
    filter_file: Optional[str] = None,
    filter_type: Optional[str] = None,
) -> list[tuple[dict, float]]:
    """
    Embed query and search Pinecone. Returns list of (metadata_dict, score) tuples.
    Optional metadata filters narrow results to a specific file or symbol type.
    """
    result = genai.embed_content(
        model="models/gemini-embedding-2",
        content=query,
        task_type="retrieval_query",
    )
    query_vector = result["embedding"]

    # Build metadata filter for Pinecone (uses their filter syntax)
    pf: Optional[dict] = None
    conditions = {}
    if filter_file:
        conditions["file_path"] = {"$eq": filter_file}
    if filter_type:
        conditions["symbol_type"] = {"$eq": filter_type}
    if conditions:
        pf = conditions

    index = _get_index()
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        filter=pf,
    )

    results = []
    for match in response.matches:
        if match.metadata:
            results.append((match.metadata, match.score))
    return results


# ---------------------------------------------------------------------------
# BM25 keyword search
# ---------------------------------------------------------------------------

def bm25_search(
    query: str,
    namespace: str,
    top_k: int = 10,
    bm25_corpus: Optional[dict] = None,
) -> list[tuple[dict, float]]:
    """
    BM25 over the in-memory chunk corpus for this namespace.
    Falls back to empty results if corpus not loaded (e.g., after server restart).

    The corpus is imported from ingestion.py at call time to avoid circular imports.
    """
    if bm25_corpus is None:
        # Lazy import to avoid circular dependency at module level
        from ingestion import bm25_corpus as _corpus
        bm25_corpus = _corpus

    corpus = bm25_corpus.get(namespace, [])
    if not corpus:
        return []

    # Tokenize corpus once per namespace (cache would be nicer but corpus is stable)
    tokenized_corpus = [tokenize_code(c.get("raw_code", "")) for c in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = tokenize_code(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    # Pair each chunk with its score
    scored = [(corpus[i], float(scores[i])) for i in range(len(corpus))]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Score fusion
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    namespace: str,
    top_k: int = 5,
    filter_file: Optional[str] = None,
    filter_type: Optional[str] = None,
    bm25_corpus: Optional[dict] = None,
) -> list[dict]:
    """
    Fuse vector and BM25 results.

    Fusion formula:
        combined_score = 0.6 * vector_norm + 0.4 * bm25_norm

    A higher weight for vector search reflects that semantic understanding
    is usually more important than exact keyword matching for code Q&A.
    Returns up to top_k chunks, token-budget-trimmed, as plain dicts.
    """
    vec_results = vector_search(query, namespace, top_k=10, filter_file=filter_file, filter_type=filter_type)
    bm_results = bm25_search(query, namespace, top_k=10, bm25_corpus=bm25_corpus)

    vec_norm = _normalize_scores(vec_results)
    bm_norm = _normalize_scores(bm_results)

    # Build score map keyed by (file_path, symbol_name, start_line)
    # so we can merge the two ranked lists
    def chunk_key(meta: dict) -> tuple:
        return (meta.get("file_path", ""), meta.get("symbol_name", ""), meta.get("start_line", 0))

    score_map: dict[tuple, dict] = {}

    for meta, score in vec_norm:
        k = chunk_key(meta)
        score_map[k] = {"meta": meta, "vector": score, "bm25": 0.0}

    for meta, score in bm_norm:
        k = chunk_key(meta)
        if k in score_map:
            score_map[k]["bm25"] = score
        else:
            score_map[k] = {"meta": meta, "vector": 0.0, "bm25": score}

    # Compute fused scores and sort
    fused = []
    for k, v in score_map.items():
        combined = 0.6 * v["vector"] + 0.4 * v["bm25"]
        fused.append((v["meta"], combined))

    fused.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [meta for meta, _ in fused[:top_k]]

    # Token budget protection before returning
    return truncate_to_token_budget(top_chunks, max_tokens=6000)


# ---------------------------------------------------------------------------
# Exact symbol lookup
# ---------------------------------------------------------------------------

def symbol_lookup(name: str, namespace: str) -> Optional[dict]:
    """
    Try exact metadata filter first; fall back to semantic search if not found.
    This mirrors a "Go to Definition" feature.
    """
    index = _get_index()
    # Pinecone metadata filter for exact name match
    try:
        # We query with a zero vector — Pinecone requires a query vector even for
        # pure metadata filters, so we send a dummy and rely entirely on the filter.
        dummy = [0.0] * 3072
        response = index.query(
            vector=dummy,
            top_k=1,
            namespace=namespace,
            include_metadata=True,
            filter={"symbol_name": {"$eq": name}},
        )
        if response.matches and response.matches[0].metadata:
            return response.matches[0].metadata
    except Exception:
        pass

    # Semantic fallback
    results = vector_search(name, namespace, top_k=1)
    if results:
        return results[0][0]
    return None
