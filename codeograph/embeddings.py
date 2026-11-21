"""Local embeddings and reranking with FastEmbed (ONNX Runtime, CPU): no API key, no GPU, no per-call cost.

Models download once from Hugging Face into ~/.codeograph/models and are cached per process.
"""

from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

from .config import get_settings

_lock = threading.Lock()


def _cache_dir() -> str:
    d = get_settings().home / "models"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


@lru_cache(maxsize=4)
def _embedder(name: str):
    from fastembed import TextEmbedding

    return TextEmbedding(name, cache_dir=_cache_dir())


@lru_cache(maxsize=2)
def _reranker(name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(name, cache_dir=_cache_dir())


def _hash_embed(texts: list[str], dim: int = 256) -> np.ndarray:
    """CODEOGRAPH_EMBED_MODEL=hash: deterministic feature hashing of code tokens. No download, no model - for tests
    and air-gapped smoke runs, not for quality."""
    import zlib

    from .search import tokenize

    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for tok in tokenize(t):
            h = zlib.crc32(tok.encode())
            out[i, h % dim] += 1.0 if (h >> 16) & 1 else -1.0
    return out


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


def embed_documents(texts: list[str], model: str | None = None, batch_size: int = 16) -> np.ndarray:
    name = model or get_settings().embed_model
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    if name == "hash":
        return _normalise(_hash_embed(texts))
    with _lock:
        vecs = list(_embedder(name).passage_embed(texts, batch_size=batch_size))
    return _normalise(np.asarray(vecs, dtype=np.float32))


def embed_query(text: str, model: str | None = None) -> np.ndarray:
    name = model or get_settings().embed_model
    if name == "hash":
        return _normalise(_hash_embed([text]))[0]
    with _lock:
        vec = next(iter(_embedder(name).query_embed(text)))
    return _normalise(np.asarray([vec], dtype=np.float32))[0]


def rerank(query: str, docs: list[str], model: str | None = None) -> list[float]:
    """Cross-encoder relevance score per doc (higher = more relevant)."""
    name = model or get_settings().rerank_model
    if not name or not docs:
        return [0.0] * len(docs)
    with _lock:
        return [float(s) for s in _reranker(name).rerank(query, docs)]
