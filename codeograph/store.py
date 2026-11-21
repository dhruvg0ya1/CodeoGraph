"""One indexed repository on disk, and the vector backend behind it.

    ~/.codeograph/indexes/<namespace>/
        meta.json      source, commit, models, counts, file hashes
        chunks.jsonl   every chunk (BM25 corpus, symbol table, citations)
        vectors.npy    float32, L2-normalised, row i = chunk i   (local backend)
        graph.json     dependency graph (node-link)
        files.json     repo-relative path -> source text (for get_file and diff review)

Everything survives a restart. Vectors live in the index folder by default (exact cosine search, lock-free, so the
API, the MCP server and the CLI can read one index at the same time) or in a Qdrant server when QDRANT_URL is set.
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np

from . import depgraph
from .chunking import Chunk
from .config import get_settings


def index_dir(namespace: str) -> Path:
    return get_settings().index_root / namespace


@dataclass
class RepoIndex:
    namespace: str
    meta: dict
    chunks: list[Chunk]
    graph: nx.DiGraph
    files: dict[str, str]
    vectors: np.ndarray | None = None                 # None when a Qdrant server holds them
    _by_id: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {c.id: i for i, c in enumerate(self.chunks)}

    def position(self, chunk_id: str) -> int | None:
        return self._by_id.get(chunk_id)


# --------------------------------------------------------------------------- vector backends
class VectorBackend:
    def write(self, namespace: str, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def search(self, index: RepoIndex, query: np.ndarray, k: int, file: str | None = None,
               kind: str | None = None) -> list[tuple[int, float]]: ...
    def delete(self, namespace: str) -> None: ...


class LocalBackend(VectorBackend):
    """Exact cosine search with numpy. A 20k-chunk repo is a 30 MB matrix and a ~2 ms query."""

    def write(self, namespace: str, chunks: list[Chunk], vectors: np.ndarray) -> None:
        pass                                            # save() writes vectors.npy inside the index folder

    def search(self, index, query, k, file=None, kind=None):
        if index.vectors is None or not len(index.vectors):
            return []
        scores = index.vectors @ query
        mask = np.ones(len(scores), dtype=bool)
        if file:
            mask &= np.array([c.file_path == file for c in index.chunks])
        if kind:
            mask &= np.array([c.symbol_type == kind for c in index.chunks])
        scores = np.where(mask, scores, -np.inf)
        k = min(k, int(mask.sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def delete(self, namespace: str) -> None:
        pass


class QdrantBackend(VectorBackend):
    """Vectors in a Qdrant server (QDRANT_URL, optional QDRANT_API_KEY); one collection per repository."""

    def __init__(self, url: str):
        import os

        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY"))

    @staticmethod
    def _name(namespace: str) -> str:
        return f"codeograph_{namespace}"

    def write(self, namespace, chunks, vectors):
        from qdrant_client import models

        name = self._name(namespace)
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(name, vectors_config=models.VectorParams(
            size=int(vectors.shape[1]), distance=models.Distance.COSINE))
        for s in range(0, len(chunks), 256):
            batch = chunks[s: s + 256]
            self.client.upsert(name, points=[
                models.PointStruct(id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.id)), vector=vectors[s + j].tolist(),
                                   payload={"pos": s + j, "file_path": c.file_path, "symbol_type": c.symbol_type})
                for j, c in enumerate(batch)])

    def search(self, index, query, k, file=None, kind=None):
        from qdrant_client import models

        must = []
        if file:
            must.append(models.FieldCondition(key="file_path", match=models.MatchValue(value=file)))
        if kind:
            must.append(models.FieldCondition(key="symbol_type", match=models.MatchValue(value=kind)))
        res = self.client.query_points(self._name(index.namespace), query=query.tolist(), limit=k,
                                       query_filter=models.Filter(must=must) if must else None, with_payload=True)
        return [(int(p.payload["pos"]), float(p.score)) for p in res.points]

    def delete(self, namespace):
        if self.client.collection_exists(self._name(namespace)):
            self.client.delete_collection(self._name(namespace))


def backend() -> VectorBackend:
    url = get_settings().qdrant_url
    return QdrantBackend(url) if url else LocalBackend()


# --------------------------------------------------------------------------- save / load
def save(namespace: str, meta: dict, chunks: list[Chunk], vectors: np.ndarray, graph: nx.DiGraph,
         files: dict[str, str]) -> None:
    d = index_dir(namespace)
    tmp = d.with_name(d.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with (tmp / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict()) + "\n")
    (tmp / "graph.json").write_text(depgraph.to_json(graph))
    (tmp / "files.json").write_text(json.dumps(files))
    remote = bool(get_settings().qdrant_url)
    if remote:                                          # vectors go to the server before the index is visible
        backend().write(namespace, chunks, vectors)
    else:
        np.save(tmp / "vectors.npy", vectors.astype(np.float32))
    meta = {**meta, "vector_backend": "qdrant" if remote else "local"}
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    if d.exists():                                      # swap in one step: readers never see half an index
        old = d.with_name(d.name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        d.rename(old)
        tmp.rename(d)
        shutil.rmtree(old, ignore_errors=True)
    else:
        tmp.rename(d)
    _CACHE.pop(namespace, None)


_CACHE: dict[str, tuple[float, RepoIndex]] = {}


def load(namespace: str) -> RepoIndex | None:
    d = index_dir(namespace)
    meta_path = d / "meta.json"
    if not meta_path.exists():
        return None
    stamp = meta_path.stat().st_mtime
    hit = _CACHE.get(namespace)
    if hit and hit[0] == stamp:
        return hit[1]
    meta = json.loads(meta_path.read_text())
    with (d / "chunks.jsonl").open() as f:
        chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
    graph = depgraph.from_json((d / "graph.json").read_text())
    files = json.loads((d / "files.json").read_text())
    vec_path = d / "vectors.npy"
    vectors = np.load(vec_path) if vec_path.exists() else None
    idx = RepoIndex(namespace, meta, chunks, graph, files, vectors)
    _CACHE[namespace] = (stamp, idx)
    return idx


def list_indexes() -> list[dict]:
    root = get_settings().index_root
    out = []
    for m in sorted(root.glob("*/meta.json")) if root.exists() else []:
        try:
            meta = json.loads(m.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out.append({k: meta.get(k) for k in ("namespace", "repo_url", "commit", "indexed_at", "chunk_count",
                                             "file_count", "languages", "embed_model")})
    return out


def delete(namespace: str) -> bool:
    d = index_dir(namespace)
    existed = d.exists()
    shutil.rmtree(d, ignore_errors=True)
    try:
        backend().delete(namespace)
    except Exception:  # noqa: BLE001 - a missing remote collection is not an error here
        pass
    _CACHE.pop(namespace, None)
    return existed
