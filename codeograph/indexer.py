"""Index a repository: clone (or read a local folder) -> chunk -> embed -> graph -> save.

Re-indexing is incremental: a chunk whose content hash is unchanged keeps its vector from the last index, so only
edited code is embedded again, and an unchanged commit is not re-indexed at all unless forced.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import depgraph, embeddings, store
from .chunking import LANGUAGE_BY_EXT, Chunk, chunk_source
from .config import get_settings

SKIP_DIRS = {"node_modules", ".git", "__pycache__", "dist", "build", ".next", ".nuxt", "venv", ".venv", "env",
             "vendor", "coverage", ".pytest_cache", ".mypy_cache", ".tox", "site-packages", "target", "out",
             ".idea", ".vscode", "third_party"}
MAX_FILE_BYTES = 400 * 1024
Progress = Callable[..., None]


def namespace_for(source: str) -> str:
    """`https://github.com/psf/requests` -> `psf-requests`; a local folder -> its name."""
    s = source.strip().rstrip("/")
    m = re.match(r"(?:https?://|git@)(?:www\.)?[^/:]+[/:]([^/]+)/([^/]+?)(?:\.git)?$", s)
    name = f"{m.group(1)}-{m.group(2)}" if m else Path(s).expanduser().resolve().name
    return re.sub(r"[^a-z0-9_-]+", "-", name.lower()).strip("-") or "repo"


def is_remote(source: str) -> bool:
    return bool(re.match(r"(https?://|git@)", source.strip()))


def collect_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.relative_to(root).parts[:-1]):
            continue
        if path.suffix.lower() not in LANGUAGE_BY_EXT or path.stat().st_size > MAX_FILE_BYTES:
            continue
        raw = path.read_bytes()
        if b"\x00" in raw[:8192]:
            continue
        rel = path.relative_to(root).as_posix()
        if rel.endswith((".min.js", ".bundle.js")):
            continue
        files[rel] = raw.decode("utf-8", errors="replace")
    return files


def _commit_of(root: Path) -> str:
    try:
        from git import Repo

        return Repo(root, search_parent_directories=True).head.commit.hexsha
    except Exception:  # noqa: BLE001 - a plain folder has no commit
        return ""


def index_repository(source: str, branch: str | None = None, force: bool = False,
                     namespace: str | None = None, progress: Progress | None = None) -> dict:
    """Index `source` (a git URL or a local folder). Returns the index's meta."""
    say = progress or (lambda **_: None)
    ns = namespace or namespace_for(source)
    settings = get_settings()
    tmp: str | None = None
    t0 = time.time()
    try:
        if is_remote(source):
            from git import Repo

            say(stage="cloning", current_file="Cloning repository...")
            tmp = tempfile.mkdtemp(prefix="codeograph-")
            Repo.clone_from(source, tmp, depth=1, **({"branch": branch} if branch else {}))
            root = Path(tmp)
        else:
            root = Path(source).expanduser().resolve()
            if not root.is_dir():
                raise FileNotFoundError(f"{source} is neither a git URL nor a folder")
        commit = _commit_of(root)
        old = store.load(ns)
        if old and not force and commit and old.meta.get("commit") == commit \
                and old.meta.get("embed_model") == settings.embed_model:
            say(stage="done", current_file="Already indexed at this commit")
            return {**old.meta, "status": "already_indexed"}

        say(stage="scanning", current_file="Scanning files...")
        files = collect_files(root)
        say(total_files=len(files))
        chunks: list[Chunk] = []
        for i, (rel, text) in enumerate(files.items(), 1):
            chunks.extend(chunk_source(text, rel))
            if i % 25 == 0 or i == len(files):
                say(stage="chunking", processed_files=i, current_file=rel)
        if not chunks:
            raise ValueError("No indexable source files found.")
        say(total_chunks=len(chunks))

        # reuse vectors of unchanged chunks
        reuse: dict[str, np.ndarray] = {}
        if old is not None and old.vectors is not None and old.meta.get("embed_model") == settings.embed_model:
            reuse = {c.content_hash: old.vectors[i] for i, c in enumerate(old.chunks)}
        todo = [i for i, c in enumerate(chunks) if c.content_hash not in reuse]
        say(stage="embedding", current_file=f"Embedding {len(todo)} new or changed chunks "
                                            f"({len(chunks) - len(todo)} reused)")
        new_vecs: dict[int, np.ndarray] = {}
        batch = 64
        for s in range(0, len(todo), batch):
            ids = todo[s: s + batch]
            vecs = embeddings.embed_documents([chunks[i].embed_text() for i in ids])
            new_vecs.update(zip(ids, vecs, strict=False))
            say(indexed_chunks=len(chunks) - len(todo) + min(s + batch, len(todo)))
        dim = next(iter(new_vecs.values())).shape[0] if new_vecs else next(iter(reuse.values())).shape[0]
        vectors = np.zeros((len(chunks), dim), dtype=np.float32)
        for i, c in enumerate(chunks):
            vectors[i] = new_vecs[i] if i in new_vecs else reuse[c.content_hash]

        say(stage="graph", current_file="Building dependency graph...")
        graph = depgraph.build_graph(files, chunks)
        calls = sum(1 for *_, d in graph.edges(data=True) if d.get("type") == "calls")
        imports = sum(1 for *_, d in graph.edges(data=True) if d.get("type") == "imports")
        meta = {
            "namespace": ns,
            "repo_url": source if is_remote(source) else str(root),
            "branch": branch or "",
            "commit": commit,
            "indexed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "embed_model": settings.embed_model,
            "chunk_count": len(chunks),
            "file_count": len(files),
            "languages": sorted({c.language for c in chunks}),
            "graph": {"nodes": graph.number_of_nodes(), "calls": calls, "imports": imports},
            "embedded": len(todo),
            "reused": len(chunks) - len(todo),
            "seconds": round(time.time() - t0, 1),
            "file_hashes": {p: hashlib.sha1(t.encode()).hexdigest()[:12] for p, t in files.items()},
        }
        store.save(ns, meta, chunks, vectors, graph, files)
        say(stage="done", current_file="Done")
        return {**meta, "status": "indexed"}
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
