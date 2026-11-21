"""REST API (FastAPI) - what the Streamlit UI talks to, and a plain HTTP interface for anything else.

    codeograph serve --port 8000      # docs at /docs
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from . import __version__, agent, diagram, impact, indexer, llm, review, search, store

app = FastAPI(title="CodeoGraph", version=__version__,
              description="Grounded answers, change impact and PR review over any codebase.")
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=2)


class IndexRequest(BaseModel):
    repo_url: str
    branch: str | None = None
    force_reindex: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    namespace: str
    history: list[ChatMessage] = []
    thread_id: str | None = None
    filter_file: str | None = None
    filter_type: str | None = None


class ImpactRequest(BaseModel):
    symbol: str
    namespace: str
    max_hops: int = 3


class ReviewRequest(BaseModel):
    diff: str
    namespace: str


def _load(namespace: str) -> store.RepoIndex:
    idx = store.load(namespace)
    if idx is None:
        raise HTTPException(404, f"Repository '{namespace}' is not indexed.")
    return idx


def _run_index(req: IndexRequest, ns: str) -> None:
    def progress(**kw):
        with _jobs_lock:
            _jobs[ns].update(kw)
    try:
        meta = indexer.index_repository(req.repo_url, branch=req.branch, force=req.force_reindex, namespace=ns,
                                        progress=progress)
        with _jobs_lock:
            _jobs[ns].update(status="done", chunk_count=meta.get("chunk_count"), file_count=meta.get("file_count"),
                             indexed_chunks=meta.get("chunk_count"))
    except Exception as e:  # noqa: BLE001 - reported through /index/status
        with _jobs_lock:
            _jobs[ns].update(status="error", error=str(e))


@app.post("/index")
def index_repo(req: IndexRequest):
    ns = indexer.namespace_for(req.repo_url)
    existing = store.load(ns)
    if existing and not req.force_reindex:
        return {"status": "already_indexed", "namespace": ns, "chunk_count": existing.meta["chunk_count"],
                "file_count": existing.meta["file_count"]}
    with _jobs_lock:
        if _jobs.get(ns, {}).get("status") == "running":
            return {"status": "started", "namespace": ns}
        _jobs[ns] = {"namespace": ns, "status": "running", "total_files": 0, "processed_files": 0,
                     "total_chunks": 0, "indexed_chunks": 0, "current_file": "Queued", "error": None}
    _pool.submit(_run_index, req, ns)
    return {"status": "started", "namespace": ns}


@app.get("/index/status")
def index_status(namespace: str = Query(...)):
    with _jobs_lock:
        job = dict(_jobs.get(namespace, {}))
    if job and job.get("status") == "running":
        return job
    idx = store.load(namespace)
    if idx:
        return {"status": "done", **{k: idx.meta.get(k) for k in ("namespace", "repo_url", "chunk_count",
                                                                   "file_count", "languages", "indexed_at")}}
    return job or {"status": "idle", "namespace": namespace}


@app.post("/query")
async def query(req: QueryRequest):
    """Streams the agent's answer as text, then `<!--SOURCES:[...]-->` with the cited code locations."""
    _load(req.namespace)
    history, pending = [], None
    for m in req.history[-8:]:
        if m.role == "user":
            pending = m.content
        elif pending is not None:
            history.append({"q": pending, "a": m.content})
            pending = None
    def events():
        for kind, payload in agent.stream(req.question, req.namespace, thread_id=req.thread_id, history=history,
                                          file=req.filter_file, kind=req.filter_type):
            if kind == "token":
                yield payload
            else:
                yield f"\n\n<!--SOURCES:{json.dumps(payload['citations'])}-->"

    return StreamingResponse(iterate_in_threadpool(events()), media_type="text/plain")


@app.post("/ask")
async def ask(req: QueryRequest):
    """The same agent as /query, as one JSON response (answer, citations, trace)."""
    _load(req.namespace)
    return await run_in_threadpool(agent.ask, req.question, req.namespace, req.thread_id, None,
                                   req.filter_file, req.filter_type)


@app.get("/search")
def search_code(namespace: str, q: str, k: int = 8, file: str | None = None, kind: str | None = None,
                mode: str = "hybrid"):
    if mode not in ("hybrid", "bm25", "dense"):
        raise HTTPException(400, "mode must be hybrid, bm25 or dense")
    return [h.to_dict(max_code=2500) for h in search.search(_load(namespace), q, k=min(max(k, 1), 30),
                                                             mode=mode, file=file, kind=kind)]


@app.get("/symbol")
def symbol(name: str = Query(...), namespace: str = Query(...)):
    found = search.find_symbol(_load(namespace), name)
    if not found:
        raise HTTPException(404, f"Symbol '{name}' not found.")
    d = search.Hit(found[0], 1.0).to_dict(max_code=20000)
    d["raw_code"] = d["code"]
    d["also_defined_in"] = [f"{c.file_path}:{c.start_line}" for c in found[1:6]]
    return d


@app.post("/impact")
def change_impact(req: ImpactRequest):
    return impact.analyze(_load(req.namespace), req.symbol, max_hops=max(1, min(req.max_hops, 5)))


@app.post("/review")
def pr_review(req: ReviewRequest):
    if not req.diff.strip():
        raise HTTPException(400, "diff cannot be empty")
    r = review.review(_load(req.namespace), req.diff)
    r["markdown"] = review.to_markdown(r, req.diff)
    return r


@app.get("/diagram")
def architecture(namespace: str = Query(...)):
    return diagram.generate(_load(namespace))


@app.get("/repos")
def repos():
    return store.list_indexes()


@app.delete("/repos/{namespace}")
def delete_repo(namespace: str):
    if not store.delete(namespace):
        raise HTTPException(404, f"Repository '{namespace}' is not indexed.")
    with _jobs_lock:
        _jobs.pop(namespace, None)
    return {"deleted": namespace}


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "llm": llm.model_label(), "repos": len(store.list_indexes())}
