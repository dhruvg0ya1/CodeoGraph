"""
main.py

FastAPI backend for RepoChat.

Key architectural decisions:
- POST /index returns immediately (BackgroundTask) so the UI never freezes
- POST /query uses StreamingResponse for real-time token streaming
- CORS is wide open (* origins) since this is a local dev tool
- All state lives in-process memory (bm25_corpus, dependency_graph, indexing_state)
  which means a server restart clears BM25 / graphs but Pinecone data persists
"""

import asyncio
import json
import os
from typing import AsyncGenerator, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pinecone import Pinecone
from pydantic import BaseModel

import indexing_state as istate
from diagram import generate_diagram
from graph import analyze_impact
from ingestion import get_pinecone_index, run_ingestion
from pr_review import review_pr, review_to_markdown
from retrieval import hybrid_search, symbol_lookup
from utils import (
    build_context_block,
    delete_indexed_repo,
    get_repo_info,
    load_indexed_repos,
    repo_url_to_namespace,
)

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "codeograph")

genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI(title="CodeoGraph API", version="1.0.0")

# CORS — must be added before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class IndexRequest(BaseModel):
    repo_url: str
    branch: Optional[str] = None
    force_reindex: bool = False


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str
    namespace: str
    history: list[ChatMessage] = []
    filter_file: Optional[str] = None
    filter_type: Optional[str] = None


class ImpactRequest(BaseModel):
    symbol: str
    namespace: str


class ReviewRequest(BaseModel):
    diff: str
    namespace: str


# ---------------------------------------------------------------------------
# POST /index
# ---------------------------------------------------------------------------


@app.post("/index")
async def index_repo(req: IndexRequest, background_tasks: BackgroundTasks):
    namespace = repo_url_to_namespace(req.repo_url)
    existing = get_repo_info(namespace)

    if existing and not req.force_reindex:
        return {
            "status": "already_indexed",
            "namespace": namespace,
            "chunk_count": existing["chunk_count"],
            "file_count": existing["file_count"],
        }

    # Initialize state before kicking off background task
    istate.init_state(namespace)

    background_tasks.add_task(run_ingestion, req.repo_url, namespace, req.branch)

    return {"status": "started", "namespace": namespace}


# ---------------------------------------------------------------------------
# GET /index/status
# ---------------------------------------------------------------------------


@app.get("/index/status")
async def index_status(namespace: str = Query(...)):
    s = istate.get_state(namespace)
    if not s:
        # No job running — check if it's already indexed
        info = get_repo_info(namespace)
        if info:
            return {"status": "done", "namespace": namespace, **info}
        return {"status": "idle", "namespace": namespace}
    return s


# ---------------------------------------------------------------------------
# POST /query  (streaming)
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTION = (
    "You are an expert code assistant for the indexed repository. "
    "Answer questions using only the provided code context. "
    "Always cite the source file and function name using backtick formatting. "
    "If you reference code from a previous answer, say so explicitly. "
    "Be precise and technical. Format all code with markdown code blocks."
)


async def _stream_gemini(
    question: str,
    history: list[ChatMessage],
    context_block: str,
) -> AsyncGenerator[str, None]:
    """
    Wire together Gemini streaming → FastAPI StreamingResponse chunks.

    We run the blocking Gemini call in a thread pool so we don't block the
    asyncio event loop. Each chunk is yielded as Server-Sent Events (text/plain
    newline-delimited) so Streamlit can consume with st.write_stream.
    """
    model = genai.GenerativeModel(
        "gemini-2.5-flash",
        system_instruction=_SYSTEM_INSTRUCTION,
    )

    # Build conversation history (last 6 turns max to keep context manageable)
    gemini_history = []
    for msg in history[-6:]:
        role = "user" if msg.role == "user" else "model"
        gemini_history.append({"role": role, "parts": [msg.content]})

    # Context + question as the final user turn
    final_user_turn = (
        f"Here is the relevant code context:\n\n{context_block}\n\n"
        f"---\n\nQuestion: {question}"
    )

    chat = model.start_chat(history=gemini_history)

    # Run blocking call in thread executor
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: chat.send_message(final_user_turn, stream=True),
    )

    for chunk in response:
        if chunk.text:
            yield chunk.text


@app.post("/query")
async def query_codebase(req: QueryRequest):
    if not req.namespace:
        raise HTTPException(status_code=400, detail="namespace is required")

    # Retrieve relevant chunks
    chunks = hybrid_search(
        query=req.question,
        namespace=req.namespace,
        top_k=5,
        filter_file=req.filter_file,
        filter_type=req.filter_type,
    )

    context_block = build_context_block(chunks) if chunks else "No relevant code found."

    # Sources metadata to send back — we embed them in a special trailing marker
    # that the frontend strips out. This lets us stream text and still deliver metadata.
    sources = [
        {
            "file_path": c.get("file_path", ""),
            "symbol_name": c.get("symbol_name", ""),
            "start_line": c.get("start_line", 0),
            "end_line": c.get("end_line", 0),
            "docstring": c.get("docstring", ""),
        }
        for c in chunks
    ]

    async def event_stream():
        async for text_chunk in _stream_gemini(req.question, req.history, context_block):
            yield text_chunk
        # Append sources as a special delimiter the frontend can parse
        yield f"\n\n<!--SOURCES:{json.dumps(sources)}-->"

    return StreamingResponse(event_stream(), media_type="text/plain")


# ---------------------------------------------------------------------------
# GET /symbol
# ---------------------------------------------------------------------------


@app.get("/symbol")
async def get_symbol(name: str = Query(...), namespace: str = Query(...)):
    result = symbol_lookup(name, namespace)
    if not result:
        raise HTTPException(status_code=404, detail=f"Symbol '{name}' not found.")
    return result


# ---------------------------------------------------------------------------
# POST /impact
# ---------------------------------------------------------------------------


@app.post("/impact")
async def change_impact(req: ImpactRequest):
    result = analyze_impact(req.symbol, req.namespace)
    return result


# ---------------------------------------------------------------------------
# POST /review
# ---------------------------------------------------------------------------


@app.post("/review")
async def pr_review(req: ReviewRequest):
    if not req.diff.strip():
        raise HTTPException(status_code=400, detail="diff cannot be empty")
    result = review_pr(req.diff, req.namespace)
    result["markdown"] = review_to_markdown(result, req.diff)
    return result


# ---------------------------------------------------------------------------
# GET /diagram
# ---------------------------------------------------------------------------


@app.get("/diagram")
async def architecture_diagram(namespace: str = Query(...)):
    return generate_diagram(namespace)


# ---------------------------------------------------------------------------
# GET /repos
# ---------------------------------------------------------------------------


@app.get("/repos")
async def list_repos():
    return load_indexed_repos()


# ---------------------------------------------------------------------------
# DELETE /repos/{namespace}
# ---------------------------------------------------------------------------


@app.delete("/repos/{namespace}")
async def delete_repo(namespace: str):
    # Delete from Pinecone
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        index.delete(delete_all=True, namespace=namespace)
    except Exception as e:
        print(f"[WARN] Could not delete Pinecone namespace {namespace}: {e}")

    # Remove from local registry
    delete_indexed_repo(namespace)

    # Clear in-memory stores
    from ingestion import bm25_corpus, dependency_graph
    bm25_corpus.pop(namespace, None)
    dependency_graph.pop(namespace, None)

    return {"status": "deleted", "namespace": namespace}


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok"}
