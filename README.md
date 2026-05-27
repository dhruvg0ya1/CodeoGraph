# 🤖 CodeoGraph

**AI-powered codebase understanding agent.** Index any GitHub repo and chat with it, analyze change impact, review PRs, and visualize architecture — all grounded in your actual source code.

---

## Features

| Feature | Description |
|---|---|
| **AST Chunking** | Tree-sitter extracts functions/classes as semantic chunks, not fixed tokens |
| **Hybrid Retrieval** | BM25 + vector search fused (0.6/0.4 weighting) for precision and recall |
| **RAG Chat** | Streaming Gemini responses grounded in retrieved code context |
| **Symbol Search** | "Go to definition" — find any function or class instantly |
| **Change Impact** | NetworkX reverse-BFS finds all dependents; Gemini rates each as Low/Medium/High risk |
| **PR Review** | Paste a git diff → structured review with risks, tests, and risk score |
| **Architecture Diagram** | Auto-generated Mermaid diagram from static import analysis |
| **Multi-repo** | Pinecone namespaces isolate repos; switch between them in the sidebar |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

Copy `.env` and fill in your keys:

```bash
cp .env .env.local
```

```
GOOGLE_API_KEY=...       # Google AI Studio — free tier works
PINECONE_API_KEY=...     # Pinecone serverless — free tier works
PINECONE_INDEX_NAME=codeograph
```

> **Pinecone index** is created automatically on first run (dimension=3072, cosine).

### 3. Start the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 4. Start the frontend

```bash
cd frontend
streamlit run app.py --server.port 8501
```

Open **http://localhost:8501**

---

## Architecture

```
frontend/app.py (Streamlit)
        │  HTTP / SSE
        ▼
backend/main.py (FastAPI)
    ├── ingestion.py   → GitPython clone → Tree-sitter parse → Google Embed → Pinecone upsert
    ├── retrieval.py   → Pinecone vector search + BM25 → score fusion
    ├── graph.py       → NetworkX reverse-BFS → Gemini risk assessment
    ├── pr_review.py   → Diff parser → context retrieval → Gemini review
    └── diagram.py     → Graph → Mermaid syntax
```

---

## Tech Stack

- **LLM**: Gemini 2.5 Flash (streaming)
- **Embeddings**: Google `models/gemini-embedding-2` (3072d)
- **Vector DB**: Pinecone serverless (cosine)
- **Code Parsing**: Tree-sitter (Python + JavaScript grammars)
- **Keyword Search**: BM25Okapi via `rank-bm25`
- **Dependency Graph**: NetworkX DiGraph
- **Repo Ingestion**: GitPython
- **Backend**: FastAPI + uvicorn
- **Frontend**: Streamlit

---

## Notes

- **Server restart** clears BM25 corpus and dependency graphs from memory. Pinecone data persists. Re-querying will still work (vector search only); re-indexing restores BM25 + graphs.
- **Rate limits**: Embedding batches sleep 1s between calls. For large repos (500+ files) expect 5–15 min indexing.
- **Pinecone index dimension**: If the existing index was created with a different embedding model, delete it or set a new `PINECONE_INDEX_NAME` before re-indexing.
- **Token safety**: Prompts are capped at 6000 context tokens before sending to Gemini.
