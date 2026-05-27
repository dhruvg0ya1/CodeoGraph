"""
ingestion.py

Full repo ingestion pipeline:
  1. Clone repo from GitHub via GitPython
  2. Walk files, filter by language/size
  3. Parse each file with Tree-sitter to extract function/class/module chunks
  4. Embed chunks via Google models/gemini-embedding-2
  5. Upsert to Pinecone (namespaced per repo)
  6. Build NetworkX dependency graph
  7. Update live progress via indexing_state

Tree-sitter note: we compile language grammars once on module load.
If a file fails to parse (syntax errors, unsupported dialect), we fall back
to treating the whole file as a single "module" chunk.
"""

import ast
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import google.generativeai as genai
import networkx as nx
from dotenv import load_dotenv
from git import Repo as GitRepo
from pinecone import Pinecone, ServerlessSpec
from tree_sitter import Language, Parser

import indexing_state as state
from utils import (
    build_chunk_text,
    normalize_github_url,
    repo_url_to_namespace,
    save_indexed_repo,
    tokenize_code,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "codeograph")

genai.configure(api_key=GOOGLE_API_KEY)

EMBED_BATCH_SIZE = 50
EMBED_SLEEP_SEC = 1.0       # avoid Google 429 rate limits between batches
EMBED_DIMENSION = 3072     # gemini-embedding-2 returns 3072-dimensional vectors
MIN_CHUNK_LINES = 5         # skip chunks shorter than this
MAX_FILE_SIZE_BYTES = 300 * 1024  # 300 KB

SUPPORTED_EXTENSIONS = {".py", ".js", ".ts", ".java", ".go", ".cpp", ".c", ".cs", ".rb", ".php"}
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", "dist", "build",
    ".next", "venv", "env", ".env", "vendor", "coverage", ".pytest_cache",
}

# ---------------------------------------------------------------------------
# Tree-sitter setup
# ---------------------------------------------------------------------------
# We load the pre-compiled shared libraries bundled by tree-sitter-python /
# tree-sitter-javascript PyPI packages (>= 0.21).

try:
    import tree_sitter_python as tspython
    import tree_sitter_javascript as tsjavascript

    PY_LANGUAGE = Language(tspython.language())
    JS_LANGUAGE = Language(tsjavascript.language())

    _py_parser = Parser()
    _py_parser.set_language(PY_LANGUAGE)

    _js_parser = Parser()
    _js_parser.set_language(JS_LANGUAGE)

    TREE_SITTER_AVAILABLE = True
except Exception as e:
    print(f"[WARNING] Tree-sitter unavailable: {e}. Falling back to file-level chunks.")
    TREE_SITTER_AVAILABLE = False
    _py_parser = _js_parser = None

# ---------------------------------------------------------------------------
# In-memory stores (shared with retrieval.py via import)
# ---------------------------------------------------------------------------

# bm25_corpus[namespace] = list of chunk dicts (same objects stored in Pinecone)
bm25_corpus: dict[str, list[dict]] = {}

# dependency_graph[namespace] = nx.DiGraph
dependency_graph: dict[str, nx.DiGraph] = {}

# ---------------------------------------------------------------------------
# Pinecone client (lazy init)
# ---------------------------------------------------------------------------

_pc: Optional[Pinecone] = None


def _get_index_dimension(index_desc) -> Optional[int]:
    if index_desc is None:
        return None
    if hasattr(index_desc, "dimension"):
        return getattr(index_desc, "dimension")
    if hasattr(index_desc, "dims"):
        return getattr(index_desc, "dims")
    if isinstance(index_desc, dict):
        return index_desc.get("dimension") or index_desc.get("dims")
    return None


def get_pinecone_index():
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in _pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        _pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait for index to be ready
        time.sleep(5)
    else:
        try:
            index_desc = _pc.describe_index(PINECONE_INDEX_NAME)
            existing_dim = _get_index_dimension(index_desc)
            if existing_dim and existing_dim != EMBED_DIMENSION:
                raise RuntimeError(
                    f"Pinecone index '{PINECONE_INDEX_NAME}' exists with dimension {existing_dim}, "
                    f"but current embedding model produces dimension {EMBED_DIMENSION}. "
                    "Delete the index in Pinecone or set a new PINECONE_INDEX_NAME namespace."
                )
        except Exception as exc:
            print(f"[ERROR] Failed to validate Pinecone index dimension: {exc}")
            raise
    return _pc.Index(PINECONE_INDEX_NAME)


# ---------------------------------------------------------------------------
# File walking
# ---------------------------------------------------------------------------

def _is_binary(path: Path) -> bool:
    """Quick binary check: look for null bytes in the first 8KB."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        return b"\x00" in chunk
    except OSError:
        return True


def collect_files(repo_dir: str) -> list[Path]:
    """Return all source files that should be indexed."""
    root = Path(repo_dir)
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        # Skip files inside forbidden directories
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix not in SUPPORTED_EXTENSIONS:
            continue
        if path.stat().st_size > MAX_FILE_SIZE_BYTES:
            continue
        if _is_binary(path):
            continue
        files.append(path)
    return files


# ---------------------------------------------------------------------------
# AST chunking — Python (tree-sitter)
# ---------------------------------------------------------------------------

def _get_docstring_ts(node, source_bytes: bytes) -> str:
    """Extract docstring from a function/class body using tree-sitter nodes."""
    # Body node is typically the last child; first expression_statement inside
    # body is the docstring if it's a string literal.
    for child in node.children:
        if child.type == "block":
            for stmt in child.children:
                if stmt.type == "expression_statement":
                    for inner in stmt.children:
                        if inner.type == "string":
                            raw = source_bytes[inner.start_byte:inner.end_byte].decode("utf-8", errors="replace")
                            return raw.strip("'\"` \n").strip()
    return ""


def _chunk_python_ts(source: str, rel_path: str) -> list[dict]:
    """
    Use Tree-sitter to extract Python functions and classes as individual chunks.
    Module-level code outside any definition becomes one "module" chunk per file.
    """
    source_bytes = source.encode("utf-8")
    tree = _py_parser.parse(source_bytes)
    lines = source.splitlines()
    chunks: list[dict] = []

    def node_to_chunk(node, symbol_type: str) -> Optional[dict]:
        # Find the name child
        name_node = node.child_by_field_name("name")
        name = source_bytes[name_node.start_byte:name_node.end_byte].decode() if name_node else "<anonymous>"
        start_line = node.start_point[0] + 1  # tree-sitter is 0-indexed
        end_line = node.end_point[0] + 1
        if (end_line - start_line + 1) < MIN_CHUNK_LINES:
            return None
        raw_code = "\n".join(lines[start_line - 1: end_line])
        docstring = _get_docstring_ts(node, source_bytes)
        return {
            "file_path": rel_path,
            "symbol_name": name,
            "symbol_type": symbol_type,
            "start_line": start_line,
            "end_line": end_line,
            "language": "python",
            "docstring": docstring,
            "raw_code": raw_code,
        }

    # Walk top-level nodes only (module body children)
    covered_lines: set[int] = set()
    for node in tree.root_node.children:
        if node.type in ("function_definition", "decorated_definition"):
            # For decorated_definition, find the actual function/class inside
            target = node
            if node.type == "decorated_definition":
                for child in node.children:
                    if child.type in ("function_definition", "class_definition"):
                        target = child
                        break
            stype = "function" if target.type == "function_definition" else "class"
            chunk = node_to_chunk(target, stype)
            if chunk:
                chunks.append(chunk)
                covered_lines.update(range(node.start_point[0], node.end_point[0] + 1))
        elif node.type == "class_definition":
            chunk = node_to_chunk(node, "class")
            if chunk:
                chunks.append(chunk)
                covered_lines.update(range(node.start_point[0], node.end_point[0] + 1))

    # Collect module-level code (lines not covered by any function/class)
    module_lines = [
        line for i, line in enumerate(lines) if i not in covered_lines
    ]
    module_code = "\n".join(module_lines).strip()
    if module_code and len(module_lines) >= MIN_CHUNK_LINES:
        chunks.append({
            "file_path": rel_path,
            "symbol_name": Path(rel_path).stem,
            "symbol_type": "module",
            "start_line": 1,
            "end_line": len(lines),
            "language": "python",
            "docstring": "",
            "raw_code": module_code,
        })

    return chunks


# ---------------------------------------------------------------------------
# AST chunking — JavaScript / TypeScript (tree-sitter)
# ---------------------------------------------------------------------------

def _chunk_js_ts(source: str, rel_path: str) -> list[dict]:
    """Extract top-level function and class declarations from JS/TS files."""
    source_bytes = source.encode("utf-8")
    tree = _js_parser.parse(source_bytes)
    lines = source.splitlines()
    lang = "typescript" if rel_path.endswith(".ts") else "javascript"
    chunks: list[dict] = []
    covered_lines: set[int] = set()

    JS_FUNC_TYPES = {
        "function_declaration",
        "generator_function_declaration",
        "arrow_function",
        "method_definition",
        "class_declaration",
        "lexical_declaration",  # const foo = () => ...
        "export_statement",
    }

    def extract(node) -> None:
        if node.type in JS_FUNC_TYPES:
            name_node = node.child_by_field_name("name")
            name = (
                source_bytes[name_node.start_byte:name_node.end_byte].decode()
                if name_node
                else "<anonymous>"
            )
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            if (end_line - start_line + 1) >= MIN_CHUNK_LINES:
                stype = "class" if "class" in node.type else "function"
                raw_code = "\n".join(lines[start_line - 1: end_line])
                chunks.append({
                    "file_path": rel_path,
                    "symbol_name": name,
                    "symbol_type": stype,
                    "start_line": start_line,
                    "end_line": end_line,
                    "language": lang,
                    "docstring": "",
                    "raw_code": raw_code,
                })
                covered_lines.update(range(node.start_point[0], node.end_point[0] + 1))
            return  # don't recurse into extracted chunks
        for child in node.children:
            extract(child)

    for child in tree.root_node.children:
        extract(child)

    # Module-level remainder
    module_lines = [l for i, l in enumerate(lines) if i not in covered_lines]
    module_code = "\n".join(module_lines).strip()
    if module_code and len(module_lines) >= MIN_CHUNK_LINES:
        chunks.append({
            "file_path": rel_path,
            "symbol_name": Path(rel_path).stem,
            "symbol_type": "module",
            "start_line": 1,
            "end_line": len(lines),
            "language": lang,
            "docstring": "",
            "raw_code": module_code,
        })
    return chunks


# ---------------------------------------------------------------------------
# Fallback: file-level chunk (no AST)
# ---------------------------------------------------------------------------

def _chunk_file_fallback(source: str, rel_path: str, language: str) -> list[dict]:
    """When Tree-sitter fails, index the entire file as one module chunk."""
    lines = source.splitlines()
    if len(lines) < MIN_CHUNK_LINES:
        return []
    return [{
        "file_path": rel_path,
        "symbol_name": Path(rel_path).stem,
        "symbol_type": "module",
        "start_line": 1,
        "end_line": len(lines),
        "language": language,
        "docstring": "",
        "raw_code": source,
    }]


# ---------------------------------------------------------------------------
# Dispatch chunker by language
# ---------------------------------------------------------------------------

_EXT_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".java": "java", ".go": "go", ".cpp": "cpp", ".c": "c",
    ".cs": "csharp", ".rb": "ruby", ".php": "php",
}


def chunk_file(path: Path, repo_root: str) -> list[dict]:
    rel_path = str(path.relative_to(repo_root)).replace("\\", "/")
    lang = _EXT_LANG.get(path.suffix, "unknown")
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    if not TREE_SITTER_AVAILABLE:
        return _chunk_file_fallback(source, rel_path, lang)

    try:
        if lang == "python":
            return _chunk_python_ts(source, rel_path)
        elif lang in ("javascript", "typescript"):
            return _chunk_js_ts(source, rel_path)
        else:
            return _chunk_file_fallback(source, rel_path, lang)
    except Exception as e:
        print(f"[WARN] Tree-sitter parse failed for {rel_path}: {e}")
        return _chunk_file_fallback(source, rel_path, lang)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    """
    Embed all chunks using models/gemini-embedding-2 in batches of EMBED_BATCH_SIZE.
    Sleeps between batches to avoid 429 rate limit errors from Google API.
    """
    embeddings: list[list[float]] = []
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i: i + EMBED_BATCH_SIZE]
        texts = [build_chunk_text(c) for c in batch]
        result = genai.embed_content(
            model="models/gemini-embedding-2",
            content=texts,
            task_type="retrieval_document",
        )
        embeddings.extend(result["embedding"])
        if i + EMBED_BATCH_SIZE < len(chunks):
            time.sleep(EMBED_SLEEP_SEC)
    return embeddings


# ---------------------------------------------------------------------------
# Pinecone upsert
# ---------------------------------------------------------------------------

def upsert_chunks(index, namespace: str, chunks: list[dict], embeddings: list[list[float]]) -> None:
    """Upsert vectors in batches of 50 (Pinecone's recommended batch size)."""
    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vec_id = f"{namespace}_{i}"
        metadata = {k: v for k, v in chunk.items() if k != "raw_code"}
        metadata["raw_code"] = chunk["raw_code"][:5000]  # Pinecone metadata cap
        vectors.append({"id": vec_id, "values": emb, "metadata": metadata})

    for i in range(0, len(vectors), 50):
        batch = vectors[i: i + 50]
        index.upsert(vectors=batch, namespace=namespace)


# ---------------------------------------------------------------------------
# Dependency graph builder
# ---------------------------------------------------------------------------

def _extract_python_deps(source: str, file_path: str, graph: nx.DiGraph) -> None:
    """Use stdlib ast to find imports and calls within a Python file."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return

    graph.add_node(file_path)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported = node.module.replace(".", "/") + ".py"
                graph.add_edge(file_path, imported, type="imports")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported = alias.name.replace(".", "/") + ".py"
                    graph.add_edge(file_path, imported, type="imports")
        elif isinstance(node, ast.FunctionDef):
            func_node = f"{file_path}::{node.name}"
            graph.add_node(func_node)
            graph.add_edge(file_path, func_node, type="defines")


def _extract_js_deps(source: str, file_path: str, graph: nx.DiGraph) -> None:
    """Regex-based import extraction for JS/TS files."""
    graph.add_node(file_path)
    # Match: import ... from '...' or require('...')
    patterns = [
        r"""from\s+['"]([^'"]+)['"]""",
        r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""",
    ]
    for pat in patterns:
        for match in re.finditer(pat, source):
            dep = match.group(1)
            # Only local imports (start with . or /)
            if dep.startswith(".") or dep.startswith("/"):
                graph.add_edge(file_path, dep, type="imports")


def build_dependency_graph(files: list[Path], repo_root: str) -> nx.DiGraph:
    graph = nx.DiGraph()
    for path in files:
        rel = str(path.relative_to(repo_root)).replace("\\", "/")
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if path.suffix == ".py":
            _extract_python_deps(source, rel, graph)
        elif path.suffix in (".js", ".ts"):
            _extract_js_deps(source, rel, graph)
        else:
            graph.add_node(rel)
    return graph


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def run_ingestion(repo_url: str, namespace: str, branch: Optional[str] = None) -> None:
    """
    Full pipeline: clone → parse → embed → upsert → graph.
    Designed to run as a FastAPI BackgroundTask (no return value).
    Updates indexing_state throughout.
    """
    clean_url = normalize_github_url(repo_url)
    tmp_dir = tempfile.mkdtemp()

    try:
        # ---- 1. Clone ----
        state.update_state(namespace, current_file="Cloning repository…")
        git_repo = GitRepo.clone_from(clean_url, tmp_dir)
        if branch:
            git_repo.git.checkout(branch)

        # ---- 2. Collect files ----
        files = collect_files(tmp_dir)
        state.update_state(namespace, total_files=len(files), current_file="Scanning files…")

        # ---- 3. Chunk all files ----
        all_chunks: list[dict] = []
        languages_seen: set[str] = set()

        for i, file_path in enumerate(files):
            state.update_state(
                namespace,
                processed_files=i + 1,
                current_file=str(file_path.relative_to(tmp_dir)),
            )
            chunks = chunk_file(file_path, tmp_dir)
            all_chunks.extend(chunks)
            if chunks:
                languages_seen.add(chunks[0]["language"])

        state.update_state(namespace, total_chunks=len(all_chunks))

        if not all_chunks:
            state.set_error(namespace, "No indexable chunks found in repository.")
            return

        # ---- 4. Embed + upsert in streaming fashion ----
        index = get_pinecone_index()
        embeddings = []

        for batch_start in range(0, len(all_chunks), EMBED_BATCH_SIZE):
            batch = all_chunks[batch_start: batch_start + EMBED_BATCH_SIZE]
            texts = [build_chunk_text(c) for c in batch]

            result = genai.embed_content(
                model="models/gemini-embedding-2",
                content=texts,
                task_type="retrieval_document",
            )
            batch_embeddings = result["embedding"]
            embeddings.extend(batch_embeddings)

            # Upsert this batch immediately
            vectors = []
            for j, (chunk, emb) in enumerate(zip(batch, batch_embeddings)):
                vec_id = f"{namespace}_{batch_start + j}"
                metadata = {k: v for k, v in chunk.items()}
                metadata["raw_code"] = chunk["raw_code"][:5000]
                vectors.append({"id": vec_id, "values": emb, "metadata": metadata})
            index.upsert(vectors=vectors, namespace=namespace)

            state.update_state(namespace, indexed_chunks=batch_start + len(batch))

            if batch_start + EMBED_BATCH_SIZE < len(all_chunks):
                time.sleep(EMBED_SLEEP_SEC)

        # ---- 5. Store in memory for BM25 ----
        bm25_corpus[namespace] = all_chunks

        # ---- 6. Dependency graph ----
        state.update_state(namespace, current_file="Building dependency graph…")
        graph = build_dependency_graph(files, tmp_dir)
        dependency_graph[namespace] = graph

        # ---- 7. Persist repo metadata ----
        save_indexed_repo(
            namespace=namespace,
            repo_url=repo_url,
            chunk_count=len(all_chunks),
            file_count=len(files),
            languages=sorted(languages_seen),
        )

        state.set_done(namespace)

    except Exception as e:
        state.set_error(namespace, str(e))
        raise

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
