"""
utils.py

Shared helpers used across the backend:
- GitHub URL normalization
- Namespace sanitization
- Token counting (word-based approximation)
- Chunk text builder (for embedding input)
- Indexed repos JSON persistence
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPOS_FILE = Path(__file__).parent / "indexed_repos.json"


# ---------------------------------------------------------------------------
# GitHub URL normalization
# ---------------------------------------------------------------------------

def normalize_github_url(url: str) -> str:
    """
    Accept any common GitHub URL format and return a clean https clone URL.

    Handles:
      - Trailing slashes and .git suffix
      - /tree/main or /tree/master suffixes (browse URLs)
      - SSH format: git@github.com:user/repo
      - Plain https://github.com/user/repo
    """
    url = url.strip().rstrip("/")

    # Convert SSH → HTTPS
    # git@github.com:user/repo  →  https://github.com/user/repo
    ssh_match = re.match(r"git@github\.com:(.+?)(?:\.git)?$", url)
    if ssh_match:
        return f"https://github.com/{ssh_match.group(1)}"

    # Strip .git suffix
    if url.endswith(".git"):
        url = url[:-4]

    # Strip /tree/<branch> or /blob/<branch>/... portions
    # e.g. https://github.com/user/repo/tree/main/subdir
    tree_match = re.match(r"(https://github\.com/[^/]+/[^/]+)(?:/tree/.+)?$", url)
    if tree_match:
        return tree_match.group(1)

    return url


def repo_url_to_namespace(repo_url: str) -> str:
    """
    Derive a stable Pinecone namespace from a repo URL.

    github.com/openai/whisper  →  openai-whisper
    Replaces '/', '.', '_' with '-' and lowercases everything.
    """
    url = normalize_github_url(repo_url)
    # Extract last two path segments: user/repo
    parts = url.rstrip("/").split("/")
    slug = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    namespace = re.sub(r"[/._]+", "-", slug).lower()
    # Truncate to Pinecone's 64-char namespace limit
    return namespace[:64]


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    Rough token estimate: 1 token ≈ 0.75 words (OpenAI / Google rule of thumb).
    Good enough for budget checks; no need for a full tokenizer here.
    """
    word_count = len(text.split())
    return int(word_count / 0.75)


def truncate_to_token_budget(chunks: list[dict], max_tokens: int = 6000) -> list[dict]:
    """
    Trim chunk list so total estimated tokens stay within budget.
    Truncates individual chunk text if a single chunk is very large.
    """
    total = 0
    result = []
    for chunk in chunks:
        text = chunk.get("raw_code", "")
        chunk_tokens = estimate_tokens(text)
        if total + chunk_tokens > max_tokens:
            remaining = max_tokens - total
            if remaining < 100:
                break
            # Truncate this chunk's code to fit
            words = text.split()
            allowed_words = int(remaining * 0.75)
            chunk = dict(chunk)
            chunk["raw_code"] = " ".join(words[:allowed_words]) + "\n# ... (truncated)"
            result.append(chunk)
            break
        result.append(chunk)
        total += chunk_tokens
    return result


# ---------------------------------------------------------------------------
# Chunk text builder
# ---------------------------------------------------------------------------

def build_chunk_text(chunk: dict) -> str:
    """
    Combine metadata and code into a single embedding-ready string.
    Docstring is prepended so semantic meaning is captured even for short functions.
    """
    parts = []
    if chunk.get("docstring"):
        parts.append(f"# {chunk['docstring']}")
    parts.append(
        f"# {chunk['symbol_type']}: {chunk['symbol_name']} "
        f"({chunk['file_path']} L{chunk['start_line']}-{chunk['end_line']})"
    )
    parts.append(chunk.get("raw_code", ""))
    return "\n".join(parts)


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into the LLM context block with citations."""
    blocks = []
    for c in chunks:
        header = (
            f"### {c['symbol_type'].upper()}: `{c['symbol_name']}`\n"
            f"**File:** `{c['file_path']}` (lines {c['start_line']}–{c['end_line']})"
        )
        if c.get("docstring"):
            header += f"\n**Docstring:** {c['docstring']}"
        code_lang = c.get("language", "python")
        block = f"{header}\n```{code_lang}\n{c['raw_code']}\n```"
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Indexed repos JSON store
# ---------------------------------------------------------------------------

def load_indexed_repos() -> list[dict]:
    if not REPOS_FILE.exists():
        return []
    try:
        return json.loads(REPOS_FILE.read_text())
    except Exception:
        return []


def save_indexed_repo(
    namespace: str,
    repo_url: str,
    chunk_count: int,
    file_count: int,
    languages: list[str],
) -> None:
    repos = load_indexed_repos()
    # Remove old entry for this namespace if re-indexing
    repos = [r for r in repos if r["namespace"] != namespace]
    repos.append(
        {
            "namespace": namespace,
            "repo_url": repo_url,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": chunk_count,
            "file_count": file_count,
            "languages": languages,
        }
    )
    REPOS_FILE.write_text(json.dumps(repos, indent=2))


def delete_indexed_repo(namespace: str) -> None:
    repos = load_indexed_repos()
    repos = [r for r in repos if r["namespace"] != namespace]
    REPOS_FILE.write_text(json.dumps(repos, indent=2))


def get_repo_info(namespace: str) -> Optional[dict]:
    return next(
        (r for r in load_indexed_repos() if r["namespace"] == namespace), None
    )


# ---------------------------------------------------------------------------
# camelCase / snake_case tokenizer (for BM25)
# ---------------------------------------------------------------------------

_SPLIT_RE = re.compile(
    r"[^a-zA-Z0-9]+"             # non-alphanumeric separators
    r"|(?<=[a-z])(?=[A-Z])"      # camelCase boundary
    r"|(?<=[A-Z])(?=[A-Z][a-z])" # ABCDef → ABC + Def
)


def tokenize_code(text: str) -> list[str]:
    """Split code text on whitespace, punctuation, and case boundaries."""
    tokens = _SPLIT_RE.split(text)
    return [t.lower() for t in tokens if len(t) > 1]
