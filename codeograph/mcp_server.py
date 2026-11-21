"""CodeoGraph as an MCP server: give Claude Desktop, Claude Code, Cursor or any MCP client grounded tools over any
repository - semantic + keyword code search, go-to-definition, dependency-aware change impact, diff review context
and architecture diagrams. None of these tools needs an API key: the host's own model does the reasoning over
what CodeoGraph retrieves. `ask` runs CodeoGraph's own LangGraph agent when an LLM is configured.

    codeograph mcp                                  # stdio
    claude mcp add codeograph -- codeograph mcp     # Claude Code
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from . import agent, diagram, impact, indexer, review, search, store

INSTRUCTIONS = (
    "CodeoGraph indexes git repositories (or local folders) and answers from their real code. Index a repository "
    "once with index_repository, then use search_code / find_symbol / get_file to read code, impact_analysis "
    "before changing a symbol, and review_diff to see what a change touches. Every result carries file paths and "
    "line numbers - cite them."
)

server = MCPServer("CodeoGraph", instructions=INSTRUCTIONS)


def _index(repo: str) -> store.RepoIndex:
    ns = repo if store.load(repo) else indexer.namespace_for(repo)
    idx = store.load(ns)
    if idx is None:
        raise ValueError(f"'{repo}' is not indexed yet - call index_repository first.")
    return idx


@server.tool()
def index_repository(repo: str, branch: str | None = None, force: bool = False) -> dict:
    """Index a git repository URL (https://github.com/owner/name) or a local folder path. Unchanged commits are
    skipped and only changed code is re-embedded. Returns the namespace to pass as `repo` to the other tools."""
    meta = indexer.index_repository(repo, branch=branch, force=force)
    return {k: meta.get(k) for k in ("status", "namespace", "commit", "file_count", "chunk_count", "languages",
                                      "graph", "embedded", "reused", "seconds")}


@server.tool()
def list_repositories() -> list[dict]:
    """Repositories already indexed, with their namespace, commit and size."""
    return store.list_indexes()


@server.tool()
def search_code(repo: str, query: str, k: int = 6, file: str | None = None, kind: str | None = None) -> list[dict]:
    """Hybrid (BM25 + dense, reranked) search over a repository's functions, methods, classes and modules.
    `kind` filters to function | method | class | module | type; `file` to one repo-relative path."""
    hits = search.search(_index(repo), query, k=max(1, min(k, 20)), file=file, kind=kind)
    return [h.to_dict(max_code=2500) for h in hits]


@server.tool()
def find_symbol(repo: str, name: str) -> list[dict]:
    """Go to definition: where a function, method (`Class.method`) or class is defined, with its code."""
    return [search.Hit(c, 1.0).to_dict(max_code=4000) for c in search.find_symbol(_index(repo), name)[:5]]


@server.tool()
def get_file(repo: str, path: str, start_line: int = 1, end_line: int | None = None) -> dict:
    """Read a source file (or a line range) from the indexed snapshot of the repository."""
    idx = _index(repo)
    text = idx.files.get(path)
    if text is None:
        close = [p for p in idx.files if p.endswith(path)][:5]
        raise ValueError(f"No file '{path}' in {idx.namespace}." + (f" Did you mean: {close}?" if close else ""))
    lines = text.splitlines()
    end = min(end_line or len(lines), len(lines), start_line + 799)
    body = "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(max(1, start_line), end + 1))
    return {"path": path, "start_line": start_line, "end_line": end, "total_lines": len(lines), "content": body}


@server.tool()
def impact_analysis(repo: str, target: str, max_hops: int = 3) -> dict:
    """What depends on a symbol or file: callers and importers found by walking the dependency graph backwards,
    nearest first, each with hop distance and a risk rating."""
    return impact.analyze(_index(repo), target, max_hops=max(1, min(max_hops, 5)), use_llm=False)


@server.tool()
def review_diff(repo: str, diff: str) -> dict:
    """Map a unified diff onto the repository: the functions it changes, their dependents outside the diff and a
    graph-based risk score (plus an LLM review when CodeoGraph has a model configured)."""
    return review.review(_index(repo), diff)


@server.tool()
def architecture_diagram(repo: str) -> dict:
    """A Mermaid diagram of how the repository's files import each other, grouped by folder."""
    return diagram.generate(_index(repo))


@server.tool()
def ask(repo: str, question: str, thread_id: str | None = None) -> dict:
    """Ask CodeoGraph's own agent (corrective RAG over the repository, answers cite sources). Needs an LLM
    configured for CodeoGraph; otherwise it returns the most relevant code locations."""
    return agent.ask(question, _index(repo).namespace, thread_id=thread_id)


@server.prompt()
def review_pull_request(repo: str, diff: str) -> str:
    """A ready-made prompt for reviewing a diff with CodeoGraph's tools."""
    return (f"Review this change to the repository `{repo}`. First call review_diff, then read the changed "
            f"symbols with find_symbol and the riskiest dependents with get_file. Report concrete risks with file "
            f"paths and line numbers, and the tests that should be added.\n\n```diff\n{diff}\n```")


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
