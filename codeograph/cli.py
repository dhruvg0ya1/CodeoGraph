"""Command line.

    codeograph index https://github.com/psf/requests        # or a local folder
    codeograph search psf-requests "how are redirects followed"
    codeograph symbol psf-requests Session.send
    codeograph impact psf-requests merge_environment_settings
    codeograph ask psf-requests "where does the timeout get applied?"
    codeograph review psf-requests change.diff
    codeograph diagram psf-requests
    codeograph repos | rm <ns>
    codeograph serve [--port 8000]      REST API
    codeograph mcp                      MCP server over stdio
"""

from __future__ import annotations

import argparse
import json
import sys


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _index(ns: str):
    from . import indexer, store

    idx = store.load(ns) or store.load(indexer.namespace_for(ns))
    if idx is None:
        sys.exit(f"'{ns}' is not indexed - run: codeograph index <url-or-path>")
    return idx


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="codeograph", description="Grounded answers about any codebase.")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("index", help="index a git URL or a local folder")
    s.add_argument("source")
    s.add_argument("--branch")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("search", help="hybrid code search")
    s.add_argument("repo")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default="hybrid")
    s = sub.add_parser("symbol", help="go to definition")
    s.add_argument("repo")
    s.add_argument("name")
    s = sub.add_parser("impact", help="what depends on a symbol or file")
    s.add_argument("repo")
    s.add_argument("target")
    s.add_argument("--hops", type=int, default=3)
    s = sub.add_parser("ask", help="ask the agent")
    s.add_argument("repo")
    s.add_argument("question")
    s = sub.add_parser("review", help="review a unified diff file (- for stdin)")
    s.add_argument("repo")
    s.add_argument("diff")
    s = sub.add_parser("diagram", help="Mermaid import diagram")
    s.add_argument("repo")
    sub.add_parser("repos", help="list indexed repositories")
    s = sub.add_parser("rm", help="delete an index")
    s.add_argument("repo")
    s = sub.add_parser("serve", help="run the REST API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    sub.add_parser("mcp", help="run the MCP server (stdio)")
    a = p.parse_args(argv)

    if a.cmd == "index":
        from . import indexer

        def say(**kw):
            if "current_file" in kw and kw.get("stage") in ("cloning", "scanning", "embedding", "graph", "done"):
                print(f"  {kw['current_file']}", file=sys.stderr)
        meta = indexer.index_repository(a.source, branch=a.branch, force=a.force, progress=say)
        meta.pop("file_hashes", None)
        _print(meta)
    elif a.cmd == "search":
        from . import search

        for h in search.search(_index(a.repo), a.query, k=a.k, mode=a.mode):
            c = h.chunk
            print(f"{h.score:8.3f}  {c.file_path}:{c.start_line}-{c.end_line}  {c.qualified_name} ({c.symbol_type})")
    elif a.cmd == "symbol":
        from . import search

        for c in search.find_symbol(_index(a.repo), a.name):
            print(f"{c.file_path}:{c.start_line}-{c.end_line}  {c.qualified_name}\n{c.code}\n")
    elif a.cmd == "impact":
        from . import impact

        _print(impact.analyze(_index(a.repo), a.target, max_hops=a.hops))
    elif a.cmd == "ask":
        from . import agent

        r = agent.ask(a.question, _index(a.repo).namespace)
        print(r["answer"])
        for c in r["citations"]:
            print(f"  [{c['n']}] {c['file_path']}:{c['start_line']}-{c['end_line']} {c['symbol_name']}")
    elif a.cmd == "review":
        from . import review

        diff = sys.stdin.read() if a.diff == "-" else open(a.diff).read()
        r = review.review(_index(a.repo), diff)
        print(review.to_markdown(r, diff))
    elif a.cmd == "diagram":
        from . import diagram

        print(diagram.generate(_index(a.repo))["mermaid"])
    elif a.cmd == "repos":
        from . import store

        _print(store.list_indexes())
    elif a.cmd == "rm":
        from . import store

        print("deleted" if store.delete(a.repo) else "not found")
    elif a.cmd == "serve":
        import uvicorn

        uvicorn.run("codeograph.api:app", host=a.host, port=a.port)
    elif a.cmd == "mcp":
        from .mcp_server import main as mcp_main

        mcp_main()


if __name__ == "__main__":
    main()
