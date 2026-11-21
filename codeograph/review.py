"""Pull-request review grounded in the repository.

The diff is mapped onto the index: every changed line range is matched to the functions/methods it touches (exact
line overlap, not a regex on hunk headers), each touched symbol's dependents come from the graph, and the LLM
reviews the diff with that context. Without an LLM the review is the deterministic part: touched symbols, blast
radius and a graph-based risk score - which an MCP client's own model can then reason over.
"""

from __future__ import annotations

import re

from . import impact, llm, store

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_diff(diff: str) -> list[dict]:
    """[{file, added: [(start,end)], removed_lines, added_lines, new_file, deleted}] from a unified diff."""
    files: list[dict] = []
    cur: dict | None = None
    new_line = 0
    for line in diff.splitlines():
        if line.startswith("diff --git"):
            cur = None
            continue
        if line.startswith("+++ "):
            path = line[4:].strip()
            path = path[2:] if path.startswith("b/") else path
            cur = {"file": path, "ranges": [], "added_lines": 0, "removed_lines": 0,
                   "new_file": False, "deleted": path == "/dev/null"}
            files.append(cur)
            continue
        if line.startswith("--- "):
            prev = line[4:].strip()
            if cur is None and prev == "/dev/null":
                pass
            continue
        m = _HUNK.match(line)
        if m and cur is not None:
            new_line = int(m.group(3))
            count = int(m.group(4) or 1)
            old_count = int(m.group(2) or 1)
            cur["ranges"].append((new_line, new_line + max(count, 1) - 1))
            if int(m.group(1)) == 0 and old_count == 0:
                cur["new_file"] = True
            continue
        if cur is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            cur["added_lines"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            cur["removed_lines"] += 1
    return files


def touched_symbols(index: store.RepoIndex, files: list[dict]) -> list[str]:
    nodes = []
    for f in files:
        for c in index.chunks:
            if c.file_path != f["file"] or c.symbol_type in ("module", "block", "class"):
                continue
            if any(s <= c.end_line and e >= c.start_line for s, e in f["ranges"]):
                nodes.append(f"{c.file_path}::{c.qualified_name}")
    return sorted(set(nodes))


def _code(index: store.RepoIndex, node: str) -> str:
    f, _, q = node.partition("::")
    return next((c.code for c in index.chunks if c.file_path == f and c.qualified_name == q), "")


def review(index: store.RepoIndex, diff: str) -> dict:
    files = parse_diff(diff)
    symbols = touched_symbols(index, files)
    blast: dict[str, dict] = {}
    for node in symbols[:12]:
        rep = impact.analyze(index, node, max_hops=2, use_llm=False)
        for a in rep["affected"]:
            if a["node"] not in symbols:
                blast.setdefault(a["node"], a)
    direct = [a for a in blast.values() if a["hops"] == 1]
    graph_risk = "High" if len(direct) >= 5 else "Medium" if direct else "Low"
    base = {
        "files": [{"file": f["file"], "added_lines": f["added_lines"], "removed_lines": f["removed_lines"]}
                  for f in files],
        "affected_symbols": [s.split("::", 1)[1] for s in symbols],
        "dependents": sorted(blast.values(), key=lambda a: (a["hops"], a["node"]))[:25],
        "graph_risk": graph_risk,
    }
    model = llm.get_model()
    if model is None:
        return {**base, "summary": f"Touches {len(symbols)} symbol(s) in {len(files)} file(s); "
                                   f"{len(direct)} direct dependent(s) outside the diff.",
                "risks": "\n".join(f"- `{a['node']}` {a['reasoning']}" for a in direct[:10]) or "- None found",
                "suggested_tests": "\n".join(f"- Cover `{s.split('::', 1)[1]}`" for s in symbols[:8]) or "- n/a",
                "risk_score": graph_risk, "risk_justification": "From the dependency graph (no LLM configured).",
                "raw_review": "", "model": "none"}

    ctx_syms = "\n\n".join(f"`{s}` (current code)\n```\n{_code(index, s)[:1500]}\n```" for s in symbols[:6])
    ctx_deps = "\n\n".join(f"`{a['node']}` - {a['reasoning']}\n```\n{_code(index, a['node'])[:800]}\n```"
                           for a in direct[:6])
    verdict = llm.ask_json(
        "You are a senior engineer reviewing a pull request against the repository it targets. Be specific, "
        "cite function names in backticks, and do not invent code that is not shown. Reply with JSON only.",
        f"DIFF:\n```diff\n{diff[:8000]}\n```\n\nCHANGED SYMBOLS (before the change):\n{ctx_syms or 'none matched'}"
        f"\n\nDIRECT DEPENDENTS OUTSIDE THE DIFF:\n{ctx_deps or 'none'}\n\n"
        'Reply as {"summary": str, "risks": [str], "suggested_tests": [str], '
        '"risk_score": "High|Medium|Low", "risk_justification": str}',
        default={}, model=model)
    if not isinstance(verdict, dict) or not verdict:
        verdict = {"summary": "The model did not return a parseable review.", "risks": [], "suggested_tests": [],
                   "risk_score": graph_risk, "risk_justification": "Fell back to the dependency-graph risk."}

    def bullets(v) -> str:
        return "\n".join(f"- {x}" for x in v) if isinstance(v, list) else str(v or "")

    risk = verdict.get("risk_score") if verdict.get("risk_score") in ("High", "Medium", "Low") else graph_risk
    return {**base, "summary": str(verdict.get("summary", "")), "risks": bullets(verdict.get("risks")),
            "suggested_tests": bullets(verdict.get("suggested_tests")), "risk_score": risk,
            "risk_justification": str(verdict.get("risk_justification", "")), "raw_review": str(verdict),
            "model": llm.model_label()}


def to_markdown(r: dict, diff: str) -> str:
    deps = "\n".join(f"- `{a['node']}` ({a['risk']}) - {a['reasoning']}" for a in r.get("dependents", [])[:15])
    return (f"# Pull request review\n\n**Risk: {r['risk_score']}** - {r.get('risk_justification', '')}\n\n"
            f"## Summary\n{r['summary']}\n\n## Risks\n{r['risks']}\n\n## Suggested tests\n{r['suggested_tests']}\n\n"
            f"## Changed symbols\n" + "\n".join(f"- `{s}`" for s in r.get("affected_symbols", [])) +
            f"\n\n## Dependents outside the diff\n{deps or '- none'}\n\n<details><summary>Diff</summary>\n\n"
            f"```diff\n{diff[:20000]}\n```\n</details>\n")
