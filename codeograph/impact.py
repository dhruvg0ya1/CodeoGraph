"""Change impact: what depends on a symbol or file, how directly, and how risky a change is.

The walk is deterministic (reverse BFS over call and import edges, nearest first). Risk comes from the graph:
a direct caller is High, a direct importer Medium, anything two or more hops away Low. When an LLM is configured,
it may re-rate the direct dependents after reading both pieces of code, in one batched call.
"""

from __future__ import annotations

from . import depgraph, llm, store

MAX_REPORTED = 25


def _risk(hops: int, via: str) -> str:
    if hops == 1:
        return "High" if via == "calls" else "Medium"
    return "Low"


def _why(node: str, hops: int, via: str, target: str) -> str:
    if hops == 1:
        return f"calls `{target}` directly" if via == "calls" else f"imports the module that defines `{target}`"
    return f"{hops} hops away (through {'calls' if via == 'calls' else 'imports'})"


def _code_for(index: store.RepoIndex, node: str) -> str:
    if "::" not in node:
        return ""
    f, q = node.split("::", 1)
    for c in index.chunks:
        if c.file_path == f and c.qualified_name == q:
            return c.code
    return ""


def analyze(index: store.RepoIndex, target: str, max_hops: int = 3, use_llm: bool = True) -> dict:
    nodes = depgraph.find_nodes(index.graph, target)
    if not nodes:
        return {"target": target, "matched": [], "affected": [],
                "summary": f"`{target}` was not found in the dependency graph."}
    deps = depgraph.dependents(index.graph, nodes, max_hops=max_hops)
    affected = []
    for node, hops, via in deps:
        file_part, _, sym = node.partition("::")
        affected.append({"symbol": sym or file_part, "file": file_part, "node": node, "hops": hops, "via": via,
                         "risk": _risk(hops, via), "reasoning": _why(node, hops, via, target)})

    model = llm.get_model() if use_llm else None
    direct = [a for a in affected if a["hops"] == 1][:10]
    if model is not None and direct:
        target_code = "\n\n".join(_code_for(index, n)[:1500] for n in nodes[:2])
        listing = "\n\n".join(f"[{i}] `{a['node']}` ({a['via']})\n```\n{_code_for(index, a['node'])[:1200]}\n```"
                              for i, a in enumerate(direct))
        verdict = llm.ask_json(
            "You assess change impact in a codebase. Reply with JSON only.",
            f"The code being changed:\n```\n{target_code}\n```\n\nIts direct dependents:\n{listing}\n\n"
            'For each dependent give {"i": <index>, "risk": "High|Medium|Low", "reason": "<one sentence>"} '
            "judging how likely a change to the target's behaviour or signature breaks it. "
            'Reply as {"ratings": [...]}.', default={"ratings": []}, model=model)
        for r in verdict.get("ratings", []) if isinstance(verdict, dict) else []:
            try:
                a = direct[int(r["i"])]
            except (KeyError, ValueError, IndexError, TypeError):
                continue
            if r.get("risk") in ("High", "Medium", "Low"):
                a["risk"], a["reasoning"] = r["risk"], str(r.get("reason") or a["reasoning"])
                a["rated_by"] = "llm"

    order = {"High": 0, "Medium": 1, "Low": 2}
    affected.sort(key=lambda a: (a["hops"], order[a["risk"]], a["node"]))
    files = {a["file"] for a in affected}
    high = sum(a["risk"] == "High" for a in affected)
    shown = affected[:MAX_REPORTED]
    summary = (f"Changing `{target}` reaches {len(affected)} dependent(s) in {len(files)} file(s) within "
               f"{max_hops} hops; {high} rated High."
               + (f" Showing the nearest {MAX_REPORTED}." if len(affected) > MAX_REPORTED else "")) \
        if affected else f"Nothing in the repository depends on `{target}` - a change here is isolated."
    return {"target": target, "matched": nodes, "affected": shown, "total_affected": len(affected),
            "summary": summary}
