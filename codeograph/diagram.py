"""Architecture diagram: the file-level import graph as Mermaid, grouped by folder."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import PurePosixPath

from . import store

MAX_FILES = 40


def _id(path: str) -> str:
    return "f_" + re.sub(r"\W", "_", path)


def generate(index: store.RepoIndex, max_files: int = MAX_FILES) -> dict:
    g = index.graph
    edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("type") == "imports" and u != v]
    if not edges:
        return {"mermaid": 'graph TD\n    A["No imports between files were detected"]', "dependency_table": [],
                "files_shown": 0}
    degree = Counter()
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    keep = {f for f, _ in degree.most_common(max_files)}
    edges = [(u, v) for u, v in edges if u in keep and v in keep]
    groups: dict[str, list[str]] = {}
    for f in sorted(keep):
        groups.setdefault(str(PurePosixPath(f).parent), []).append(f)
    lines = ["graph LR"]
    for folder, members in sorted(groups.items()):
        label = folder if folder != "." else "(root)"
        lines.append(f'    subgraph {_id(folder)}["{label}"]')
        lines += [f'        {_id(f)}["{PurePosixPath(f).name}"]' for f in members]
        lines.append("    end")
    lines += [f"    {_id(u)} --> {_id(v)}" for u, v in sorted(set(edges))]
    table = []
    for f in sorted(keep):
        table.append({"file": PurePosixPath(f).name, "full_path": f,
                      "imports": sorted(v for u, v in edges if u == f),
                      "imported_by": sorted(u for u, v in edges if v == f)})
    return {"mermaid": "\n".join(lines), "dependency_table": table, "files_shown": len(keep)}
