"""
diagram.py

Generates a Mermaid architecture diagram from the NetworkX dependency graph.
Only file-level import relationships are shown (not individual symbols) to keep
the diagram readable. Isolated files (no edges) are filtered out.
"""

import networkx as nx
from pathlib import Path
from typing import Optional


def _get_graph(namespace: str) -> Optional[nx.DiGraph]:
    from ingestion import dependency_graph
    return dependency_graph.get(namespace)


def _safe_node_id(path: str) -> str:
    """Convert a file path to a valid Mermaid node ID (no slashes or dots)."""
    return path.replace("/", "_").replace(".", "_").replace("-", "_")


def _short_label(path: str) -> str:
    """Show just the filename for cleaner diagram labels."""
    return Path(path).name


def generate_diagram(namespace: str) -> dict:
    """
    Build Mermaid graph TD syntax and a flat dependency table.

    Returns:
        {
            "mermaid": "graph TD\n    ...",
            "dependency_table": [
                {"file": "auth.py", "imports": ["utils.py"], "imported_by": ["main.py"]}
            ]
        }
    """
    graph = _get_graph(namespace)
    if graph is None:
        return {
            "mermaid": "graph TD\n    A[\"No dependency graph — re-index repo\"]",
            "dependency_table": [],
        }

    # Keep only file-level nodes (no :: symbol nodes)
    file_nodes = [n for n in graph.nodes if "::" not in n]
    # Build a file-only subgraph
    file_graph = nx.DiGraph()
    for node in file_nodes:
        file_graph.add_node(node)
    for u, v, data in graph.edges(data=True):
        if "::" not in u and "::" not in v and u != v:
            file_graph.add_edge(u, v, **data)

    # Filter to connected nodes only
    connected = {n for n in file_graph.nodes if file_graph.degree(n) > 0}
    if not connected:
        return {
            "mermaid": "graph TD\n    A[\"No file-level imports detected\"]",
            "dependency_table": [],
        }

    # Limit to 60 nodes max to keep Mermaid renderable
    if len(connected) > 60:
        # Pick the 60 most connected nodes
        by_degree = sorted(connected, key=lambda n: file_graph.degree(n), reverse=True)
        connected = set(by_degree[:60])

    lines = ["graph TD"]
    seen_edges: set[tuple] = set()

    for node in sorted(connected):
        nid = _safe_node_id(node)
        label = _short_label(node)
        for successor in file_graph.successors(node):
            if successor not in connected:
                continue
            sid = _safe_node_id(successor)
            slabel = _short_label(successor)
            edge = (nid, sid)
            if edge not in seen_edges:
                lines.append(f'    {nid}["{label}"] --> {sid}["{slabel}"]')
                seen_edges.add(edge)

    mermaid = "\n".join(lines)

    # Build flat dependency table
    table = []
    for node in sorted(connected):
        imports = [
            _short_label(s) for s in file_graph.successors(node) if s in connected
        ]
        imported_by = [
            _short_label(p) for p in file_graph.predecessors(node) if p in connected
        ]
        if imports or imported_by:
            table.append({
                "file": _short_label(node),
                "full_path": node,
                "imports": imports,
                "imported_by": imported_by,
            })

    return {"mermaid": mermaid, "dependency_table": table}
