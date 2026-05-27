"""
graph.py

Change impact analysis using the NetworkX dependency graph built during ingestion.

Algorithm:
1. Find the target node (file or symbol) in the directed graph
2. Reverse the graph so edges point from dependents TO dependencies
3. BFS from the target to collect all transitively affected nodes
4. For each affected node, retrieve its code chunk from Pinecone
5. Ask Gemini to rate the risk of breaking that dependent

Risk ratings: Low / Medium / High
"""

import os
from typing import Optional

import google.generativeai as genai
import networkx as nx
from dotenv import load_dotenv

from retrieval import vector_search

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))


def _get_graph(namespace: str) -> Optional[nx.DiGraph]:
    """Retrieve the dependency graph for this namespace from ingestion memory."""
    from ingestion import dependency_graph
    return dependency_graph.get(namespace)


def _find_node(graph: nx.DiGraph, target: str) -> Optional[str]:
    """
    Find a node by exact match or partial substring.
    Supports both file paths (auth.py) and qualified symbols (auth.py::login).
    """
    # Exact match
    if target in graph.nodes:
        return target
    # Substring match (e.g., user types "login" and we have "src/auth.py::login")
    matches = [n for n in graph.nodes if target in n]
    return matches[0] if matches else None


def _get_affected_nodes(graph: nx.DiGraph, source_node: str) -> list[str]:
    """
    Reverse BFS: find all nodes that (transitively) depend on source_node.
    We reverse the graph so that dependents are reachable via standard BFS.
    """
    reversed_graph = graph.reverse(copy=False)
    # BFS from source_node in reversed graph gives us all dependents
    affected = set(nx.bfs_tree(reversed_graph, source_node).nodes)
    affected.discard(source_node)  # exclude the target itself
    return list(affected)


def _assess_risk(
    target: str,
    dependent_node: str,
    target_code: str,
    dependent_code: str,
) -> dict:
    """
    Ask Gemini to rate the risk level of a specific dependent being broken.
    Returns {"symbol": ..., "file": ..., "risk": ..., "reasoning": ...}
    """
    prompt = (
        f"You are a senior software engineer performing change impact analysis.\n\n"
        f"MODIFIED: `{target}`\n"
        f"```\n{target_code[:1500]}\n```\n\n"
        f"DEPENDENT: `{dependent_node}`\n"
        f"```\n{dependent_code[:1500]}\n```\n\n"
        f"Given that `{target}` was modified, assess the risk to `{dependent_node}` "
        f"which calls or imports it.\n"
        f"Respond in this exact format:\n"
        f"RISK: <Low|Medium|High>\n"
        f"REASON: <one sentence explanation>\n"
    )

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    text = response.text.strip()

    risk = "Medium"
    reason = "Unable to determine risk automatically."

    for line in text.splitlines():
        if line.startswith("RISK:"):
            risk_val = line.split(":", 1)[1].strip()
            if risk_val in ("Low", "Medium", "High"):
                risk = risk_val
        elif line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    # Parse file path vs symbol name from qualified node (file.py::symbol)
    if "::" in dependent_node:
        file_part, sym_part = dependent_node.split("::", 1)
    else:
        file_part = dependent_node
        sym_part = dependent_node

    return {
        "symbol": sym_part,
        "file": file_part,
        "risk": risk,
        "reasoning": reason,
    }


def analyze_impact(symbol: str, namespace: str) -> dict:
    """
    Full change impact analysis for a symbol or file.

    Returns:
    {
        "target": "authenticate_user",
        "affected": [
            {"symbol": "login_route", "file": "routes/auth.py",
             "risk": "High", "reasoning": "..."}
        ],
        "summary": "Modifying this affects N files..."
    }
    """
    graph = _get_graph(namespace)
    if graph is None:
        return {
            "target": symbol,
            "affected": [],
            "summary": (
                "Dependency graph not available. "
                "Re-index the repository to enable change impact analysis."
            ),
        }

    node = _find_node(graph, symbol)
    if node is None:
        return {
            "target": symbol,
            "affected": [],
            "summary": f"Symbol `{symbol}` not found in dependency graph.",
        }

    affected_nodes = _get_affected_nodes(graph, node)

    if not affected_nodes:
        return {
            "target": symbol,
            "affected": [],
            "summary": f"`{symbol}` has no dependents — changes here are isolated.",
        }

    # Retrieve target's code from Pinecone
    target_results = vector_search(symbol, namespace, top_k=1)
    target_code = target_results[0][0].get("raw_code", "") if target_results else ""

    # For each dependent, retrieve code and assess risk
    # Cap at 10 dependents to avoid excessive Gemini calls
    assessed: list[dict] = []
    for dep_node in affected_nodes[:10]:
        dep_results = vector_search(dep_node, namespace, top_k=1)
        dep_code = dep_results[0][0].get("raw_code", "") if dep_results else ""
        assessment = _assess_risk(node, dep_node, target_code, dep_code)
        assessed.append(assessment)

    # Sort by risk severity
    risk_order = {"High": 0, "Medium": 1, "Low": 2}
    assessed.sort(key=lambda x: risk_order.get(x["risk"], 1))

    high_count = sum(1 for a in assessed if a["risk"] == "High")
    total = len(affected_nodes)
    capped = total > 10

    summary = (
        f"Modifying `{symbol}` affects {total} dependent(s)"
        f"{' (showing top 10)' if capped else ''}. "
        f"{high_count} are rated High risk."
    )

    return {
        "target": symbol,
        "affected": assessed,
        "summary": summary,
    }
