"""The question-answering agent: a LangGraph state machine doing corrective, cited RAG over one repository.

    START -> classify -> retrieve -> grade --relevant--> generate -> verify -> END
                            ^            \\--none relevant, attempts left--> rewrite --+
                            +-----------------------------------------------------------+

  classify  rules, no LLM: a symbol lookup ("where is `send` defined"), an impact question ("what breaks if I
            change X") or a general question; pulls the code identifier out of the question.
  retrieve  symbol -> go-to-definition + hybrid search; impact -> dependency-graph walk + the dependents' code;
            question -> hybrid search (BM25 + dense, fused, reranked).
  grade     the LLM keeps only the sources that help answer; none useful -> rewrite the query (max 2 retries).
  generate  answers ONLY from numbered sources and cites them as [n].
  verify    deterministic: every [n] must point at a retrieved source; invalid ones are removed and reported.

Conversation memory is a LangGraph checkpointer keyed by thread_id, so follow-ups ("and who calls it?") work.
With no LLM configured the same graph runs and answers extractively (the best sources, their signatures and docs).
"""

from __future__ import annotations

import operator
import re
import uuid
from collections.abc import Iterator
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from . import impact as impact_mod
from . import llm, search, store

MAX_ATTEMPTS = 2
TOP_K = 8


class State(TypedDict, total=False):
    question: str
    namespace: str
    filter_file: str
    filter_kind: str
    history: Annotated[list[dict], operator.add]
    intent: str
    target: str
    query: str
    attempts: int
    hits: list[dict]
    relevant: list[int]
    impact: dict
    answer: str
    citations: list[dict]
    invalid_citations: list[int]
    trace: Annotated[list[str], operator.add]


_IMPACT = re.compile(r"\b(impact|break|breaks|affect|affects|affected|depend|depends|dependents?|who calls|callers?"
                     r"|called by|used by|uses of|usages?|ripple)\b", re.I)
_SYMBOL = re.compile(r"\b(where is|where's|defined|definition|find|locate|show me|go to)\b", re.I)


def _identifiers(text: str, index: store.RepoIndex | None) -> list[str]:
    found = []
    for m in re.finditer(r"`([^`]+)`|([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)|"
                         r"([A-Za-z_]*[a-z][A-Z][A-Za-z0-9_]*)|([A-Za-z0-9]*_[A-Za-z0-9_]+)|([A-Za-z_]\w*)\(\)", text):
        tok = next(g for g in m.groups() if g)
        found.append(tok.strip().removesuffix("()"))
    if not found and index is not None:                     # plain words that are symbol names in this repo
        names = {c.symbol_name for c in index.chunks if c.symbol_type not in ("module", "block")}
        found = [w for w in re.findall(r"[A-Za-z_]\w{3,}", text) if w in names]
    return found


def _index(state: State) -> store.RepoIndex:
    idx = store.load(state["namespace"])
    if idx is None:
        raise LookupError(f"Repository '{state['namespace']}' is not indexed.")
    return idx


# --------------------------------------------------------------------------- nodes
def classify(state: State) -> dict:
    q = state["question"]
    idents = _identifiers(q, store.load(state["namespace"]))
    intent = "question"
    if idents and _IMPACT.search(q):
        intent = "impact"
    elif idents and _SYMBOL.search(q):
        intent = "symbol"
    return {"intent": intent, "target": idents[0] if idents else "", "query": q, "attempts": 0,
            "trace": [f"classify: {intent}" + (f" ({idents[0]})" if idents else "")]}


def retrieve(state: State) -> dict:
    idx = _index(state)
    hits: list[search.Hit] = []
    extra: dict[str, Any] = {}
    if state.get("intent") == "symbol" and state.get("target"):
        hits = [search.Hit(c, 1.0) for c in search.find_symbol(idx, state["target"])[:3]]
    elif state.get("intent") == "impact" and state.get("target"):
        rep = impact_mod.analyze(idx, state["target"], use_llm=False)
        extra["impact"] = rep
        wanted = set(rep.get("matched", [])[:2]) | {a["node"] for a in rep.get("affected", [])[:5]}
        hits = [search.Hit(c, 1.0) for c in idx.chunks if f"{c.file_path}::{c.qualified_name}" in wanted]
    seen = {h.chunk.id for h in hits}
    for h in search.search(idx, state.get("query") or state["question"], k=TOP_K,
                           file=state.get("filter_file") or None, kind=state.get("filter_kind") or None):
        if h.chunk.id not in seen and len(hits) < TOP_K:
            hits.append(h)
            seen.add(h.chunk.id)
    return {"hits": [dict(h.to_dict(), id=h.chunk.id) for h in hits], **extra,
            "trace": [f"retrieve: {len(hits)} sources for '{(state.get('query') or '')[:60]}'"]}


def grade(state: State) -> dict:
    hits = state.get("hits", [])
    model = llm.get_model()
    if model is None or not hits:
        return {"relevant": list(range(len(hits))), "trace": ["grade: skipped (no LLM)" if model is None
                                                              else "grade: nothing retrieved"]}
    listing = "\n\n".join(f"[{i}] {h['file_path']} `{h['qualified_name']}`\n{(h['signature'] or '')}\n"
                          f"{h['code'][:600]}" for i, h in enumerate(hits))
    verdict = llm.ask_json(
        "You decide which code snippets help answer a question about a codebase. Reply with JSON only.",
        f"Question: {state['question']}\n\nSnippets:\n{listing}\n\n"
        'Reply as {"relevant": [<indices of snippets that help answer>]}. Be inclusive when unsure.',
        default={"relevant": list(range(len(hits)))}, model=model)
    keep = [i for i in (verdict.get("relevant", []) if isinstance(verdict, dict) else [])
            if isinstance(i, int) and 0 <= i < len(hits)]
    return {"relevant": keep, "trace": [f"grade: {len(keep)}/{len(hits)} relevant"]}


def rewrite(state: State) -> dict:
    model = llm.get_model()
    reply = llm.ask(
        "You turn a question about a codebase into a better code-search query: the likely function, class, "
        "module and variable names (snake_case/camelCase variants), plus key terms. One line, no prose.",
        f"Question: {state['question']}\nPrevious query: {state.get('query')}", model) if model else state["question"]
    q = reply.strip().splitlines()[0][:300] if reply.strip() else state["question"]
    return {"query": q, "attempts": state.get("attempts", 0) + 1, "trace": [f"rewrite: {q[:80]}"]}


def _route_after_grade(state: State) -> str:
    if state.get("relevant") or state.get("attempts", 0) >= MAX_ATTEMPTS or llm.get_model() is None:
        return "generate"
    return "rewrite"


SYSTEM = (
    "You are CodeoGraph, a precise assistant for one software repository. Answer ONLY from the numbered sources. "
    "Cite every claim with the source number in square brackets, like [2]. Name functions and files in backticks. "
    "If the sources do not contain the answer, say so plainly instead of guessing. Be concise and technical; "
    "use short code blocks only when they help."
)


def _sources(state: State) -> list[search.Hit]:
    idx = _index(state)
    hits = state.get("hits", [])
    rel = state.get("relevant")
    chosen = [hits[i] for i in rel] if rel else hits
    out = []
    for h in chosen:
        pos = idx.position(h["id"])
        if pos is not None:
            out.append(search.Hit(idx.chunks[pos], h.get("score", 0.0)))
    return out


def generate(state: State) -> dict:
    srcs = _sources(state)
    model = llm.get_model()
    if model is None:
        if not srcs:
            ans = "No matching code was found in this repository."
        else:
            lines = ["No LLM is configured, so here are the most relevant places in the code:"]
            for n, h in enumerate(srcs[:5], 1):
                c = h.chunk
                doc = f" - {c.docstring.splitlines()[0]}" if c.docstring else ""
                lines.append(f"- [{n}] `{c.qualified_name}` in `{c.file_path}` lines {c.start_line}-{c.end_line}"
                             f"{doc}")
            if state.get("impact"):
                lines.append(f"\nImpact: {state['impact']['summary']}")
            ans = "\n".join(lines)
        return {"answer": ans, "trace": ["generate: extractive (no LLM)"]}
    ctx = search.context_block(srcs)
    if state.get("impact"):
        rep = state["impact"]
        ctx += "\n\nDependency-graph analysis: " + rep["summary"] + "\n" + "\n".join(
            f"- `{a['node']}` ({a['risk']}, {a['reasoning']})" for a in rep.get("affected", [])[:12])
    past = state.get("history", [])[-4:]
    convo = "".join(f"Earlier question: {t['q']}\nEarlier answer: {t['a'][:600]}\n\n" for t in past)
    from langchain_core.messages import HumanMessage, SystemMessage

    msg = model.invoke([SystemMessage(content=SYSTEM),
                        HumanMessage(content=f"{convo}Sources:\n\n{ctx}\n\nQuestion: {state['question']}")])
    return {"answer": llm.text_of(msg), "trace": [f"generate: {len(srcs)} sources"]}


def verify(state: State) -> dict:
    srcs = _sources(state)
    answer = state.get("answer", "")
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    valid = [n for n in used if 1 <= n <= len(srcs)]
    invalid = [n for n in used if n not in valid]
    for n in invalid:
        answer = answer.replace(f"[{n}]", "")
    cited = valid or list(range(1, min(len(srcs), 5) + 1))
    citations = [{"n": n, "file_path": srcs[n - 1].chunk.file_path, "symbol_name": srcs[n - 1].chunk.qualified_name,
                  "start_line": srcs[n - 1].chunk.start_line, "end_line": srcs[n - 1].chunk.end_line,
                  "docstring": srcs[n - 1].chunk.docstring[:200], "cited": n in valid} for n in cited]
    return {"answer": answer, "citations": citations, "invalid_citations": invalid,
            "history": [{"q": state["question"], "a": answer}],
            "trace": [f"verify: {len(valid)} valid citation(s)" + (f", removed {invalid}" if invalid else "")]}


# --------------------------------------------------------------------------- graph
def build_graph(checkpointer: Any | None = None):
    g = StateGraph(State)
    for name, fn in (("classify", classify), ("retrieve", retrieve), ("grade", grade), ("rewrite", rewrite),
                     ("generate", generate), ("verify", verify)):
        g.add_node(name, fn)
    g.add_edge(START, "classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _route_after_grade, {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "verify")
    g.add_edge("verify", END)
    return g.compile(checkpointer=checkpointer)


_MEMORY = InMemorySaver()
GRAPH = build_graph(_MEMORY)


def _inputs(question: str, namespace: str, history: list[dict] | None, file: str | None = None,
            kind: str | None = None) -> dict:
    # per-turn fields are reset; `history` and `trace` accumulate on the thread
    return {"question": question, "namespace": namespace, "filter_file": file or "", "filter_kind": kind or "",
            "hits": [], "relevant": [], "impact": {},
            "answer": "", "citations": [], "invalid_citations": [], **({"history": history} if history else {})}


def ask(question: str, namespace: str, thread_id: str | None = None, history: list[dict] | None = None,
        file: str | None = None, kind: str | None = None) -> dict:
    cfg = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}, "recursion_limit": 25}
    final = GRAPH.invoke(_inputs(question, namespace, history, file, kind), cfg)
    return {"answer": final.get("answer", ""), "citations": final.get("citations", []),
            "intent": final.get("intent"), "invalid_citations": final.get("invalid_citations", []),
            "trace": final.get("trace", [])[-12:], "thread_id": cfg["configurable"]["thread_id"],
            "model": llm.model_label()}


def stream(question: str, namespace: str, thread_id: str | None = None, history: list[dict] | None = None,
           file: str | None = None, kind: str | None = None) -> Iterator[tuple[str, Any]]:
    """Yields ("token", text) while the answer is written, then ("final", result dict)."""
    cfg = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}, "recursion_limit": 25}
    final: dict = {}
    streamed = False
    for mode, payload in GRAPH.stream(_inputs(question, namespace, history, file, kind), cfg,
                                      stream_mode=["messages", "values"]):
        if mode == "messages":
            chunk, meta = payload
            if meta.get("langgraph_node") == "generate":
                text = llm.text_of(chunk)
                if text:
                    streamed = True
                    yield "token", text
        else:
            final = payload
    if not streamed and final.get("answer"):
        yield "token", final["answer"]
    yield "final", {"answer": final.get("answer", ""), "citations": final.get("citations", []),
                    "intent": final.get("intent"), "invalid_citations": final.get("invalid_citations", []),
                    "trace": final.get("trace", [])[-12:], "thread_id": cfg["configurable"]["thread_id"]}
