"""CodeoGraph - the web app.

    streamlit run ui/app.py

Talks to the codeograph package directly (no separate server). Indexing runs in a background thread with live
progress; the demo repositories index themselves on the first visit.
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:                      # run from a checkout without `pip install -e .`
    sys.path.insert(0, str(ROOT))

from codeograph import agent, impact, indexer, llm, review, search, store  # noqa: E402

st.set_page_config(page_title="CodeoGraph - cited answers over any codebase", page_icon="🤖", layout="wide",
                   initial_sidebar_state="expanded")

DEMOS = {"psf/requests": "https://github.com/psf/requests",
         "pallets/click": "https://github.com/pallets/click",
         "CodeoGraph": "https://github.com/dhruvg0ya1/CodeoGraph"}
AUTO_DEMOS = [] if os.environ.get("CODEOGRAPH_NO_DEMOS") else ["https://github.com/psf/requests",
                                                                 "https://github.com/dhruvg0ya1/CodeoGraph"]
REPO = "https://github.com/dhruvg0ya1/CodeoGraph"

# --------------------------------------------------------------------------- style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
code, pre, .stCode code, .cg-mono, .cg-card .m { font-family: 'JetBrains Mono', monospace !important; }
*, *::before, *::after { border-radius: 0px !important; }
[data-testid="stIconMaterial"], .material-symbols-rounded, [class*="material-symbols"] {
  font-family: 'Material Symbols Rounded' !important; }
#MainMenu, footer, header { display: none !important; }

/* layout */
.stApp { background: #0d1b1b; }
.block-container { padding: 2rem 3rem !important; max-width: 1320px; }
[data-testid="stSidebar"] { background: #0a1a1a; border-right: 1px solid #0d7377; }
[data-testid="stSidebar"] * { color: #b0e0e0; }
[data-testid="stSidebar"] label { color: #14a098 !important; font-weight: 600; letter-spacing: 0.3px; }

/* type */
h1, h2, h3, h4, h5, h6 { color: #14f0f0 !important; font-weight: 700 !important; letter-spacing: -0.5px; }
h1 { font-size: 2rem !important; }
p, li, .stMarkdown { color: #c8e8e8; }
.stMarkdown strong { color: #14f0f0; }
a { color: #14a098 !important; } a:hover { color: #14f0f0 !important; }
hr { border-color: #0d7377 !important; opacity: 0.4; }
[data-testid="stCaptionContainer"], .stCaption { color: #5a9a9a !important; letter-spacing: 0.3px; }

/* inputs */
.stTextInput input, .stTextArea textarea, div[data-baseweb="select"] > div {
  background: #0a1a1a !important; border: 1px solid #0d7377 !important; color: #e0f7f7 !important;
  box-shadow: none !important; caret-color: #14f0f0; }
.stTextInput input:focus, .stTextArea textarea:focus { border-color: #14f0f0 !important; box-shadow: 0 0 0 1px #14f0f0 !important; }
.stTextInput input::placeholder, .stTextArea textarea::placeholder, [data-testid="stChatInput"] textarea::placeholder { color: #4a8a8a !important; }
[data-testid="stChatInput"] { border: 1px solid #0d7377 !important; background: #0a1a1a !important; }
[data-testid="stChatInput"]:focus-within { border-color: #14f0f0 !important; box-shadow: 0 0 0 1px #14f0f0; }

/* buttons */
.stButton > button, .stDownloadButton > button {
  background: transparent; color: #14a098; border: 1px solid #0d7377; font-weight: 600; font-size: 12px;
  letter-spacing: 0.6px; text-transform: uppercase; padding: 6px 18px; transition: all .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { background: #0d7377; color: #e0f7f7; border-color: #14a098; }
.stButton > button[kind="primary"] { background: #14f0f0; color: #0a1a1a !important; border: 1px solid #14f0f0; font-weight: 700; }
.stButton > button[kind="primary"] p { color: #0a1a1a !important; }
.stButton > button[kind="primary"]:hover { background: #14a098; border-color: #14a098; }
.stButton > button:disabled { background: transparent !important; color: #2f5f5f !important; border-color: #123a3a !important; }

/* pills + segmented control */
[data-testid="stButtonGroup"] button { background: #0a1a1a !important; border: 1px solid #0d7377 !important;
  color: #b0e0e0 !important; font-size: 13px; }
[data-testid="stButtonGroup"] button:hover { border-color: #14a098 !important; color: #e0f7f7 !important; }
[data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[kind*="Active"] {
  background: #0d7377 !important; color: #ffffff !important; border-color: #14f0f0 !important; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #0d7377; gap: 0; }
.stTabs [data-baseweb="tab"] { color: #5a9a9a !important; font-weight: 600; font-size: 13px; letter-spacing: 0.6px;
  text-transform: uppercase; padding: 10px 22px; }
.stTabs [data-baseweb="tab"] p { color: inherit !important; text-transform: uppercase; letter-spacing: 0.6px;
  font-weight: 600; font-size: 13px; }
[data-testid="stChatInput"] textarea { background: transparent !important; border: none !important; color: #e0f7f7 !important; }
[data-testid="stSidebar"] a { font-size: 0.8rem; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color: #14f0f0 !important; }
.stTabs [data-baseweb="tab-highlight"] { background: #14f0f0 !important; }
.stTabs [data-baseweb="tab"]:hover { color: #14a098 !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 22px; }

/* chat, expanders, alerts, code, progress */
[data-testid="stChatMessage"] { background: #0f1f1f; border: 1px solid #0d7377; padding: 16px 20px; margin-bottom: 6px; }
[data-testid="stChatMessage"]:hover { border-color: #14a098; }
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] { background: #0d7377 !important; }
[data-testid="stChatMessageAvatarUser"] svg, [data-testid="stChatMessageAvatarAssistant"] svg { fill: #e0f7f7 !important;
  color: #e0f7f7 !important; }
[data-testid="stExpander"] { background: #0f1f1f; border: 1px solid #0d7377; }
[data-testid="stExpander"]:hover { border-color: #14a098; }
[data-testid="stExpander"] summary { color: #14a098 !important; font-weight: 600; }
[data-testid="stExpander"] summary:hover { color: #14f0f0 !important; }
div[data-testid="stAlert"] { border: 1px solid #0d7377 !important; background: #0f1f1f !important; }
div[data-testid="stAlert"] p { color: #c8e8e8 !important; }
.stCode { border: 1px solid #0d7377; background: #0a1a1a; }
.stProgress > div > div > div { background: #0a1a1a; }
.stProgress > div > div > div > div { background: #14f0f0; }
[data-testid="stDataFrame"] { border: 1px solid #0d7377; }
::-webkit-scrollbar { width: 6px; height: 6px; } ::-webkit-scrollbar-track { background: #0a1a1a; }
::-webkit-scrollbar-thumb { background: #0d7377; } ::-webkit-scrollbar-thumb:hover { background: #14a098; }

/* CodeoGraph components */
.cg-brand { font-size: 1.35rem; font-weight: 800; color: #14f0f0 !important; letter-spacing: -0.3px; }
.cg-tag { color: #5a9a9a !important; font-size: 0.8rem; margin: 2px 0 14px 0; }
.cg-sub { color: #5a9a9a; font-size: 0.85rem; margin: -6px 0 16px 0; }
.stat-card { background: #0f1f1f; border: 1px solid #0d7377; padding: 14px 16px; text-align: center; }
.stat-card:hover { border-color: #14a098; }
.stat-card .value { font-size: 28px; font-weight: 800; color: #14f0f0; letter-spacing: -1px; line-height: 1.1; }
.stat-card .label { font-size: 10.5px; color: #5a9a9a; margin-top: 4px; letter-spacing: 1px; text-transform: uppercase; font-weight: 600; }
.cg-card { background: #0f1f1f; border: 1px solid #0d7377; padding: 12px 16px; margin-bottom: 6px; }
.cg-card:hover { border-color: #14a098; }
.cg-card .t { color: #e0f7f7; font-weight: 600; font-size: 0.98rem; }
.cg-card .m { color: #5a9a9a; font-size: 0.78rem; margin-top: 2px; }
.cg-card .d { color: #c8e8e8; font-size: 0.86rem; margin-top: 6px; }
.cg-badge { display: inline-block; padding: 1px 8px; font-size: 10px; font-weight: 700; letter-spacing: 0.6px;
  text-transform: uppercase; margin-left: 10px; vertical-align: middle; border: 1px solid #0d7377; color: #14a098; }
.k-function { background: #0d7377; color: #e0f7f7; border-color: #0d7377; }
.k-method { background: transparent; color: #14f0f0; border-color: #14f0f0; }
.k-class { background: #14a098; color: #0a1a1a; border-color: #14a098; }
.badge-High, .badge-high { background:#e0115f; color:#fff; border-color:#e0115f; }
.badge-Medium, .badge-medium { background:#14a098; color:#fff; border-color:#14a098; }
.badge-Low, .badge-low { background:#0d7377; color:#fff; border-color:#0d7377; }
.cg-bar { height: 3px; background: #0a1a1a; margin-top: 8px; }
.cg-bar span { display: block; height: 100%; background: #14a098; }
.cg-note { border-left: 2px solid #14f0f0; padding: 8px 14px; background: #0f1f1f; color: #b0e0e0; font-size: 0.85rem;
  margin: 2px 0 14px 0; }
.cg-step { color: #5a9a9a; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace !important; }
.cg-model { display: inline-block; border: 1px solid #0d7377; padding: 3px 10px; font-size: 11px; letter-spacing: 0.5px;
  text-transform: uppercase; color: #5a9a9a; }
</style>
""", unsafe_allow_html=True)


def esc(s) -> str:
    return html.escape("" if s is None else str(s))


TEST_PATH = search.TEST_PATH


# --------------------------------------------------------------------------- indexing jobs
@st.cache_resource
def _registry() -> dict:
    return {"lock": threading.Lock(), "jobs": {}, "pool": ThreadPoolExecutor(max_workers=1), "auto": False}


def start_index(url: str, force: bool = False) -> str:
    reg = _registry()
    ns = indexer.namespace_for(url)
    with reg["lock"]:
        if reg["jobs"].get(ns, {}).get("status") in ("queued", "running"):
            return ns
        reg["jobs"][ns] = {"status": "queued", "url": url, "current_file": "Waiting for the indexer...",
                           "total_files": 0, "processed_files": 0, "total_chunks": 0, "indexed_chunks": 0}

    def run():
        def cb(**kw):
            with reg["lock"]:
                reg["jobs"][ns].update(kw, status="running")
        try:
            meta = indexer.index_repository(url, force=force, progress=cb)
            with reg["lock"]:
                reg["jobs"][ns].update(status="done", chunk_count=meta.get("chunk_count"))
        except Exception as e:  # noqa: BLE001 - shown in the UI
            with reg["lock"]:
                reg["jobs"][ns].update(status="error", error=str(e)[:300])
    reg["pool"].submit(run)
    return ns


def active_jobs() -> dict:
    reg = _registry()
    with reg["lock"]:
        return {k: dict(v) for k, v in reg["jobs"].items() if v["status"] in ("queued", "running", "error")}


def auto_demos() -> None:
    reg = _registry()
    if reg["auto"]:
        return
    reg["auto"] = True
    have = {r["namespace"] for r in store.list_indexes()}
    for url in AUTO_DEMOS:
        if indexer.namespace_for(url) not in have:
            start_index(url)


@st.fragment(run_every=1.5)
def progress_panel() -> None:
    jobs = active_jobs()
    if not jobs:
        if st.session_state.get("_had_jobs"):
            st.session_state["_had_jobs"] = False
            st.rerun()
        return
    st.session_state["_had_jobs"] = True
    for ns, j in jobs.items():
        if j["status"] == "error":
            st.error(f"Indexing {ns} failed: {j.get('error')}")
            with _registry()["lock"]:
                _registry()["jobs"][ns]["status"] = "failed-shown"
            continue
        tot = j.get("total_chunks") or 0
        frac = (j.get("indexed_chunks", 0) / tot) if tot else (
            (j.get("processed_files", 0) / j["total_files"]) * 0.3 if j.get("total_files") else 0.02)
        st.progress(min(max(frac, 0.02), 1.0), text=f"Indexing {ns} - {j.get('current_file', '')}"[:120])


# --------------------------------------------------------------------------- helpers
def stats(idx: store.RepoIndex) -> dict:
    kinds: dict[str, int] = {}
    for c in idx.chunks:
        kinds[c.symbol_type] = kinds.get(c.symbol_type, 0) + 1
    g = idx.meta.get("graph", {})
    return {"files": idx.meta.get("file_count", 0), "functions": kinds.get("function", 0) + kinds.get("method", 0),
            "classes": kinds.get("class", 0), "calls": g.get("calls", 0), "imports": g.get("imports", 0),
            "languages": ", ".join(idx.meta.get("languages", [])), "commit": (idx.meta.get("commit") or "")[:7]}


def chunk_of(idx: store.RepoIndex, file_path: str, qual: str):
    return next((c for c in idx.chunks if c.file_path == file_path and c.qualified_name == qual), None)


def lang(c) -> str:
    return {"tsx": "typescript"}.get(c.language, c.language)


def card(title: str, kind: str, meta: str, desc: str = "", score: float | None = None) -> str:
    bar = (f'<div class="cg-bar"><span style="width:{max(4, min(100, score * 100)):.0f}%"></span></div>'
           if score is not None else "")
    return (f'<div class="cg-card"><div class="t">{esc(title)}<span class="cg-badge k-{esc(kind)}">{esc(kind)}</span>'
            f'</div><div class="m">{esc(meta)}</div>' + (f'<div class="d">{esc(desc)}</div>' if desc else "") +
            f"{bar}</div>")


def metric(col, value, label: str) -> None:
    col.markdown(f'<div class="stat-card"><div class="value">{esc(value)}</div><div class="label">{esc(label)}</div>'
                 f'</div>', unsafe_allow_html=True)


def suggestions(ns: str) -> list[str]:
    if "requests" in ns:
        return ["How does it decide to drop the Authorization header on a redirect?",
                "Where is the proxy chosen for a URL?", "What breaks if I change `merge_setting`?",
                "Where is `Session.send` defined?"]
    if "codeograph" in ns:
        return ["How does the agent check its citations?", "How are BM25 and dense scores fused?",
                "What breaks if I change `tokenize`?", "Where is `index_repository` defined?"]
    if "click" in ns:
        return ["How are command-line options parsed?", "How does a group dispatch to a subcommand?",
                "What breaks if I change `Context.invoke`?", "Where is `echo` defined?"]
    return ["Where does the program start?", "How are errors handled?", "What are the main classes?"]


def top_targets(idx: store.RepoIndex, n: int = 8) -> list[str]:
    g = idx.graph
    scored = [(sum(1 for _, _, d in g.in_edges(node, data=True) if d.get("type") == "calls"), node)
              for node in g.nodes if "::" in node]
    return [node.split("::", 1)[1] for s, node in sorted(scored, reverse=True)[:n] if s > 0]


def dq(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def show_citations(idx: store.RepoIndex, msg: dict) -> None:
    cites = [c for c in msg.get("citations", []) if c.get("cited")] or msg.get("citations", [])[:3]
    for c in cites:
        ch = chunk_of(idx, c["file_path"], c["symbol_name"])
        with st.expander(f"[{c['n']}]  {c['symbol_name']}  ·  {c['file_path']}:{c['start_line']}-{c['end_line']}"):
            if ch:
                st.code(ch.code, language=lang(ch), line_numbers=True)
    if msg.get("trace"):
        with st.expander("How CodeoGraph answered"):
            for step in msg["trace"]:
                st.markdown(f'<div class="cg-step">→ {esc(step)}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- sidebar
auto_demos()
repos = store.list_indexes()
names = [r["namespace"] for r in repos]
if st.session_state.get("ns") not in names:
    st.session_state.ns = names[0] if names else None

with st.sidebar:
    st.markdown('<div class="cg-brand">CodeoGraph</div><div class="cg-tag">Cited answers, change impact and PR '
                'review over any codebase</div>', unsafe_allow_html=True)
    if names:
        st.session_state.ns = st.selectbox("Repository", names, index=names.index(st.session_state.ns))
    with st.expander("Add a repository", expanded=not names):
        url = st.text_input("Public GitHub URL", placeholder="https://github.com/owner/repo")
        if st.button("Index it", type="primary", use_container_width=True, disabled=not url.strip()):
            start_index(url.strip())
            st.toast(f"Indexing {indexer.namespace_for(url.strip())} - about a minute for a small repo.")
        st.caption("Or one of these:")
        for label, demo_url in DEMOS.items():
            if st.button(label, use_container_width=True, key=f"demo_{label}"):
                if indexer.namespace_for(demo_url) in names:
                    st.session_state.ns = indexer.namespace_for(demo_url)
                    st.rerun()
                start_index(demo_url)
    progress_panel()
    st.divider()
    model = llm.model_label()
    st.markdown(f'<span class="cg-model">LLM: {esc(model if model != "none" else "none - extractive")}</span>',
                unsafe_allow_html=True)
    st.markdown(f"[Source on GitHub]({REPO}) · [How it is measured]({REPO}#evaluation)")

ns = st.session_state.ns
idx = store.load(ns) if ns else None

# --------------------------------------------------------------------------- empty state
if idx is None:
    st.markdown("# CodeoGraph")
    st.markdown('<div class="cg-sub">Ask any codebase and get answers that cite the code, by file and line.</div>',
                unsafe_allow_html=True)
    st.info("The demo repositories are being indexed - progress is in the sidebar, and this page fills in by itself.")
    st.stop()

s = stats(idx)
st.markdown(f"# {ns}")
st.markdown(f'<div class="cg-sub cg-mono">{esc(idx.meta.get("repo_url", ""))}'
            f'{"  @ " + esc(s["commit"]) if s["commit"] else ""}  ·  {esc(s["languages"])}</div>',
            unsafe_allow_html=True)
cols = st.columns(5)
for col, (val, label) in zip(cols, [(s["files"], "files"), (s["functions"], "functions & methods"),
                                    (s["classes"], "classes"), (s["calls"], "call edges"),
                                    (s["imports"], "import edges")], strict=True):
    metric(col, val, label)
st.write("")

t_ask, t_search, t_impact, t_review, t_arch, t_bench = st.tabs(
    ["Ask", "Search", "Impact", "PR review", "Architecture", "Benchmarks"])

# --------------------------------------------------------------------------- ask
with t_ask:
    if llm.get_model() is None:
        st.markdown('<div class="cg-note">This demo runs without an LLM key, so answers are <b>extractive</b>: the '
                    'agent still classifies the question, retrieves, checks and cites - and returns the exact '
                    'functions that answer you. Run it with <code>GOOGLE_API_KEY</code> or any OpenAI-compatible '
                    'model for written answers.</div>', unsafe_allow_html=True)
    chats = st.session_state.setdefault("chats", {})
    history = chats.setdefault(ns, [])
    conversation = st.container()                  # created first, so it sits above the input
    picked = st.pills("Try one", suggestions(ns), key=f"pill_{ns}")
    typed = st.chat_input(f"Ask anything about {ns}")
    question = typed
    if picked and st.session_state.get(f"_last_pill_{ns}") != picked:
        question = question or picked
    st.session_state[f"_last_pill_{ns}"] = picked
    conversation.__enter__()

    for m in history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                show_citations(idx, m)

    if question:
        past = [{"q": history[i]["content"], "a": history[i + 1]["content"]}
                for i in range(0, len(history) - 1, 2) if history[i + 1]["role"] == "assistant"][-4:]
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            final: dict = {}

            def tokens():
                for kind, payload in agent.stream(question, ns, history=past or None):
                    if kind == "token":
                        yield payload
                    else:
                        final.update(payload)
            with st.spinner("Retrieving, grading and checking citations..."):
                st.write_stream(tokens())
            msg = {"role": "assistant", "content": final.get("answer", ""), "citations": final.get("citations", []),
                   "trace": final.get("trace", [])}
            show_citations(idx, msg)
            history.append(msg)
    conversation.__exit__(None, None, None)

# --------------------------------------------------------------------------- search
with t_search:
    c1, c2, c3 = st.columns([5, 2, 2])
    q = c1.text_input("Search the code", placeholder="e.g. retry with backoff, parse the config file ...",
                      key=f"q_{ns}")
    mode = c2.segmented_control("Retrieval", ["Hybrid", "BM25", "Dense"], default="Hybrid", key="mode")
    kind = c3.selectbox("Kind", ["any", "function", "method", "class", "module", "type"], key="kind")
    with_tests = st.toggle("Include test files", value=False, key="with_tests")
    if q.strip():
        hits = search.search(idx, q, k=10, mode=(mode or "Hybrid").lower(), kind=None if kind == "any" else kind,
                             skip_tests=not with_tests)
        if not hits:
            st.info("Nothing matched.")
        top = max((h.score for h in hits), default=1.0) or 1.0
        for rank, h in enumerate(hits, 1):
            c = h.chunk
            doc = c.docstring.splitlines()[0] if c.docstring else (c.signature or "")
            st.markdown(card(f"{rank}. {c.qualified_name}", c.symbol_type,
                             f"{c.file_path}:{c.start_line}-{c.end_line}", doc[:220], h.score / top),
                        unsafe_allow_html=True)
            with st.expander("Code", expanded=rank == 1):
                st.code(c.code, language=lang(c), line_numbers=True)
    else:
        st.markdown('<div class="cg-note">Hybrid search fuses BM25 over split identifiers (<code>getUserById</code> '
                    '→ get, user, by, id) with dense embeddings. Switch to BM25 or Dense to see how each one ranks '
                    'on its own.</div>', unsafe_allow_html=True)
    st.markdown("#### Go to definition")
    sym = st.text_input("Symbol", placeholder="Session.send, tokenize, Cart ...", key=f"sym_{ns}",
                        label_visibility="collapsed")
    if sym.strip():
        found = search.find_symbol(idx, sym)
        if not found:
            st.warning(f"`{sym}` is not defined in {ns}.")
        for c in found[:3]:
            st.markdown(card(c.qualified_name, c.symbol_type, f"{c.file_path}:{c.start_line}-{c.end_line}",
                             c.signature), unsafe_allow_html=True)
            st.code(c.code, language=lang(c), line_numbers=True)

# --------------------------------------------------------------------------- impact
with t_impact:
    quick = top_targets(idx)
    c1, c2 = st.columns([5, 2])
    target = c1.text_input("What are you about to change?", placeholder="a function, Class.method or a file path",
                           key=f"imp_{ns}")
    hops = c2.slider("Hops", 1, 4, 3)
    pick = st.pills("Most depended-on", quick, key=f"imp_pill_{ns}") if quick else None
    target = target.strip() or (pick or "")
    if target:
        rep = impact.analyze(idx, target, max_hops=hops, use_llm=False)
        aff = rep["affected"]
        counts = {r: sum(a["risk"] == r for a in aff) for r in ("High", "Medium", "Low")}
        m = st.columns(4)
        metric(m[0], rep.get("total_affected", 0), "dependents")
        metric(m[1], counts["High"], "call it directly")
        metric(m[2], counts["Medium"], "import its module")
        metric(m[3], len({a["file"] for a in aff}), "files")
        st.write("")
        st.caption(rep["summary"])
        if aff:
            nodes = set(rep["matched"][:3]) | {a["node"] for a in aff[:30]}
            risk = {a["node"]: a["risk"] for a in aff}
            color = {"High": "#e0115f", "Medium": "#14a098", "Low": "#0d7377"}
            lines = ['digraph G { rankdir=RL; bgcolor="transparent"; nodesep=0.25; ranksep=0.6;',
                     'node [shape=box style="filled" fontname="Helvetica" fontsize=11 fontcolor="#ffffff" '
                     'penwidth=0]; edge [color="#14a098" arrowsize=0.6];']
            for n in nodes:
                label = n.split("::", 1)[1] if "::" in n else n
                fill = "#14f0f0" if n in rep["matched"] else color.get(risk.get(n, "Low"), "#0d7377")
                font = "#0a1a1a" if n in rep["matched"] else "#ffffff"
                lines.append(f'"{dq(n)}" [label="{dq(label)}\\n{dq(n.split("::")[0])}" fillcolor="{fill}" '
                             f'fontcolor="{font}"];')
            for u, v, d in idx.graph.edges(data=True):
                if u in nodes and v in nodes and d.get("type") in ("calls", "imports"):
                    style = "solid" if d["type"] == "calls" else "dashed"
                    lines.append(f'"{dq(u)}" -> "{dq(v)}" [style={style}];')
            lines.append("}")
            st.graphviz_chart("\n".join(lines), use_container_width=True)
            st.caption("Cyan: what you change · red: calls it directly · teal: imports its module · dark: further "
                       "away. Solid arrows are calls, dashed are imports.")
            st.dataframe([{"risk": a["risk"], "symbol": a["symbol"], "file": a["file"], "hops": a["hops"],
                           "via": a["via"], "why": a["reasoning"]} for a in aff], use_container_width=True,
                         hide_index=True)

# --------------------------------------------------------------------------- review
with t_review:
    def sample_diff() -> str:
        for name in top_targets(idx, 20):
            c = next((c for c in idx.chunks if c.qualified_name == name and c.symbol_type in ("function", "method")
                      and c.language == "python"), None)
            if not c:
                continue
            src = idx.files.get(c.file_path, "").splitlines()
            body = next((i for i in range(c.start_line, min(c.end_line, len(src))) if src[i].strip()
                         and not src[i].strip().startswith(('"""', "'''", "#", "@", "def ", ")"))), None)
            if body is None:
                continue
            line = src[body]
            indent = line[: len(line) - len(line.lstrip())]
            return (f"diff --git a/{c.file_path} b/{c.file_path}\n--- a/{c.file_path}\n+++ b/{c.file_path}\n"
                    f"@@ -{body + 1},1 +{body + 1},2 @@\n+{indent}# behaviour change under review\n {line}\n")
        return ""

    if st.button("Load a sample diff for this repository"):
        st.session_state[f"diff_{ns}"] = sample_diff()
    diff = st.text_area("Paste a unified diff (the output of git diff)", height=180, key=f"diff_{ns}",
                        placeholder="diff --git a/... b/...")
    if diff.strip():
        r = review.review(idx, diff)
        st.markdown(f'<div class="cg-card"><div class="t">Risk<span class="cg-badge badge-{esc(r["risk_score"])}">'
                    f'{esc(r["risk_score"])}</span></div><div class="d">{esc(r["summary"])}</div>'
                    f'<div class="m">{esc(r.get("risk_justification", ""))}</div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Changed symbols**")
            for sname in r["affected_symbols"] or ["(none matched the index)"]:
                st.markdown(f"- `{sname}`")
            st.markdown("**Risks**")
            st.markdown(r["risks"])
        with c2:
            st.markdown("**Dependents outside the diff**")
            st.dataframe([{"risk": a["risk"], "symbol": a["symbol"], "file": a["file"], "hops": a["hops"]}
                          for a in r["dependents"]], use_container_width=True, hide_index=True)
            st.markdown("**Suggested tests**")
            st.markdown(r["suggested_tests"])
        st.download_button("Download the review (Markdown)", review.to_markdown(r, diff), file_name="review.md")

# --------------------------------------------------------------------------- architecture
with t_arch:
    g = idx.graph
    edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("type") == "imports" and u != v]
    if not edges:
        st.info("No imports between files were found in this repository.")
    else:
        deg: dict[str, int] = {}
        for u, v in edges:
            deg[u] = deg.get(u, 0) + 1
            deg[v] = deg.get(v, 0) + 1
        c1, c2 = st.columns([4, 1])
        limit = c1.slider("Files shown (most connected first)", 10, 80, 35)
        arch_tests = c2.toggle("Include tests", value=False, key="arch_tests")
        ranked_files = [f for f in sorted(deg, key=lambda f: -deg[f]) if arch_tests or not TEST_PATH.search(f)]
        keep = set(ranked_files[:limit])
        groups: dict[str, list[str]] = {}
        for f in keep:
            groups.setdefault(str(Path(f).parent), []).append(f)
        lines = ['digraph A { rankdir=LR; bgcolor="transparent"; nodesep=0.2; ranksep=0.5;',
                 'node [shape=box style="filled" fillcolor="#0f1f1f" color="#0d7377" fontcolor="#c8e8e8" '
                 'fontname="Helvetica" fontsize=10]; edge [color="#14a098" arrowsize=0.5];']
        for i, (folder, files) in enumerate(sorted(groups.items())):
            lines.append(f'subgraph cluster_{i} {{ label="{dq(folder)}"; fontcolor="#5a9a9a"; color="#0d7377"; '
                         'fontname="Helvetica"; fontsize=10;')
            lines += [f'"{dq(f)}" [label="{dq(Path(f).name)}"];' for f in files]
            lines.append("}")
        lines += [f'"{dq(u)}" -> "{dq(v)}";' for u, v in edges if u in keep and v in keep]
        lines.append("}")
        st.graphviz_chart("\n".join(lines), use_container_width=True)
        st.caption(f"{len(keep)} of {len(deg)} connected files, grouped by folder; an arrow means 'imports'.")

# --------------------------------------------------------------------------- benchmarks
with t_bench:
    res = ROOT / "evals" / "results"
    st.markdown("CodeoGraph is measured in CI on **40 hand-written questions about psf/requests**. Most describe "
                "behaviour in plain English without naming the function, and every change to the search code has "
                "to keep Recall@5 above the gate.")
    try:
        rows = json.loads((res / "retrieval.json").read_text())
    except (OSError, json.JSONDecodeError):
        rows = []
    if rows:
        base = [r for r in rows if r["model"] == rows[0]["model"]]
        default = next((r for r in base if "default" in r["config"]), base[0])
        m = st.columns(4)
        metric(m[0], f"{default['recall@1']:.2f}", "Recall@1 (default pipeline)")
        metric(m[1], f"{default['recall@5']:.2f}", "Recall@5")
        metric(m[2], f"{default['mrr@10']:.3f}", "MRR@10")
        metric(m[3], f"{default['p50_ms']:.0f} ms", "median query time")
        st.write("")
        st.bar_chart({re.sub(r" \(default\)", " *", r["config"]): r["mrr@10"] for r in base}, horizontal=True,
                     x_label="MRR@10", color="#14a098", height=320)
        st.caption("* default. The generic MS MARCO cross-encoder made code search worse and costs over a second a "
                   "query, so reranking is off by default; score-weighted fusion beat Reciprocal Rank Fusion.")
        with st.expander("Every configuration"):
            st.dataframe([{k: r[k] for k in ("model", "config", "recall@1", "recall@5", "recall@10", "mrr@10",
                                             "ndcg@10", "p50_ms")} for r in rows], use_container_width=True,
                         hide_index=True)
    try:
        ans = json.loads((res / "answers.json").read_text())["summary"]
        st.markdown("#### Answers: LangGraph agent, scored with RAGAS")
        m = st.columns(4)
        metric(m[0], f"{ans['target_cited']:.0%}", "cite the right function")
        metric(m[1], f"{ans['has_citation']:.0%}", "carry a valid citation")
        metric(m[2], f"{ans.get('ragas_context_precision') or 0:.2f}", "RAGAS context precision")
        metric(m[3], ans["model"].split(":")[-1], "model (CPU runner)")
        st.caption("Answered and judged by a small open model on a CPU runner, so treat answer quality as a floor; "
                   "the retrieval numbers above do not depend on the model.")
    except (OSError, json.JSONDecodeError, KeyError):
        pass
