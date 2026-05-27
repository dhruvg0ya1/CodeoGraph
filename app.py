"""
frontend/app.py

RepoChat — Streamlit UI.

Architecture notes:
- All API calls go to the FastAPI backend at BACKEND_URL (default localhost:8000)
- st.session_state holds: active_namespace, messages, repos list
- Chat responses stream via requests.get with stream=True, parsed line by line
- The special <!--SOURCES:...--> trailer is stripped and rendered as an expander
"""

import json
import time
from typing import Optional

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BACKEND_URL = "http://localhost:8000"

st.set_page_config(
    page_title="CodeoGraph",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark, minimal, production-looking
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0f0f13;
        border-right: 1px solid #1e1e2e;
    }

    /* Main area */
    .stApp { background: #13131a; }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #1a1a27;
        border: 1px solid #2a2a3e;
        border-radius: 10px;
        margin-bottom: 8px;
        padding: 12px;
    }

    /* Inputs */
    .stTextInput > div > div > input,
    .stTextArea textarea {
        background: #1a1a27 !important;
        border: 1px solid #2a2a3e !important;
        color: #e2e2f0 !important;
        border-radius: 8px !important;
    }

    /* Buttons */
    .stButton > button {
        background: #6366f1;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #4f52d4; }

    /* Risk badges */
    .badge-high   { background:#ef4444; color:white; padding:3px 10px; border-radius:12px; font-weight:700; font-size:12px; }
    .badge-medium { background:#f59e0b; color:white; padding:3px 10px; border-radius:12px; font-weight:700; font-size:12px; }
    .badge-low    { background:#22c55e; color:white; padding:3px 10px; border-radius:12px; font-weight:700; font-size:12px; }

    /* Stat cards */
    .stat-card {
        background: #1a1a27;
        border: 1px solid #2a2a3e;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .stat-card .value { font-size: 28px; font-weight: 700; color: #6366f1; }
    .stat-card .label { font-size: 12px; color: #888; margin-top: 2px; }

    /* Progress bar color */
    .stProgress > div > div { background: #6366f1; }

    /* Expander */
    [data-testid="stExpander"] {
        background: #1a1a27;
        border: 1px solid #2a2a3e;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "active_namespace" not in st.session_state:
    st.session_state.active_namespace = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "repos" not in st.session_state:
    st.session_state.repos = []
if "page" not in st.session_state:
    st.session_state.page = "Index Repo"
if "pending_reindex_url" not in st.session_state:
    st.session_state.pending_reindex_url = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_repos() -> list[dict]:
    try:
        r = requests.get(f"{BACKEND_URL}/repos", timeout=5)
        return r.json() if r.ok else []
    except Exception:
        return []


def get_active_repo_info() -> Optional[dict]:
    ns = st.session_state.active_namespace
    return next((r for r in st.session_state.repos if r["namespace"] == ns), None)


def risk_badge(level: str) -> str:
    cls = f"badge-{level.lower()}"
    return f'<span class="{cls}">{level}</span>'


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🤖 CodeoGraph")
    st.markdown("*AI-powered codebase understanding*")
    st.divider()

    # Refresh repos list
    st.session_state.repos = fetch_repos()
    repo_options = {r["namespace"]: r["repo_url"] for r in st.session_state.repos}

    if repo_options:
        selected_ns = st.selectbox(
            "Active Repository",
            options=list(repo_options.keys()),
            format_func=lambda ns: repo_options.get(ns, ns),
            index=(
                list(repo_options.keys()).index(st.session_state.active_namespace)
                if st.session_state.active_namespace in repo_options
                else 0
            ),
        )
        st.session_state.active_namespace = selected_ns
    else:
        st.info("No repos indexed yet.")

    # Active repo stats
    info = get_active_repo_info()
    if info:
        st.markdown("**Repo Stats**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="stat-card"><div class="value">{info["chunk_count"]}</div>'
                f'<div class="label">Chunks</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div class="stat-card"><div class="value">{info["file_count"]}</div>'
                f'<div class="label">Files</div></div>',
                unsafe_allow_html=True,
            )
        langs = ", ".join(info.get("languages", [])) or "—"
        st.caption(f"Languages: {langs}")

    st.divider()

    # Navigation
    pages = ["Index Repo", "Chat", "Change Impact", "PR Review", "Architecture"]
    for p in pages:
        if st.button(p, use_container_width=True, key=f"nav_{p}"):
            st.session_state.page = p
            st.rerun()

    # Delete repo
    if st.session_state.active_namespace and st.session_state.repos:
        st.divider()
        if st.button("🗑 Delete Active Repo", use_container_width=True):
            requests.delete(
                f"{BACKEND_URL}/repos/{st.session_state.active_namespace}", timeout=10
            )
            st.session_state.active_namespace = None
            st.session_state.repos = []
            st.rerun()


# ===========================================================================
# PAGE: Index Repo
# ===========================================================================

def page_index():
    st.title("Index Repository")
    st.markdown("Clone and index a GitHub repository for semantic code search.")

    with st.form("index_form"):
        repo_url = st.text_input(
            "GitHub URL",
            placeholder="https://github.com/owner/repo",
        )
        branch = st.text_input("Branch (optional — auto-detects default)", placeholder="main")
        submitted = st.form_submit_button("Index Repository", use_container_width=True)

    if submitted and repo_url.strip():
        # Check if already indexed
        check = requests.post(
            f"{BACKEND_URL}/index",
            json={"repo_url": repo_url.strip(), "branch": branch.strip() or None, "force_reindex": False},
            timeout=10,
        )
        if check.ok:
            data = check.json()

            if data.get("status") == "already_indexed":
                st.warning(
                    f"This repo is already indexed — **{data['chunk_count']} chunks** "
                    f"across **{data['file_count']} files**."
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Use Existing"):
                        st.session_state.active_namespace = data["namespace"]
                        st.session_state.page = "Chat"
                        st.rerun()
                with col2:
                    if st.button("🔄 Re-index"):
                        st.session_state.pending_reindex_url = repo_url.strip()
                        # Force re-index
                        requests.post(
                            f"{BACKEND_URL}/index",
                            json={"repo_url": repo_url.strip(), "branch": branch.strip() or None, "force_reindex": True},
                            timeout=10,
                        )
                        _poll_indexing(data["namespace"])
            else:
                namespace = data["namespace"]
                st.session_state.active_namespace = namespace
                _poll_indexing(namespace)
        else:
            st.error(f"Backend error: {check.text}")


def _poll_indexing(namespace: str):
    """Poll /index/status every 2 seconds and show live progress."""
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    progress_bar = progress_placeholder.progress(0)
    max_wait = 600  # 10 minutes
    elapsed = 0

    while elapsed < max_wait:
        try:
            r = requests.get(
                f"{BACKEND_URL}/index/status",
                params={"namespace": namespace},
                timeout=5,
            )
            s = r.json()
        except Exception:
            time.sleep(2)
            elapsed += 2
            continue

        status = s.get("status", "running")
        total = s.get("total_files", 1) or 1
        processed = s.get("processed_files", 0)
        indexed = s.get("indexed_chunks", 0)
        current = s.get("current_file", "")

        pct = min(int((processed / total) * 100), 99)
        progress_bar.progress(pct / 100)
        status_placeholder.markdown(
            f"**Processing file {processed} of {total}** — {indexed} chunks indexed  \n"
            f"`{current}`"
        )

        if status == "done":
            progress_bar.progress(1.0)
            status_placeholder.empty()
            st.success(
                f"✅ Indexing complete! **{indexed} chunks** from **{total} files**."
            )
            st.session_state.repos = fetch_repos()
            st.session_state.active_namespace = namespace
            break
        elif status == "error":
            st.error(f"Indexing failed: {s.get('error', 'Unknown error')}")
            break

        time.sleep(2)
        elapsed += 2


# ===========================================================================
# PAGE: Chat
# ===========================================================================

def page_chat():
    st.title("Chat with Codebase")

    if not st.session_state.active_namespace:
        st.warning("No active repository. Index one first.")
        return

    ns = st.session_state.active_namespace

    # --- Symbol search ---
    with st.container():
        col1, col2 = st.columns([4, 1])
        with col1:
            symbol_query = st.text_input(
                "🔍 Jump to symbol",
                placeholder="Type a function or class name…",
                label_visibility="collapsed",
            )
        with col2:
            search_btn = st.button("Go", use_container_width=True)

    if search_btn and symbol_query.strip():
        r = requests.get(
            f"{BACKEND_URL}/symbol",
            params={"name": symbol_query.strip(), "namespace": ns},
            timeout=10,
        )
        if r.ok:
            sym = r.json()
            with st.expander(
                f"📍 `{sym.get('symbol_name')}` — {sym.get('file_path')} "
                f"(L{sym.get('start_line')}–{sym.get('end_line')})",
                expanded=True,
            ):
                if sym.get("docstring"):
                    st.markdown(f"*{sym['docstring']}*")
                st.code(sym.get("raw_code", ""), language=sym.get("language", "python"))
        else:
            st.error(f"Symbol not found: `{symbol_query}`")

    st.divider()

    # --- Sidebar filters ---
    with st.sidebar:
        st.markdown("**Chat Filters**")
        filter_file = st.text_input("Filter by file", placeholder="src/auth.py")
        filter_type = st.selectbox(
            "Filter by type",
            ["(none)", "function", "class", "module"],
        )
        filter_type_val = None if filter_type == "(none)" else filter_type
        filter_file_val = filter_file.strip() or None

    # --- Chat messages ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📎 Sources", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f"**`{src['symbol_name']}`** — `{src['file_path']}` "
                            f"L{src['start_line']}–{src['end_line']}"
                        )
                        if src.get("docstring"):
                            st.caption(src["docstring"])

    # --- Chat input ---
    if user_input := st.chat_input("Ask anything about the codebase…"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Build history payload
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
        ]

        payload = {
            "question": user_input,
            "namespace": ns,
            "history": history,
            "filter_file": filter_file_val,
            "filter_type": filter_type_val,
        }

        with st.chat_message("assistant"):
            full_text = ""
            sources: list[dict] = []
            text_placeholder = st.empty()

            # Stream response
            try:
                with requests.post(
                    f"{BACKEND_URL}/query",
                    json=payload,
                    stream=True,
                    timeout=120,
                ) as resp:
                    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full_text += chunk

                # Parse out the sources trailer
                SOURCE_MARKER = "<!--SOURCES:"
                if SOURCE_MARKER in full_text:
                    text_part, rest = full_text.split(SOURCE_MARKER, 1)
                    sources_json = rest.rstrip("-->").strip()
                    try:
                        sources = json.loads(sources_json)
                    except Exception:
                        sources = []
                    full_text = text_part.strip()

                text_placeholder.markdown(full_text)

                if sources:
                    with st.expander("📎 Sources", expanded=False):
                        for src in sources:
                            st.markdown(
                                f"**`{src['symbol_name']}`** — `{src['file_path']}` "
                                f"L{src['start_line']}–{src['end_line']}"
                            )
                            if src.get("docstring"):
                                st.caption(src["docstring"])

            except Exception as e:
                full_text = f"⚠️ Stream error: {e}"
                text_placeholder.error(full_text)

        st.session_state.messages.append(
            {"role": "assistant", "content": full_text, "sources": sources}
        )


# ===========================================================================
# PAGE: Change Impact
# ===========================================================================

def page_impact():
    st.title("Change Impact Analysis")
    st.markdown(
        "Enter a function or file name to see what else in the codebase "
        "would be affected if it changed."
    )

    if not st.session_state.active_namespace:
        st.warning("No active repository.")
        return

    col1, col2 = st.columns([4, 1])
    with col1:
        symbol = st.text_input(
            "Function or file name",
            placeholder="authenticate_user",
            label_visibility="collapsed",
        )
    with col2:
        analyze_btn = st.button("Analyze Impact", use_container_width=True)

    if analyze_btn and symbol.strip():
        with st.spinner("Analyzing dependency graph…"):
            r = requests.post(
                f"{BACKEND_URL}/impact",
                json={"symbol": symbol.strip(), "namespace": st.session_state.active_namespace},
                timeout=120,
            )

        if r.ok:
            data = r.json()
            st.markdown(f"### Target: `{data['target']}`")
            st.info(data["summary"])

            affected = data.get("affected", [])
            if not affected:
                st.success("No dependents found — this symbol is safe to change in isolation.")
                return

            # Risk table
            st.markdown("#### Affected Symbols")
            for item in affected:
                risk = item.get("risk", "Medium")
                col_sym, col_file, col_risk = st.columns([2, 3, 1])
                with col_sym:
                    st.markdown(f"`{item['symbol']}`")
                with col_file:
                    st.markdown(f"`{item['file']}`")
                with col_risk:
                    st.markdown(risk_badge(risk), unsafe_allow_html=True)

                with st.expander(f"Details — {item['symbol']}"):
                    st.markdown(item.get("reasoning", ""))
                st.divider()
        else:
            st.error(f"Error: {r.text}")


# ===========================================================================
# PAGE: PR Review
# ===========================================================================

def page_pr_review():
    st.title("PR Review Agent")
    st.markdown(
        "Paste a `git diff` to get an AI-powered code review grounded in your codebase."
    )

    if not st.session_state.active_namespace:
        st.warning("No active repository.")
        return

    diff_text = st.text_area(
        "Git diff",
        height=280,
        placeholder="Paste output of `git diff main...feature-branch` here…",
        label_visibility="collapsed",
    )

    if st.button("Review PR", use_container_width=True) and diff_text.strip():
        with st.spinner("Reviewing diff with codebase context…"):
            r = requests.post(
                f"{BACKEND_URL}/review",
                json={"diff": diff_text, "namespace": st.session_state.active_namespace},
                timeout=180,
            )

        if r.ok:
            data = r.json()
            risk = data.get("risk_score", "Medium")

            # Risk score badge at top
            st.markdown(
                f"### Risk Score: &nbsp; {risk_badge(risk)}",
                unsafe_allow_html=True,
            )
            st.markdown(data.get("risk_justification", ""))

            tab_summary, tab_risks, tab_tests, tab_symbols = st.tabs(
                ["Summary", "Risks", "Suggested Tests", "Affected Symbols"]
            )

            with tab_summary:
                st.markdown(data.get("summary", ""))

            with tab_risks:
                st.markdown(data.get("risks", ""))

            with tab_tests:
                st.markdown(data.get("suggested_tests", ""))

            with tab_symbols:
                syms = data.get("affected_symbols", [])
                if syms:
                    for s in syms:
                        st.markdown(f"- `{s}`")
                else:
                    st.markdown("No specific symbols detected.")

            # Download button
            st.download_button(
                "⬇️ Export Review as Markdown",
                data=data.get("markdown", ""),
                file_name="pr_review.md",
                mime="text/markdown",
            )
        else:
            st.error(f"Error: {r.text}")


# ===========================================================================
# PAGE: Architecture
# ===========================================================================

def page_architecture():
    st.title("Architecture Diagram")
    st.markdown("Auto-generated from static import analysis of the indexed repo.")

    if not st.session_state.active_namespace:
        st.warning("No active repository.")
        return

    if st.button("Generate Diagram", use_container_width=True):
        with st.spinner("Analyzing imports…"):
            r = requests.get(
                f"{BACKEND_URL}/diagram",
                params={"namespace": st.session_state.active_namespace},
                timeout=30,
            )

        if r.ok:
            data = r.json()
            mermaid = data.get("mermaid", "")
            table = data.get("dependency_table", [])

            # Count nodes to set height
            node_count = mermaid.count("-->") + 5
            height = max(400, min(node_count * 60, 1200))

            # Render via Mermaid.js CDN
            mermaid_html = f"""
            <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
            <script>mermaid.initialize({{startOnLoad:true, theme:'dark', securityLevel:'loose'}});</script>
            <div class="mermaid" style="background:#1a1a27; padding:20px; border-radius:10px;">
{mermaid}
            </div>
            """
            st.components.v1.html(mermaid_html, height=height, scrolling=True)

            # Download diagram
            md_content = f"```mermaid\n{mermaid}\n```"
            st.download_button(
                "⬇️ Export Diagram as Markdown",
                data=md_content,
                file_name="architecture.md",
                mime="text/markdown",
            )

            # Dependency table
            if table:
                st.markdown("### File Dependency Table")
                st.markdown(
                    "| File | Imports | Imported By |",
                )
                st.markdown("| --- | --- | --- |")
                for row in table:
                    imports_str = ", ".join(f"`{f}`" for f in row["imports"]) or "—"
                    imported_by_str = ", ".join(f"`{f}`" for f in row["imported_by"]) or "—"
                    st.markdown(
                        f"| `{row['file']}` | {imports_str} | {imported_by_str} |"
                    )
        else:
            st.error(f"Error: {r.text}")


# ===========================================================================
# Router
# ===========================================================================

page = st.session_state.page

if page == "Index Repo":
    page_index()
elif page == "Chat":
    page_chat()
elif page == "Change Impact":
    page_impact()
elif page == "PR Review":
    page_pr_review()
elif page == "Architecture":
    page_architecture()
