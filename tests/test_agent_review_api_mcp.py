import asyncio

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from codeograph import agent, llm, review
from codeograph.api import app

DIFF = """diff --git a/shop/pricing.py b/shop/pricing.py
--- a/shop/pricing.py
+++ b/shop/pricing.py
@@ -12,4 +12,4 @@ def apply_discount(price: float, percent: float) -> float:
 def add_tax(price: float) -> float:
     \"\"\"Add GST at TAX_RATE.\"\"\"
-    return round(price * (1 + TAX_RATE), 2)
+    return round(price * (1 + TAX_RATE), 3)
"""


# ------------------------------------------------------------------ agent
def test_agent_without_llm_is_extractive(indexed):
    r = agent.ask("how does the card get charged with retries?", "mini")
    assert "No LLM is configured" in r["answer"] and "charge_card" in r["answer"]
    assert r["citations"][0]["symbol_name"] == "charge_card"
    assert r["intent"] == "question"


def test_agent_classifies_impact_and_symbol(indexed):
    assert agent.ask("what breaks if I change `final_price`?", "mini")["intent"] == "impact"
    assert agent.ask("where is Cart.total defined?", "mini")["intent"] == "symbol"


def test_agent_with_llm_grades_generates_and_verifies(indexed):
    llm.set_model(FakeListChatModel(responses=[
        '{"relevant": [0]}',                                                   # grade
        "`charge_card` retries with exponential backoff [1]; see also [7].",   # generate ([7] does not exist)
    ]))
    r = agent.ask("how are card charges retried?", "mini")
    assert "[1]" in r["answer"] and "[7]" not in r["answer"]
    assert r["invalid_citations"] == [7]
    assert r["citations"] == [r["citations"][0]] and r["citations"][0]["cited"] is True


def test_agent_rewrites_when_nothing_is_relevant(indexed):
    llm.set_model(FakeListChatModel(responses=[
        '{"relevant": []}',                    # grade: nothing useful
        "charge_card retries backoff",         # rewrite
        '{"relevant": [0]}',                   # grade again
        "It retries [1].",                     # generate
    ]))
    r = agent.ask("payment robustness?", "mini")
    assert any(t.startswith("rewrite:") for t in r["trace"])
    assert r["answer"] == "It retries [1]."


def test_agent_remembers_the_thread(indexed):
    llm.set_model(FakeListChatModel(responses=['{"relevant": [0]}', "First [1].", '{"relevant": [0]}', "Second [1]."]))
    a = agent.ask("what does add_tax do?", "mini", thread_id="t1")
    agent.ask("and who calls it?", "mini", thread_id="t1")
    state = agent.GRAPH.get_state({"configurable": {"thread_id": "t1"}}).values
    assert [h["q"] for h in state["history"]] == ["what does add_tax do?", "and who calls it?"]
    assert a["thread_id"] == "t1"


def test_agent_stream_yields_tokens_then_final(indexed):
    events = list(agent.stream("how is tax added?", "mini"))
    assert events[-1][0] == "final" and any(k == "token" for k, _ in events[:-1])


# ------------------------------------------------------------------ review
def test_parse_diff_and_touched_symbols(indexed):
    idx, _ = indexed
    files = review.parse_diff(DIFF)
    assert files[0]["file"] == "shop/pricing.py" and files[0]["ranges"] == [(12, 15)]
    assert files[0]["added_lines"] == 1 and files[0]["removed_lines"] == 1
    assert review.touched_symbols(idx, files) == ["shop/pricing.py::add_tax"]


def test_review_without_llm(indexed):
    idx, _ = indexed
    r = review.review(idx, DIFF)
    assert "add_tax" in r["affected_symbols"]
    assert any(d["node"] == "shop/pricing.py::final_price" for d in r["dependents"])
    assert r["risk_score"] in ("High", "Medium", "Low") and r["model"] == "none"
    assert "# Pull request review" in review.to_markdown(r, DIFF)


def test_review_with_llm(indexed):
    idx, _ = indexed
    llm.set_model(FakeListChatModel(responses=[
        '```json\n{"summary": "Rounds tax to 3 places.", "risks": ["Totals change"], '
        '"suggested_tests": ["test add_tax rounding"], "risk_score": "Medium", "risk_justification": "money"}\n```']))
    r = review.review(idx, DIFF)
    assert r["summary"] == "Rounds tax to 3 places." and r["risks"] == "- Totals change"


def test_parse_json_variants():
    assert llm.parse_json('noise {"a": [1, "}"]} trailing') == {"a": [1, "}"]}
    assert llm.parse_json("```json\n[1, 2]\n```") == [1, 2]


# ------------------------------------------------------------------ API
def test_api_end_to_end(indexed):
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/repos").json()[0]["namespace"] == "mini"
    assert c.get("/index/status", params={"namespace": "mini"}).json()["status"] == "done"
    hits = c.get("/search", params={"namespace": "mini", "q": "exponential backoff", "k": 3}).json()
    assert hits[0]["qualified_name"] == "charge_card"
    sym = c.get("/symbol", params={"namespace": "mini", "name": "Cart.add"}).json()
    assert sym["file_path"] == "shop/cart.py" and "qty must be positive" in sym["raw_code"]
    assert c.get("/symbol", params={"namespace": "mini", "name": "zzz"}).status_code == 404
    assert c.post("/impact", json={"symbol": "add_tax", "namespace": "mini"}).json()["affected"]
    assert "graph LR" in c.get("/diagram", params={"namespace": "mini"}).json()["mermaid"]
    body = c.post("/query", json={"question": "how is tax computed?", "namespace": "mini"}).text
    assert "<!--SOURCES:" in body
    rv = c.post("/review", json={"diff": DIFF, "namespace": "mini"}).json()
    assert rv["markdown"].startswith("# Pull request review")
    assert c.post("/ask", json={"question": "x", "namespace": "nope"}).status_code == 404


def test_api_index_job(repo_copy):
    c = TestClient(app)
    r = c.post("/index", json={"repo_url": str(repo_copy)}).json()
    assert r["status"] == "started"
    from codeograph.api import _pool

    _pool.submit(lambda: None).result()        # wait for the queued job
    import time

    for _ in range(200):
        s = c.get("/index/status", params={"namespace": r["namespace"]}).json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert s["status"] == "done", s
    assert c.post("/index", json={"repo_url": str(repo_copy)}).json()["status"] == "already_indexed"
    assert c.delete(f"/repos/{r['namespace']}").json() == {"deleted": r["namespace"]}


# ------------------------------------------------------------------ MCP
def test_mcp_tools(indexed):
    from mcp import Client

    from codeograph.mcp_server import server

    async def run():
        async with Client(server) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert {"index_repository", "search_code", "find_symbol", "get_file", "impact_analysis",
                    "review_diff", "architecture_diagram", "ask", "list_repositories"} <= names
            res = await client.call_tool("find_symbol", {"repo": "mini", "name": "charge_card"})
            assert not res.is_error and "charge_card" in str(res.content)
            res = await client.call_tool("impact_analysis", {"repo": "mini", "target": "apply_discount"})
            assert "final_price" in str(res.content)
            res = await client.call_tool("get_file", {"repo": "mini", "path": "shop/cart.py", "start_line": 1,
                                                      "end_line": 3})
            assert "The shopping cart" in str(res.content)
            res = await client.call_tool("search_code", {"repo": "nope", "query": "x"})
            assert res.is_error

    asyncio.run(run())
