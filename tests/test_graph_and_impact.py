from codeograph import depgraph, diagram, impact


def _edges(g, kind):
    return {(u, v) for u, v, d in g.edges(data=True) if d.get("type") == kind}


def test_python_imports_resolve_to_files(indexed):
    idx, _ = indexed
    imports = _edges(idx.graph, "imports")
    assert ("shop/cart.py", "shop/pricing.py") in imports                  # relative import
    assert ("shop/payments/gateway.py", "shop/cart.py") in imports         # absolute import


def test_calls_resolved_through_imports_and_self(indexed):
    idx, _ = indexed
    calls = _edges(idx.graph, "calls")
    assert ("shop/cart.py::Cart.total", "shop/pricing.py::final_price") in calls
    assert ("shop/cart.py::Cart.total", "shop/cart.py::Cart.subtotal") in calls   # self.subtotal()
    assert ("shop/pricing.py::final_price", "shop/pricing.py::apply_discount") in calls
    assert ("shop/payments/gateway.py::checkout", "shop/payments/gateway.py::charge_card") in calls


def test_typescript_imports_and_calls(indexed):
    idx, _ = indexed
    assert ("web/src/api.ts", "web/src/format.ts") in _edges(idx.graph, "imports")
    assert ("web/src/api.ts::loadTotal", "web/src/format.ts::formatMoney") in _edges(idx.graph, "calls")


def test_dependents_are_nearest_first(indexed):
    idx, _ = indexed
    nodes = depgraph.find_nodes(idx.graph, "apply_discount")
    assert nodes == ["shop/pricing.py::apply_discount"]
    deps = depgraph.dependents(idx.graph, nodes, max_hops=3)
    hops = {n: h for n, h, _ in deps}
    assert hops["shop/pricing.py::final_price"] == 1
    assert hops["shop/cart.py::Cart.total"] == 2
    assert hops["shop/payments/gateway.py::checkout"] == 3


def test_impact_risk_and_summary(indexed):
    idx, _ = indexed
    rep = impact.analyze(idx, "final_price", max_hops=2)
    first = rep["affected"][0]
    assert first["node"] == "shop/cart.py::Cart.total" and first["risk"] == "High" and first["hops"] == 1
    assert "dependent" in rep["summary"]
    assert impact.analyze(idx, "does_not_exist")["affected"] == []


def test_find_nodes_variants(indexed):
    idx, _ = indexed
    g = idx.graph
    assert depgraph.find_nodes(g, "Cart.total") == ["shop/cart.py::Cart.total"]
    assert depgraph.find_nodes(g, "shop/cart.py") == ["shop/cart.py"]
    assert depgraph.find_nodes(g, "`charge_card`") == ["shop/payments/gateway.py::charge_card"]


def test_graph_round_trip(indexed):
    idx, _ = indexed
    g2 = depgraph.from_json(depgraph.to_json(idx.graph))
    assert set(g2.edges) == set(idx.graph.edges)


def test_diagram_mermaid(indexed):
    idx, _ = indexed
    d = diagram.generate(idx)
    assert d["mermaid"].startswith("graph LR") and "-->" in d["mermaid"]
    assert any(r["full_path"] == "shop/cart.py" and "shop/pricing.py" in r["imports"] for r in d["dependency_table"])
