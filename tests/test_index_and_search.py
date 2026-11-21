import subprocess

from codeograph import indexer, search, store
from codeograph.search import tokenize


def test_tokenize_splits_identifiers():
    toks = tokenize("HTTPAdapter.send_request getUserById")
    for t in ("httpadapter", "http", "adapter", "send_request", "send", "request", "getuserbyid", "get", "user"):
        assert t in toks


def test_namespace_for():
    assert indexer.namespace_for("https://github.com/psf/requests") == "psf-requests"
    assert indexer.namespace_for("https://github.com/psf/requests.git") == "psf-requests"
    assert indexer.namespace_for("git@github.com:Owner/My.Repo.git") == "owner-my-repo"


def test_index_meta(indexed):
    idx, meta = indexed
    assert meta["status"] == "indexed"
    assert meta["file_count"] == 8 and meta["chunk_count"] == len(idx.chunks)
    assert {"python", "typescript", "go"} <= set(meta["languages"])
    assert idx.vectors.shape[0] == len(idx.chunks)
    assert store.list_indexes()[0]["namespace"] == "mini"


def test_reindex_is_incremental(repo_copy):
    first = indexer.index_repository(str(repo_copy), namespace="mini")
    again = indexer.index_repository(str(repo_copy), namespace="mini", force=True)
    assert again["embedded"] == 0 and again["reused"] == first["chunk_count"]
    p = repo_copy / "shop" / "pricing.py"
    p.write_text(p.read_text().replace("round(price * (1 + TAX_RATE), 2)", "round(price * (1 + TAX_RATE), 3)"))
    third = indexer.index_repository(str(repo_copy), namespace="mini", force=True)
    assert 1 <= third["embedded"] <= 2                        # add_tax (+ the module chunk at most)


def test_unchanged_commit_is_skipped(repo_copy):
    subprocess.run(["git", "init", "-q"], cwd=repo_copy, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"], cwd=repo_copy, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"], cwd=repo_copy,
                   check=True)
    indexer.index_repository(str(repo_copy), namespace="mini")
    assert indexer.index_repository(str(repo_copy), namespace="mini")["status"] == "already_indexed"


def test_bm25_and_hybrid_find_the_right_function(indexed):
    idx, _ = indexed
    for mode in ("bm25", "hybrid"):
        hits = search.search(idx, "retry with exponential backoff when charging a card", k=3, mode=mode)
        assert hits[0].chunk.qualified_name == "charge_card", mode


def test_filters(indexed):
    idx, _ = indexed
    hits = search.search(idx, "price", k=10, file="shop/pricing.py")
    assert hits and all(h.chunk.file_path == "shop/pricing.py" for h in hits)
    hits = search.search(idx, "total", k=10, kind="method")
    assert hits and all(h.chunk.symbol_type == "method" for h in hits)


def test_exact_symbol_boost(indexed):
    idx, _ = indexed
    assert search.search(idx, "what does `add_tax` do", k=3)[0].chunk.qualified_name == "add_tax"


def test_find_symbol(indexed):
    idx, _ = indexed
    assert search.find_symbol(idx, "Cart.total")[0].file_path == "shop/cart.py"
    assert search.find_symbol(idx, "formatMoney")[0].language == "typescript"
    assert search.find_symbol(idx, "CHARGE_CARD()")[0].qualified_name == "charge_card"
    assert search.find_symbol(idx, "nope") == []


def test_context_block_respects_budget(indexed):
    idx, _ = indexed
    hits = search.search(idx, "cart", k=8)
    block = search.context_block(hits, max_tokens=200)
    assert block.startswith("[1] ") and len(block) < 200 * 4 + 400
