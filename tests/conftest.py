from __future__ import annotations

import shutil
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "minirepo"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test gets its own index folder, the download-free hash embedder, no reranker and no LLM."""
    monkeypatch.setenv("CODEOGRAPH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("CODEOGRAPH_EMBED_MODEL", "hash")
    monkeypatch.setenv("CODEOGRAPH_RERANK_MODEL", "none")
    monkeypatch.setenv("CODEOGRAPH_LLM", "none")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    from codeograph import llm, store

    llm.set_model(None)
    store._CACHE.clear()
    yield
    llm.set_model(None)


@pytest.fixture
def repo_copy(tmp_path) -> Path:
    dst = tmp_path / "minirepo"
    shutil.copytree(FIXTURE, dst)
    return dst


@pytest.fixture
def indexed(repo_copy):
    from codeograph import indexer, store

    meta = indexer.index_repository(str(repo_copy), namespace="mini")
    return store.load("mini"), meta
