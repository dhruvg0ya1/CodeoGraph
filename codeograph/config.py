"""Settings, read from the environment (and a .env file) every time they are asked for, so tests and long-running
servers pick up changes without a restart."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - python-dotenv is a dependency, this only guards odd installs
    pass


@dataclass(frozen=True)
class Settings:
    home: Path
    embed_model: str
    rerank_model: str | None
    fusion: str
    dense_weight: float
    candidates: int
    llm_provider: str
    gemini_model: str
    openai_model: str
    openai_base_url: str | None
    qdrant_url: str | None
    max_context_tokens: int

    @property
    def index_root(self) -> Path:
        return self.home / "indexes"


def _none(v: str | None) -> str | None:
    return None if v is None or v.strip().lower() in ("", "none", "off", "0", "false") else v.strip()


def get_settings() -> Settings:
    env = os.environ.get
    return Settings(
        home=Path(env("CODEOGRAPH_HOME", str(Path.home() / ".codeograph"))).expanduser(),
        # defaults chosen by evals/retrieval_eval.py (README "Results"): weighted fusion beat RRF on MRR, and the
        # MS MARCO cross-encoder LOWERED Recall@5 (0.93 -> 0.80) at ~2.5 s a query, so reranking is opt-in
        embed_model=env("CODEOGRAPH_EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        rerank_model=_none(env("CODEOGRAPH_RERANK_MODEL", "none")),
        fusion=env("CODEOGRAPH_FUSION", "weighted"),
        dense_weight=float(env("CODEOGRAPH_DENSE_WEIGHT", "0.6")),
        candidates=int(env("CODEOGRAPH_CANDIDATES", "30")),
        llm_provider=env("CODEOGRAPH_LLM", "auto").lower(),
        gemini_model=env("GEMINI_MODEL", "gemini-2.5-flash"),
        openai_model=env("OPENAI_MODEL", "gpt-4.1-mini"),
        openai_base_url=_none(env("OPENAI_BASE_URL")),
        qdrant_url=_none(env("QDRANT_URL")),
        max_context_tokens=int(env("CODEOGRAPH_MAX_CONTEXT_TOKENS", "6000")),
    )
