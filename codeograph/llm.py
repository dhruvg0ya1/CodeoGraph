"""The chat model behind answers, grading and reviews - any provider, or none.

    CODEOGRAPH_LLM=gemini   GOOGLE_API_KEY (free tier works)            GEMINI_MODEL: one model or a chain, best first
                            (default gemini-3.5-flash-lite, then 3.1-flash-lite, 3.5-flash, 2.5-flash - each has
                            its own free daily quota, so the next one answers when one runs out)
    CODEOGRAPH_LLM=openai   OPENAI_BASE_URL + OPENAI_API_KEY            model OPENAI_MODEL
                            -> OpenAI, or any OpenAI-compatible server: vLLM, Ollama, LM Studio, OpenRouter, Groq
    CODEOGRAPH_LLM=none     no model: search, symbols, impact, diagrams and MCP still work; answers are extractive
    CODEOGRAPH_LLM=auto     (default) gemini if GOOGLE_API_KEY, else openai if OPENAI_API_KEY/OPENAI_BASE_URL, else none
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import re
import time
from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .config import get_settings

_override: BaseChatModel | None = None
_offline: contextvars.ContextVar[bool] = contextvars.ContextVar("codeograph_offline", default=False)


@contextlib.contextmanager
def offline():
    """Run a block as if no LLM were configured (extractive answers) - e.g. when a public demo's quota is used up."""
    token = _offline.set(True)
    try:
        yield
    finally:
        _offline.reset(token)


def set_model(model: BaseChatModel | None) -> None:
    """Inject a model (tests, notebooks). None restores provider selection from the environment."""
    global _override
    _override = model
    _build.cache_clear()


def provider() -> str:
    p = get_settings().llm_provider
    if p != "auto":
        return p
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_BASE_URL"):
        return "openai"
    return "none"


_GONE = re.compile(r"PerDay|no longer available|not found|NOT_FOUND|404|PERMISSION_DENIED", re.I)


class FallbackChat(BaseChatModel):
    """Several chat models tried in order - the next one answers when one fails.

    Free API tiers give each model its own small daily quota (Gemini: 20 requests a day per model on some models),
    so a chain of models keeps a demo or an eval answering long after the first quota is gone. A model that is out of
    its daily quota or retired is skipped for `rest_seconds`; any other error (a 503 "high demand", a timeout) only
    moves that one call on to the next model. `used` counts which model answered, for honest labels in evals.
    """

    models: list[BaseChatModel]
    names: list[str]
    rest_seconds: float = 1800.0
    resting: dict[str, float] = {}
    used: dict[str, int] = {}

    @property
    def _llm_type(self) -> str:
        return "codeograph-fallback"

    @property
    def model_name(self) -> str:
        if self.used:
            return "+".join(n for n, _ in sorted(self.used.items(), key=lambda kv: -kv[1]))
        return self.names[0]

    def _live(self) -> list[tuple[str, BaseChatModel]]:
        now = time.time()
        live = [(n, m) for n, m in zip(self.names, self.models, strict=True) if self.resting.get(n, 0) <= now]
        return live or list(zip(self.names, self.models, strict=True))           # all resting: try them all again

    def _failed(self, name: str, err: Exception) -> None:
        if _GONE.search(f"{type(err).__name__} {err}"):
            self.resting[name] = time.time() + self.rest_seconds

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last: Exception | None = None
        for name, m in self._live():
            try:
                out = m._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                self.used[name] = self.used.get(name, 0) + 1
                return out
            except Exception as e:  # noqa: BLE001 - the next model takes the call
                self._failed(name, e)
                last = e
        raise last or RuntimeError("no model in the chain answered")

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        last: Exception | None = None
        for name, m in self._live():
            started = False
            try:
                for chunk in m._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
                    started = True
                    yield chunk
                self.used[name] = self.used.get(name, 0) + 1
                return
            except Exception as e:  # noqa: BLE001
                if started:                                            # half an answer is out: cannot switch now
                    raise
                self._failed(name, e)
                last = e
        raise last or RuntimeError("no model in the chain answered")


def gemini_models(spec: str) -> list[str]:
    """GEMINI_MODEL may name one model or a comma-separated chain, best first."""
    return [m.strip() for m in spec.split(",") if m.strip()]


@lru_cache(maxsize=4)
def _build(name: str, model: str, base_url: str | None) -> BaseChatModel | None:
    if name == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        names = gemini_models(model)
        chats = [ChatGoogleGenerativeAI(model=n, google_api_key=key, temperature=0, max_retries=1, timeout=90)
                 for n in names]
        return chats[0] if len(chats) == 1 else FallbackChat(models=chats, names=names)
    if name == "openai":
        from langchain_openai import ChatOpenAI

        key = os.environ.get("OPENAI_API_KEY", "not-needed")
        return ChatOpenAI(model=model, base_url=base_url, api_key=key or "not-needed", temperature=0,
                          max_retries=4, timeout=120)
    return None


def get_model() -> BaseChatModel | None:
    if _offline.get():
        return None
    if _override is not None:
        return _override
    s = get_settings()
    name = provider()
    if name == "gemini":
        return _build("gemini", s.gemini_model, None)
    if name == "openai":
        return _build("openai", s.openai_model, s.openai_base_url)
    return None


def model_label() -> str:
    m = get_model()
    if m is None:
        return "none"
    return f"{provider()}:{getattr(m, 'model_name', None) or getattr(m, 'model', None) or type(m).__name__}"


def text_of(msg: Any) -> str:
    content = getattr(msg, "content", msg)
    if isinstance(content, list):                       # some providers return content blocks
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def ask(system: str, user: str, model: BaseChatModel | None = None) -> str:
    m = model or get_model()
    if m is None:
        raise RuntimeError("No LLM configured (set GOOGLE_API_KEY, OPENAI_BASE_URL or CODEOGRAPH_LLM).")
    return text_of(m.invoke([SystemMessage(content=system), HumanMessage(content=user)]))


def parse_json(text: str) -> Any:
    """The first JSON object or array in a model reply (fenced or not). Raises ValueError if there is none."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        start = t.find(opener)
        if start < 0:
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return json.loads(t[start: i + 1])
    raise ValueError(f"no JSON in model reply: {text[:200]!r}")


def ask_json(system: str, user: str, default: Any, model: BaseChatModel | None = None) -> Any:
    """ask() for a JSON reply; one retry with the parse error, then `default`."""
    try:
        reply = ask(system, user, model)
        try:
            return parse_json(reply)
        except (ValueError, json.JSONDecodeError) as e:
            reply = ask(system, f"{user}\n\nYour last reply could not be parsed ({e}). Reply with JSON only.", model)
            return parse_json(reply)
    except Exception:  # noqa: BLE001 - grading/review must degrade, not crash the request
        return default
