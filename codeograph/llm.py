"""The chat model behind answers, grading and reviews - any provider, or none.

    CODEOGRAPH_LLM=gemini   GOOGLE_API_KEY (free tier works)            model GEMINI_MODEL
    CODEOGRAPH_LLM=openai   OPENAI_BASE_URL + OPENAI_API_KEY            model OPENAI_MODEL
                            -> OpenAI, or any OpenAI-compatible server: vLLM, Ollama, LM Studio, OpenRouter, Groq
    CODEOGRAPH_LLM=github   GITHUB_TOKEN (GitHub Models, free tier)     model GITHUB_MODEL
    CODEOGRAPH_LLM=none     no model: search, symbols, impact, diagrams and MCP still work; answers are extractive
    CODEOGRAPH_LLM=auto     (default) gemini if GOOGLE_API_KEY, else openai if OPENAI_API_KEY/OPENAI_BASE_URL, else none
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .config import get_settings

_override: BaseChatModel | None = None


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


@lru_cache(maxsize=4)
def _build(name: str, model: str, base_url: str | None) -> BaseChatModel | None:
    if name == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        return ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0)
    if name in ("openai", "github"):
        from langchain_openai import ChatOpenAI

        key = os.environ.get("GITHUB_TOKEN") if name == "github" else os.environ.get("OPENAI_API_KEY", "not-needed")
        return ChatOpenAI(model=model, base_url=base_url, api_key=key or "not-needed", temperature=0,
                          max_retries=4, timeout=120)
    return None


def get_model() -> BaseChatModel | None:
    if _override is not None:
        return _override
    s = get_settings()
    name = provider()
    if name == "gemini":
        return _build("gemini", s.gemini_model, None)
    if name == "openai":
        return _build("openai", s.openai_model, s.openai_base_url)
    if name == "github":
        return _build("github", s.github_model, "https://models.github.ai/inference")
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
