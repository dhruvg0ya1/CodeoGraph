"""The Streamlit UI runs end to end against its in-process API: every page renders with an indexed repository."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

UI = str(Path(__file__).resolve().parent.parent / "ui" / "app.py")


def test_every_page_renders(indexed, monkeypatch):
    monkeypatch.delenv("BACKEND_URL", raising=False)
    at = AppTest.from_file(UI, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    for page in ["Index Repo", "Chat", "Change Impact", "PR Review", "Architecture"]:
        at.session_state["page"] = page
        at.run()
        assert not at.exception, (page, at.exception)
