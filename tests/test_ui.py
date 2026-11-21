"""The Streamlit app runs end to end on an indexed repository: every tab renders, and asking, searching, impact
and review all work without an exception."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

UI = str(Path(__file__).resolve().parent.parent / "ui" / "app.py")


def test_app_end_to_end(indexed, monkeypatch):
    monkeypatch.setenv("CODEOGRAPH_NO_DEMOS", "1")
    at = AppTest.from_file(UI, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert any("mini" in m.value for m in at.markdown), "the hero shows the indexed repository"

    at.text_input(key="q_mini").set_value("retry with exponential backoff").run()
    assert not at.exception, at.exception
    assert any("charge_card" in m.value for m in at.markdown)

    at.text_input(key="imp_mini").set_value("apply_discount").run()
    assert not at.exception, at.exception

    at.text_input(key="sym_mini").set_value("Cart.total").run()
    assert not at.exception, at.exception

    at.chat_input[0].set_value("how are card charges retried?").run()
    assert not at.exception, at.exception
    assert any("charge_card" in m.value for m in at.markdown)

    at.button[[b.label for b in at.button].index("Load a sample diff for this repository")].click().run()
    assert not at.exception, at.exception
