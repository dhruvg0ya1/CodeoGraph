"""The model chain: a free-tier quota running out moves the call to the next model, never to an error."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from codeograph import llm
from codeograph.llm import FallbackChat


class _Broken(FakeListChatModel):
    message: str = "429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier"

    def _call(self, *a, **k):
        raise RuntimeError(self.message)

    def _stream(self, *a, **k):
        raise RuntimeError(self.message)
        yield  # pragma: no cover


def test_quota_gone_moves_to_next_model_and_rests_the_first():
    chain = FallbackChat(models=[_Broken(responses=["x"]), FakeListChatModel(responses=["second", "again"])],
                         names=["first", "second"])
    assert llm.ask("sys", "q", chain) == "second"
    assert "first" in chain.resting                         # out of its daily quota: skipped from now on
    assert llm.ask("sys", "q", chain) == "again"
    assert chain.used == {"second": 2}
    assert chain.model_name == "second"


def test_transient_error_does_not_rest_the_model():
    busy = _Broken(responses=["x"], message="503 UNAVAILABLE: high demand")
    chain = FallbackChat(models=[busy, FakeListChatModel(responses=["ok"])], names=["busy", "spare"])
    assert llm.ask("sys", "q", chain) == "ok"
    assert chain.resting == {}                               # tried again on the next call


def test_streaming_falls_through_too():
    chain = FallbackChat(models=[_Broken(responses=["x"]), FakeListChatModel(responses=["streamed"])],
                         names=["a", "b"])
    assert "".join(llm.text_of(c) for c in chain.stream("hi")) == "streamed"


def test_all_failing_raises_the_last_error():
    chain = FallbackChat(models=[_Broken(responses=["x"]), _Broken(responses=["y"], message="boom")],
                         names=["a", "b"])
    try:
        llm.ask("sys", "q", chain)
    except RuntimeError as e:
        assert "boom" in str(e)
    else:
        raise AssertionError("expected the last error")


def test_gemini_model_spec_parses_a_chain():
    assert llm.gemini_models("a, b ,,c") == ["a", "b", "c"]
    assert llm.gemini_models("gemini-3.5-flash") == ["gemini-3.5-flash"]
