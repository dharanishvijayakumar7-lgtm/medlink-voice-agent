"""Behavior lock for the LLM provider fallback chain.

No network and no real keys: the providers are constructed for real (the plugins
accept any string as an API key at construction time) and only the *shape* of the
chain is asserted - order, which links drop out, and what happens when nothing is
configured.
"""

import pytest
from livekit.agents import llm

from config import settings
from llm_factory import NoLLMProviderConfiguredError, build_llm


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    """Start every test from "nothing configured" so .env.local cannot leak in."""
    for attr in ("gemini_api_key", "groq_api_key", "cerebras_api_key"):
        monkeypatch.setattr(settings, attr, "")


def _keys(monkeypatch, **kwargs):
    for name, value in kwargs.items():
        monkeypatch.setattr(settings, f"{name}_api_key", value)


def test_no_keys_fails_loudly(monkeypatch):
    """Better a clear startup error than a call that dies on the first turn."""
    with pytest.raises(NoLLMProviderConfiguredError, match="GEMINI_API_KEY"):
        build_llm()


def test_all_three_build_a_fallback_chain_in_order(monkeypatch):
    _keys(monkeypatch, gemini="g", groq="q", cerebras="c")
    built = build_llm()
    assert isinstance(built, llm.FallbackAdapter)
    # Gemini primary, then Groq, then Cerebras.
    models = [inner.model for inner in built._llm_instances]
    assert models == [
        settings.gemini_model,
        settings.groq_model,
        settings.cerebras_model,
    ]


def test_single_provider_skips_the_adapter(monkeypatch):
    """One provider has nothing to fall back to - don't let a timeout kill it."""
    _keys(monkeypatch, gemini="g")
    built = build_llm()
    assert not isinstance(built, llm.FallbackAdapter)
    assert built.model == settings.gemini_model


def test_missing_key_drops_out_of_the_chain(monkeypatch):
    """No Gemini key => Groq leads, Cerebras backs it up."""
    _keys(monkeypatch, groq="q", cerebras="c")
    built = build_llm()
    assert isinstance(built, llm.FallbackAdapter)
    models = [inner.model for inner in built._llm_instances]
    assert models == [settings.groq_model, settings.cerebras_model]


def test_chain_honours_the_configured_attempt_timeout(monkeypatch):
    """A slow provider must be abandoned, not just a failing one."""
    _keys(monkeypatch, gemini="g", groq="q")
    monkeypatch.setattr(settings, "llm_attempt_timeout", 2.5)
    built = build_llm()
    assert isinstance(built, llm.FallbackAdapter)
    assert built._attempt_timeout == 2.5


def test_a_broken_provider_does_not_sink_the_chain(monkeypatch):
    """If one plugin refuses to construct, the rest still form a chain."""
    import llm_factory

    def _boom() -> llm.LLM:
        raise RuntimeError("plugin exploded")

    monkeypatch.setattr(
        llm_factory,
        "_CHAIN",
        (
            ("Broken", "gemini_api_key", "gemini_model", _boom),
            ("Groq", "groq_api_key", "groq_model", llm_factory._groq),
            ("Cerebras", "cerebras_api_key", "cerebras_model", llm_factory._cerebras),
        ),
    )
    _keys(monkeypatch, gemini="g", groq="q", cerebras="c")
    built = build_llm()
    assert isinstance(built, llm.FallbackAdapter)
    assert len(built._llm_instances) == 2
