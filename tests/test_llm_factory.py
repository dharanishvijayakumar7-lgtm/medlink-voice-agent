"""Behavior lock for the LLM builder. Sarvam only, no fallback chain."""

import pytest
from livekit.plugins import sarvam

from config import settings
from llm_factory import SarvamNotConfiguredError, build_llm


def test_missing_key_fails_loudly(monkeypatch):
    """Better a clear startup error than a call that dies on the first turn."""
    monkeypatch.setattr(settings, "sarvam_api_key", "")
    with pytest.raises(SarvamNotConfiguredError, match="SARVAM_API_KEY"):
        build_llm()


def test_builds_a_plain_sarvam_llm(monkeypatch):
    """No FallbackAdapter: one provider, so nothing to switch between."""
    monkeypatch.setattr(settings, "sarvam_api_key", "test-key")
    built = build_llm()
    assert isinstance(built, sarvam.LLM)
    assert built.model == settings.llm_model


def test_configured_model_is_unchanged():
    """The evaluated-and-chosen model IDs must not drift during cleanup."""
    from config import SARVAM_LLM_MODEL, SARVAM_STT_MODEL, SARVAM_TTS_MODEL

    assert SARVAM_LLM_MODEL == "sarvam-105b-conversations"
    assert SARVAM_STT_MODEL == "saaras:v3-realtime"
    assert SARVAM_TTS_MODEL == "bulbul:v3"


def test_no_other_provider_is_reachable():
    """Nothing in src/ should import a non-Sarvam model provider any more."""
    import pathlib

    src = pathlib.Path(__file__).resolve().parent.parent / "src"
    offenders = []
    for path in src.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for bad in ("plugins import google", "plugins import groq", "with_cerebras"):
            if bad in text:
                offenders.append(f"{path.name}: {bad}")
    assert not offenders, offenders
