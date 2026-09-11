"""Build the reasoning LLM as an ordered provider fallback chain.

Speech (STT/TTS) stays on Sarvam, but the LLM is the chattiest stage - one call
per caller turn, plus speculative ones - so it runs on separate providers. That
keeps Sarvam's rate limit for the audio path and removes the single-vendor
dependency for reasoning.

Order: **Gemini -> Groq -> Cerebras**. LiveKit's `llm.FallbackAdapter` does the
work: it moves to the next provider when one returns a rate limit or an API
error, *and* when one is simply too slow to produce a first token
(`attempt_timeout`, default 5s). Failover happens mid-call with no restart, and
the adapter marks a failed provider unhealthy so it is not retried on every turn.

A provider with no key is dropped from the chain at startup rather than failing
at call time, so one key is enough to run. Set them in `.env.local`:

    GEMINI_API_KEY=...
    GROQ_API_KEY=...
    CEREBRAS_API_KEY=...

Nothing here hardcodes a key; every value comes from `config.settings`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from livekit.agents import llm

from config import settings

logger = logging.getLogger("medlink.llm")

# Medical guidance: favour consistency over variety, on every provider.
TEMPERATURE = 0.3


class NoLLMProviderConfiguredError(RuntimeError):
    """Raised when not one of the chain's API keys is set."""


def _gemini() -> llm.LLM:
    from livekit.plugins import google

    return google.LLM(
        model=settings.gemini_model,
        api_key=settings.gemini_api_key,
        # Never use Vertex AI - that is the billed Google Cloud path. This is the
        # AI Studio free tier.
        vertexai=False,
        temperature=TEMPERATURE,
    )


def _groq() -> llm.LLM:
    from livekit.plugins import groq

    return groq.LLM(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=TEMPERATURE,
    )


def _cerebras() -> llm.LLM:
    from livekit.plugins import openai

    # Cerebras is OpenAI-compatible, so it reuses the openai plugin that the
    # Sarvam LLM already pulls in - no extra dependency.
    return openai.LLM.with_cerebras(
        model=settings.cerebras_model,
        api_key=settings.cerebras_api_key,
        temperature=TEMPERATURE,
    )


# (display name, settings attr holding the key, settings attr holding the model,
# builder). Order is the fallback order - primary first.
_CHAIN: tuple[tuple[str, str, str, Callable[[], llm.LLM]], ...] = (
    ("Gemini", "gemini_api_key", "gemini_model", _gemini),
    ("Groq", "groq_api_key", "groq_model", _groq),
    ("Cerebras", "cerebras_api_key", "cerebras_model", _cerebras),
)


def build_llm() -> llm.LLM:
    chain: list[llm.LLM] = []
    labels: list[str] = []

    for name, key_attr, model_attr, build in _CHAIN:
        if not getattr(settings, key_attr):
            logger.info("LLM: %s skipped - no key set", name)
            continue
        try:
            chain.append(build())
        except Exception:
            # A plugin that will not construct must not take the whole chain
            # down; the next provider is still worth trying.
            logger.exception("LLM: %s failed to initialise - skipping", name)
            continue
        labels.append(f"{name} {getattr(settings, model_attr)}")

    if not chain:
        raise NoLLMProviderConfiguredError(
            "No LLM provider is configured. Set at least one of GEMINI_API_KEY, "
            "GROQ_API_KEY or CEREBRAS_API_KEY in .env.local (Gemini is the "
            "primary; the others are fallbacks)."
        )

    if len(chain) == 1:
        # Nothing to fall back to - skip the adapter so a slow-but-working
        # provider is not killed by attempt_timeout with no alternative.
        logger.warning("LLM: %s only - no fallback available", labels[0])
        return chain[0]

    logger.info("LLM: %s", " -> ".join(labels))
    return llm.FallbackAdapter(chain, attempt_timeout=settings.llm_attempt_timeout)
