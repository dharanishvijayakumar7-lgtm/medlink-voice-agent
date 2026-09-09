"""Assemble STT and TTS from `settings.speech_provider`.

Free-tier only by default: Bhashini (government ULCA APIs, no cost) is the
intended primary. Until the Bhashini wrapper lands in P1.7, anything not yet
implemented degrades to LiveKit Inference, which is bundled with the LiveKit
Cloud free tier - so the agent always starts and `console` mode always works.

`free_tier_only` (default True) is a hard guard: providers that bill are
refused outright rather than silently costing money.
"""

from __future__ import annotations

import logging

from livekit.agents import inference, stt, tts

from config import settings

logger = logging.getLogger("medlink.speech")

# Providers that bill per use. Blocked while settings.free_tier_only is True.
PAID_PROVIDERS = frozenset({"google", "azure", "sarvam"})

# LiveKit Inference defaults - bundled with the LiveKit Cloud free tier.
# nova-3:multi is the only multilingual option here; Indic quality is mediocre,
# which is exactly why Bhashini becomes the primary in P1.7.
_FALLBACK_STT = "deepgram/nova-3:multi"
_FALLBACK_TTS = "cartesia/sonic-2"


def _check_allowed(provider: str) -> None:
    if provider in PAID_PROVIDERS and settings.free_tier_only:
        raise RuntimeError(
            f"speech_provider={provider!r} bills per use, but MEDLINK_FREE_TIER_ONLY "
            "is set. Set MEDLINK_FREE_TIER_ONLY=false only if you intend to spend money."
        )


def build_stt() -> stt.STT:
    provider = settings.speech_provider.lower()
    _check_allowed(provider)

    if provider == "bhashini":
        if not settings.bhashini_api_key:
            logger.warning(
                "speech_provider=bhashini but BHASHINI_API_KEY is not set - "
                "using LiveKit Inference STT (%s) for now.",
                _FALLBACK_STT,
            )
        else:
            try:
                from speech.bhashini import BhashiniSTT
            except ImportError:
                logger.warning(
                    "Bhashini STT not implemented yet (P1.7) - using %s", _FALLBACK_STT
                )
            else:
                logger.info("STT: Bhashini (free)")
                return BhashiniSTT()

    return inference.STT(model=_FALLBACK_STT)


def build_tts() -> tts.TTS:
    provider = settings.speech_provider.lower()
    _check_allowed(provider)

    if provider == "bhashini":
        if not settings.bhashini_api_key:
            logger.warning(
                "speech_provider=bhashini but BHASHINI_API_KEY is not set - "
                "using LiveKit Inference TTS (%s) for now.",
                _FALLBACK_TTS,
            )
        else:
            try:
                from speech.bhashini import BhashiniTTS
            except ImportError:
                logger.warning(
                    "Bhashini TTS not implemented yet (P1.7) - using %s", _FALLBACK_TTS
                )
            else:
                logger.info("TTS: Bhashini (free)")
                return BhashiniTTS()

    return inference.TTS(model=_FALLBACK_TTS)
