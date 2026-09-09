"""Assemble STT and TTS from `settings.speech_provider`.

Free-tier only by default: Bhashini (government ULCA APIs, no cost) is the
primary. Bhashini's inference is request/response rather than streaming, so its
STT is wrapped in LiveKit's ``stt.StreamAdapter`` with a local Silero VAD, which
segments the caller's audio into utterances and calls Bhashini per utterance.
Silero runs on-device and is also free.

If Bhashini credentials are missing or the module cannot load, we degrade to
LiveKit Inference (bundled with the LiveKit Cloud free tier) so the agent always
starts and `console` mode always works.

``free_tier_only`` (default True) is a hard guard: providers that bill are
refused outright rather than silently costing money.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from livekit.agents import inference, stt, tts, vad

from config import settings

logger = logging.getLogger("medlink.speech")

# Providers that bill per use. Blocked while settings.free_tier_only is True.
PAID_PROVIDERS = frozenset({"google", "azure", "sarvam"})

# LiveKit Inference defaults - bundled with the LiveKit Cloud free tier.
# nova-3:multi is the only multilingual option here; Indic quality is mediocre,
# which is exactly why Bhashini is the primary.
_FALLBACK_STT = "deepgram/nova-3:multi"
_FALLBACK_TTS = "cartesia/sonic-2"


class PaidProviderBlockedError(RuntimeError):
    """Raised when a billing provider is selected under free_tier_only."""


def _check_allowed(provider: str) -> None:
    if provider in PAID_PROVIDERS and settings.free_tier_only:
        raise PaidProviderBlockedError(
            f"speech_provider={provider!r} bills per use, but MEDLINK_FREE_TIER_ONLY "
            "is set. Set MEDLINK_FREE_TIER_ONLY=false only if you intend to spend money."
        )


@lru_cache(maxsize=1)
def get_vad() -> vad.VAD:
    """Local Silero VAD - runs on-device, no API, no cost."""
    from livekit.plugins import silero

    return silero.VAD.load()


def _bhashini_ready() -> bool:
    return bool(settings.bhashini_api_key)


def build_stt() -> stt.STT:
    provider = settings.speech_provider.lower()
    _check_allowed(provider)

    if provider == "bhashini":
        if not _bhashini_ready():
            logger.warning(
                "speech_provider=bhashini but BHASHINI_API_KEY is not set - "
                "using LiveKit Inference STT (%s) for now.",
                _FALLBACK_STT,
            )
        else:
            try:
                from speech.bhashini import BhashiniSTT

                # Bhashini ASR is request/response, so let LiveKit's VAD-driven
                # adapter turn the live audio stream into discrete utterances.
                logger.info("STT: Bhashini via StreamAdapter + Silero VAD (free)")
                return stt.StreamAdapter(stt=BhashiniSTT(), vad=get_vad())
            except Exception:
                logger.exception(
                    "could not start Bhashini STT - falling back to %s", _FALLBACK_STT
                )

    return inference.STT(model=_FALLBACK_STT)


def build_tts() -> tts.TTS:
    provider = settings.speech_provider.lower()
    _check_allowed(provider)

    if provider == "bhashini":
        if not _bhashini_ready():
            logger.warning(
                "speech_provider=bhashini but BHASHINI_API_KEY is not set - "
                "using LiveKit Inference TTS (%s) for now.",
                _FALLBACK_TTS,
            )
        else:
            try:
                from speech.bhashini import BhashiniTTS

                logger.info("TTS: Bhashini (free)")
                return BhashiniTTS(sample_rate=settings.audio_sample_rate)
            except Exception:
                logger.exception(
                    "could not start Bhashini TTS - falling back to %s", _FALLBACK_TTS
                )

    return inference.TTS(model=_FALLBACK_TTS)
