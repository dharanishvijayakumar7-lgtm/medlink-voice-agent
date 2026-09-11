"""Assemble STT and TTS from `settings.speech_provider`.

Two providers:

- ``sarvam`` (the default): Sarvam AI, the same vendor and the same
  ``SARVAM_API_KEY`` that serves the LLM. Both stages are **native WebSocket
  streaming**, which is what keeps a single-vendor pipeline fast: STT emits
  interim transcripts while the caller is still talking, the LLM starts
  generating on the final, and TTS streams audio back as tokens arrive, so the
  three stages overlap instead of running as three blocking round trips.
- ``bhashini``: Government of India ULCA APIs, free, all 22 languages.
  Bhashini's inference is request/response rather than streaming, so its STT is
  wrapped in LiveKit's ``stt.StreamAdapter`` with a local (free) Silero VAD that
  segments the caller's audio into utterances. Selected by setting
  ``speech_provider=bhashini`` and filling the ``BHASHINI_*`` keys.

A Bhashini stage that cannot start degrades to Sarvam so a call still connects.
Sarvam itself fails loudly: without a key there is no pipeline at all, and a
silent downgrade would just hide the misconfiguration until someone noticed the
voice was wrong.

``free_tier_only`` (default True) is a spend guard for providers we have not
funded (``google`` / ``azure``). Sarvam is exempt - it runs on prepaid credits
that were bought for exactly this.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from livekit.agents import stt, tts, vad

from config import DEFAULT_LANGUAGE_CODE, settings

logger = logging.getLogger("medlink.speech")

# Providers that would bill with no budget behind them, and which nothing here
# knows how to construct anyway. Blocked while settings.free_tier_only is True.
# Sarvam is deliberately absent: it is prepaid and it is the default pipeline.
PAID_PROVIDERS = frozenset({"google", "azure"})

# Sarvam STT does its own language identification, so the pipeline does not have
# to guess which of the six MedLink languages a caller will open in.
SARVAM_STT_LANGUAGE = "auto"


class PaidProviderBlockedError(RuntimeError):
    """Raised when an unfunded billing provider is selected under free_tier_only."""


class SarvamNotConfiguredError(RuntimeError):
    """Raised when SARVAM_API_KEY is missing."""


def _check_allowed(provider: str) -> None:
    if provider in PAID_PROVIDERS and settings.free_tier_only:
        raise PaidProviderBlockedError(
            f"speech_provider={provider!r} bills per use and has no budget or wiring "
            "here, but MEDLINK_FREE_TIER_ONLY is set. Use 'sarvam' (prepaid credits) "
            "or 'bhashini' (free)."
        )


def _require_sarvam_key() -> None:
    if not settings.sarvam_api_key:
        raise SarvamNotConfiguredError(
            "SARVAM_API_KEY is not set. STT, LLM and TTS all run on Sarvam - get a "
            "key at https://dashboard.sarvam.ai and put it in .env.local."
        )


@lru_cache(maxsize=1)
def get_vad() -> vad.VAD:
    """Local Silero VAD - runs on-device, no API, no cost."""
    from livekit.plugins import silero

    return silero.VAD.load()


def _bhashini_ready() -> bool:
    return bool(settings.bhashini_api_key)


def _sarvam_stt() -> stt.STT:
    """Sarvam realtime STT over WebSocket.

    ``STTRealtime`` pins ``saaras:v3-realtime`` internally and takes no model
    argument - ``settings.sarvam_stt_model`` records it for the log line only.
    Server-side VAD endpointing means no StreamAdapter and no local segmentation
    on this path.
    """
    _require_sarvam_key()
    from livekit.plugins import sarvam

    logger.info(
        "STT: Sarvam %s (%s, streaming)",
        settings.sarvam_stt_model,
        settings.sarvam_stt_stream_type,
    )
    return sarvam.STTRealtime(
        language=SARVAM_STT_LANGUAGE,
        stream_type=settings.sarvam_stt_stream_type,
        # Keep finals in the language the caller actually spoke; the LLM is
        # multilingual and the reply has to come back in that language.
        mode="transcribe",
        endpointing="vad",
        encoding="linear16",
        # 8 kHz for SIP. Sarvam accepts only 8000 or 16000 here, so passing the
        # telephony rate straight through avoids a resample on the hot path.
        sample_rate=settings.audio_sample_rate,
        api_key=settings.sarvam_api_key,
    )


def _sarvam_tts() -> tts.TTS:
    """Sarvam Bulbul TTS over WebSocket streaming."""
    _require_sarvam_key()
    from livekit.plugins import sarvam

    logger.info(
        "TTS: Sarvam %s / %s", settings.sarvam_tts_model, settings.sarvam_tts_speaker
    )
    return sarvam.TTS(
        target_language_code=DEFAULT_LANGUAGE_CODE,
        model=settings.sarvam_tts_model,
        speaker=settings.sarvam_tts_speaker,
        speech_sample_rate=settings.audio_sample_rate,
        api_key=settings.sarvam_api_key,
    )


def build_stt() -> stt.STT:
    provider = settings.speech_provider.lower()
    _check_allowed(provider)

    if provider == "bhashini":
        if not _bhashini_ready():
            logger.warning(
                "speech_provider=bhashini but BHASHINI_API_KEY is not set - "
                "using Sarvam STT for now.",
            )
        else:
            try:
                from speech.bhashini import BhashiniSTT

                # Bhashini ASR is request/response, so let LiveKit's VAD-driven
                # adapter turn the live audio stream into discrete utterances.
                logger.info("STT: Bhashini via StreamAdapter + Silero VAD (free)")
                return stt.StreamAdapter(stt=BhashiniSTT(), vad=get_vad())
            except Exception:
                logger.exception("could not start Bhashini STT - falling back to Sarvam")

    return _sarvam_stt()


def build_tts() -> tts.TTS:
    provider = settings.speech_provider.lower()
    _check_allowed(provider)

    if provider == "bhashini":
        if not _bhashini_ready():
            logger.warning(
                "speech_provider=bhashini but BHASHINI_API_KEY is not set - "
                "using Sarvam TTS for now.",
            )
        else:
            try:
                from speech.bhashini import BhashiniTTS

                logger.info("TTS: Bhashini (free)")
                return BhashiniTTS(sample_rate=settings.audio_sample_rate)
            except Exception:
                logger.exception("could not start Bhashini TTS - falling back to Sarvam")

    return _sarvam_tts()
