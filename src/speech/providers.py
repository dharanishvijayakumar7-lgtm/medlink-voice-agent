"""Assemble STT and TTS. Sarvam only.

Both stages are native WebSocket streaming on the same `SARVAM_API_KEY` that
serves the LLM: STT emits interim transcripts while the caller is still talking,
and TTS streams audio back as tokens arrive, so the stages overlap instead of
running as blocking round trips.

The provider-switching layer (LiveKit Inference, Bhashini, a `free_tier_only`
spend guard) was removed once Sarvam was chosen for all three stages - there is
nothing left to switch between.
"""

from __future__ import annotations

import logging

from livekit.agents import stt, tts

from config import DEFAULT_LANGUAGE_CODE, settings

logger = logging.getLogger("medlink.speech")

# Sarvam's STT identifies the language itself, so the pipeline does not have to
# guess which of the six MedLink languages a caller opens in. agent.py retargets
# the TTS to whatever comes back.
SARVAM_STT_LANGUAGE = "auto"


class SarvamNotConfiguredError(RuntimeError):
    """Raised when SARVAM_API_KEY is missing."""


def _require_key() -> None:
    if not settings.sarvam_api_key:
        raise SarvamNotConfiguredError(
            "SARVAM_API_KEY is not set. STT, LLM and TTS all run on Sarvam - get a "
            "key at https://dashboard.sarvam.ai and put it in .env.local."
        )


def build_stt() -> stt.STT:
    """Sarvam realtime STT over WebSocket.

    ``STTRealtime`` pins ``saaras:v3-realtime`` internally and takes no model
    argument - ``settings.sarvam_stt_model`` records it for the log line only.
    Server-side VAD endpointing means no StreamAdapter and no local segmentation.
    """
    _require_key()
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
        # Sarvam's VAD waits 1000ms of silence by default before closing an
        # utterance, and that is paid before LiveKit's endpointing delay starts.
        vad_min_silence_ms=settings.stt_min_silence_ms,
        api_key=settings.sarvam_api_key,
    )


def build_tts() -> tts.TTS:
    """Sarvam Bulbul TTS over WebSocket streaming."""
    _require_key()
    from livekit.plugins import sarvam

    logger.info(
        "TTS: Sarvam %s / %s (%s, %s Hz)",
        settings.sarvam_tts_model,
        settings.sarvam_tts_speaker,
        settings.tts_codec,
        settings.tts_sample_rate,
    )
    return sarvam.TTS(
        # Always English to begin with; the caller can ask for another language
        # mid-call and workflows.base retargets this instance.
        target_language_code=DEFAULT_LANGUAGE_CODE,
        model=settings.sarvam_tts_model,
        speaker=settings.sarvam_tts_speaker,
        # Synthesise above the 8 kHz line rate and let LiveKit downsample,
        # rather than losing detail at the source.
        speech_sample_rate=settings.tts_sample_rate,
        # Raw PCM: the mp3 default was decoded and then re-encoded to G.711 for
        # the phone, compressing the same audio twice.
        output_audio_codec=settings.tts_codec,
        api_key=settings.sarvam_api_key,
    )
