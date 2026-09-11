"""Build the reasoning LLM. Sarvam only.

MedLink ran a provider chain (Gemini -> Groq -> Cerebras -> Sarvam) while working
out which free tiers were usable. That evaluation is over: Sarvam serves all
three pipeline stages, so there is one vendor, one key, and no switching.

Model choice lives in `config.SARVAM_LLM_MODEL` (override with
`MEDLINK_LLM_MODEL`). `sarvam-105b-conversations` is the multi-turn/voice-tuned
variant and resolves to Sarvam's `/v1` endpoint; the plugin picks the endpoint
from the model name.
"""

from __future__ import annotations

import logging

from livekit.agents import llm

from config import settings

logger = logging.getLogger("medlink.llm")

# Medical guidance: favour consistency over variety.
TEMPERATURE = 0.3


class SarvamNotConfiguredError(RuntimeError):
    """Raised when SARVAM_API_KEY is missing."""


def build_llm() -> llm.LLM:
    if not settings.sarvam_api_key:
        raise SarvamNotConfiguredError(
            "SARVAM_API_KEY is not set. Every MedLink pipeline stage (STT, LLM, "
            "TTS) runs on Sarvam - get a key at https://dashboard.sarvam.ai and "
            "put it in .env.local."
        )

    from livekit.plugins import sarvam

    logger.info("LLM: Sarvam %s", settings.llm_model)
    return sarvam.LLM(
        model=settings.llm_model,
        api_key=settings.sarvam_api_key,
        temperature=TEMPERATURE,
    )
