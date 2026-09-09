"""Build the reasoning LLM, free-tier only.

Order of preference:
  1. Gemini via a Google **AI Studio** key (`GOOGLE_API_KEY`) - genuinely free
     tier, no credit card, no Google Cloud billing account. `vertexai=False` is
     passed explicitly so we can never fall through to billed Vertex AI.
  2. LiveKit Inference with an open-weight model - bundled with the LiveKit
     Cloud free tier, so `console` mode still works before a Gemini key exists.

Nothing here may construct a provider that bills while `free_tier_only` is set.
"""

from __future__ import annotations

import logging

from livekit.agents import inference, llm

from config import settings

logger = logging.getLogger("medlink.llm")


def build_llm() -> llm.LLM:
    if settings.google_api_key:
        from livekit.plugins import google

        logger.info("LLM: Gemini %s (AI Studio free tier)", settings.llm_model)
        return google.LLM(
            model=settings.llm_model,
            api_key=settings.google_api_key,
            # Never use Vertex AI - that is the billed Google Cloud path.
            vertexai=False,
            temperature=0.3,  # medical guidance: favour consistency
        )

    logger.warning(
        "GOOGLE_API_KEY not set - falling back to LiveKit Inference (%s). "
        "Get a free Gemini key at https://aistudio.google.com/apikey",
        settings.llm_fallback_model,
    )
    return inference.LLM(model=settings.llm_fallback_model)
