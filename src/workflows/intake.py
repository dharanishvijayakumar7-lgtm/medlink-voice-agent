"""Intake: greet, find out who is unwell and what is wrong, then hand off.

Deliberately narrow. It captures the chief complaint plus basic patient context
and gets out of the way - all the clinical questioning happens in triage.
"""

from __future__ import annotations

import logging

from livekit.agents import RunContext, function_tool

from config import DEFAULT_LANGUAGE_CODE
from knowledge.triage_kb import apply_to_session
from session_state import MedLinkUserData
from workflows.base import SHARED_STYLE, MedLinkAgent

logger = logging.getLogger("medlink.workflow")

# Fixed opening line - spoken verbatim, no LLM, so the call always starts the
# same way and with zero time-to-first-word. Extend per language as TTS voices
# are added in P1.7.
GREETINGS: dict[str, str] = {
    "en-IN": (
        "Hello, this is MedLink, a health helpline. I am not a doctor, but I can "
        "listen and help you decide what to do next. You can speak in Hindi, Tamil, "
        "Telugu, Kannada, Malayalam or English. Please tell me, what is troubling you?"
    ),
    "hi-IN": (
        "नमस्ते, मैं मेडलिंक हूँ, एक स्वास्थ्य हेल्पलाइन। मैं डॉक्टर नहीं हूँ, "
        "लेकिन आपकी बात सुनकर बता सकता हूँ कि आगे क्या करना चाहिए। "
        "बताइए, आपको क्या तकलीफ हो रही है?"
    ),
    "ta-IN": (
        "வணக்கம், நான் மெட்லிங்க், ஒரு சுகாதார உதவி எண். நான் மருத்துவர் அல்ல, "
        "ஆனால் நீங்கள் சொல்வதைக் கேட்டு அடுத்து என்ன செய்வது என்று சொல்ல முடியும். "
        "சொல்லுங்கள், உங்களுக்கு என்ன பிரச்சினை?"
    ),
    "te-IN": (
        "నమస్కారం, నేను మెడ్‌లింక్, ఒక ఆరోగ్య సహాయ లైన్. నేను వైద్యుడిని కాదు, "
        "కానీ మీరు చెప్పేది విని తర్వాత ఏమి చేయాలో చెప్పగలను. "
        "చెప్పండి, మీకు ఏమి ఇబ్బంది?"
    ),
    "kn-IN": (
        "ನಮಸ್ಕಾರ, ನಾನು ಮೆಡ್‌ಲಿಂಕ್, ಒಂದು ಆರೋಗ್ಯ ಸಹಾಯವಾಣಿ. ನಾನು ವೈದ್ಯನಲ್ಲ, "
        "ಆದರೆ ನೀವು ಹೇಳುವುದನ್ನು ಕೇಳಿ ಮುಂದೆ ಏನು ಮಾಡಬೇಕೆಂದು ಹೇಳಬಲ್ಲೆ. "
        "ಹೇಳಿ, ನಿಮಗೆ ಏನು ತೊಂದರೆ?"
    ),
    "ml-IN": (
        "നമസ്കാരം, ഞാൻ മെഡ്‌ലിങ്ക്, ഒരു ആരോഗ്യ ഹെൽപ്പ്‌ലൈൻ. ഞാൻ ഡോക്ടറല്ല, "
        "പക്ഷേ നിങ്ങൾ പറയുന്നത് കേട്ട് അടുത്തത് എന്ത് ചെയ്യണമെന്ന് പറയാൻ കഴിയും. "
        "പറയൂ, നിങ്ങൾക്ക് എന്താണ് ബുദ്ധിമുട്ട്?"
    ),
}

INSTRUCTIONS = f"""\
You are MedLink, a health helpline assistant taking a phone call.

Your ONLY job right now is to find out:
1. What is troubling the caller (their main complaint, in their own words).
2. Who it is for - the caller themselves, or someone else such as a child.
3. Roughly how old that person is.

{SHARED_STYLE}

# Rules
- The caller has already been greeted. Do NOT greet them again.
- Let them describe the problem in their own words first. Do not interrupt.
- Ask at most TWO short questions here (who it is for, and their age) and only
  if you do not already know.
- As soon as you know the complaint, call `record_complaint`. Do not try to
  diagnose, reassure at length, or suggest any medicine - another part of the
  system does that next.
"""


class IntakeAgent(MedLinkAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(instructions=INSTRUCTIONS, **kwargs)

    async def on_enter(self) -> None:
        data: MedLinkUserData = self.data
        greeting = GREETINGS.get(data.language) or GREETINGS[DEFAULT_LANGUAGE_CODE]
        # A returning caller's stored language means the greeting is not English,
        # so the voice has to be retargeted before it speaks - otherwise Bulbul
        # reads Tamil script with English phonetics. New callers stay on the
        # default until their first utterance is transcribed, at which point
        # agent.py's `user_input_transcribed` hook takes over.
        tts = self.session.tts
        if data.language != DEFAULT_LANGUAGE_CODE and hasattr(tts, "update_options"):
            try:
                tts.update_options(target_language_code=data.language)
            except Exception:
                logger.exception("could not set greeting language %s", data.language)
        # say() not generate_reply(): deterministic wording, no LLM round trip.
        await self.session.say(greeting)

    @function_tool
    async def record_complaint(
        self,
        context: RunContext[MedLinkUserData],
        complaint: str,
        patient_age_years: int | None = None,
        is_for_child: bool = False,
    ):
        """Record the caller's main health complaint and who it is about.

        Call this as soon as you understand what is wrong. It moves the call on
        to the questioning stage.

        Args:
            complaint: The main problem in the caller's own words, in English.
            patient_age_years: Age of the person who is unwell, if known.
            is_for_child: True if the caller is asking on behalf of a child.
        """
        data = context.userdata
        data.chief_complaint = complaint
        data.patient.is_for_child = is_for_child
        if patient_age_years is not None:
            data.patient.age_years = patient_age_years

        # Pull this presentation's follow-up questions, allowed OTC categories
        # and referral criteria from the curated triage KB.
        entry = apply_to_session(data, complaint)

        logger.info(
            "intake complete",
            extra={
                "call_id": data.call_id,
                "complaint": complaint,
                "triage_entry": entry.id if entry else None,
            },
        )

        from workflows.triage import TriageAgent

        return (
            TriageAgent(chat_ctx=self.chat_ctx),
            "Thank you. Let me ask a few quick questions.",
        )
