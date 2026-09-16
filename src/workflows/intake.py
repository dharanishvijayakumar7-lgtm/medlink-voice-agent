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

# Placeholder strings a model emits for "no name given" - never a real name.
_NO_NAME = {"null", "none", "unknown", "n/a", "na", "not given", "not provided"}

# Fixed opening line - spoken verbatim, no LLM, so the call always starts the
# same way and with zero time-to-first-word. Kept short and warm: the English
# version used to list all six languages and took ~15 seconds to say.
# The TTS voice is female, so gendered languages use feminine first-person forms
# (Hindi "सकती", not "सकता"), and Telugu/Kannada use the neutral loanword
# "doctor" instead of the masculine native noun.
GREETINGS: dict[str, str] = {
    "en-IN": (
        "Hello, this is MedLink. I'm not a doctor, but I'm here to listen and help, "
        "in whichever language you're comfortable with. "
        "Tell me, what's been troubling you?"
    ),
    "hi-IN": (
        "नमस्ते, मैं मेडलिंक हूँ, एक स्वास्थ्य हेल्पलाइन। मैं डॉक्टर नहीं हूँ, "
        "लेकिन आपकी बात सुनकर बता सकती हूँ कि आगे क्या करना चाहिए। "
        "बताइए, आपको क्या तकलीफ हो रही है?"
    ),
    "ta-IN": (
        "வணக்கம், நான் மெட்லிங்க், ஒரு சுகாதார உதவி எண். நான் மருத்துவர் அல்ல, "
        "ஆனால் நீங்கள் சொல்வதைக் கேட்டு அடுத்து என்ன செய்வது என்று சொல்ல முடியும். "
        "சொல்லுங்கள், உங்களுக்கு என்ன பிரச்சினை?"
    ),
    "te-IN": (
        "నమస్కారం, నేను మెడ్‌లింక్, ఒక ఆరోగ్య సహాయ లైన్. నేను డాక్టర్‌ని కాదు, "
        "కానీ మీరు చెప్పేది విని తర్వాత ఏమి చేయాలో చెప్పగలను. "
        "చెప్పండి, మీకు ఏమి ఇబ్బంది?"
    ),
    "kn-IN": (
        "ನಮಸ್ಕಾರ, ನಾನು ಮೆಡ್‌ಲಿಂಕ್, ಒಂದು ಆರೋಗ್ಯ ಸಹಾಯವಾಣಿ. ನಾನು ಡಾಕ್ಟರ್ ಅಲ್ಲ, "
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
You are MedLink, answering a health helpline call. You have just greeted the
caller. Right now you are simply getting to know what is wrong.

{SHARED_STYLE}

# This part of the call
- Do NOT greet them again. Let them tell you what is wrong in their own words.
- If they share their name or age first but not the problem, welcome that
  warmly and gently ask what has been troubling them. Never reply with just
  "anything else?".
- Once you know what is wrong, show you understand in a few kind words. If you
  don't yet know who it is for or roughly their age, ask naturally.
- Then call `record_complaint` (with their name and age if they said them).
  Don't try to work out the cause or suggest medicine yet - you will come back
  to that once you understand more.
"""


class IntakeAgent(MedLinkAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(instructions=INSTRUCTIONS, **kwargs)

    async def on_enter(self) -> None:
        # Always English, whatever the caller spoke last time. They can ask for
        # another language at any point and workflows.base switches instantly.
        greeting = GREETINGS[DEFAULT_LANGUAGE_CODE]
        self.data.language = DEFAULT_LANGUAGE_CODE
        try:
            self.session.tts.update_options(
                target_language_code=DEFAULT_LANGUAGE_CODE
            )
        except Exception:
            logger.exception("could not set the greeting language")
        # say() not generate_reply(): deterministic wording, no LLM round trip.
        await self.session.say(greeting)

    @function_tool
    async def record_complaint(
        self,
        context: RunContext[MedLinkUserData],
        complaint: str,
        patient_age_years: int | None = None,
        is_for_child: bool = False,
        patient_name: str | None = None,
    ):
        """Record the caller's main health complaint and who it is about.

        Call this as soon as you understand what is wrong. It moves the call on
        to the questioning stage.

        Args:
            complaint: The main problem in the caller's own words, in English.
            patient_age_years: Age of the person who is unwell, if known.
            is_for_child: True if the caller is asking on behalf of a child.
            patient_name: The unwell person's name, only if the caller said it.
        """
        data = context.userdata
        data.chief_complaint = complaint
        data.patient.is_for_child = is_for_child
        if patient_age_years is not None:
            data.patient.age_years = patient_age_years
        # Callers often open with their name. Intake used to have nowhere to put
        # it, so it was acknowledged ("Thank you, Dharanish") and then lost.
        # The model sometimes fills an unknown name with the *string* "null",
        # which would otherwise be saved as a patient called "null".
        name = (patient_name or "").strip()
        if name and name.casefold() not in _NO_NAME and data.patient_name is None:
            data.patient_name = name[:128]

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

        # No tool result on purpose: a returned string is handed to the LLM, which
        # then speaks it before the hand-off ("Thank you. Let me ask a few quick
        # questions.") - a scripted gear-change the caller hears. The triage
        # agent's on_enter carries the conversation on instead.
        return TriageAgent(chat_ctx=self.chat_ctx)
