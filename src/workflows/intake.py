"""Intake: greet, find out who is unwell and what is wrong, then hand off.

Deliberately narrow. It captures the chief complaint plus basic patient context
and gets out of the way - all the clinical questioning happens in triage.
"""

from __future__ import annotations

import logging

from livekit.agents import RunContext, function_tool

from config import DEFAULT_LANGUAGE_CODE
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
        "Namaste, main MedLink hoon, ek swasthya helpline. Main doctor nahi hoon, "
        "lekin main aapki baat sunkar bata sakta hoon ki aage kya karna chahiye. "
        "Bataiye, aapko kya takleef ho rahi hai?"
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

        logger.info(
            "intake complete",
            extra={"call_id": data.call_id, "complaint": complaint},
        )

        from workflows.triage import TriageAgent

        return (
            TriageAgent(chat_ctx=self.chat_ctx),
            "Thank you. Let me ask a few quick questions.",
        )
