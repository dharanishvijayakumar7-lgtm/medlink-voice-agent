"""Recommend: safe self-care and, only if the filters allow it, an OTC medicine.

The agent does not choose a medicine. `medicine.recommend` runs the hard safety
filters and returns fully-formed text built from structured formulary fields;
this agent may only read it out and rephrase it into the caller's language.
"""

from __future__ import annotations

import asyncio
import logging

from livekit.agents import RunContext, function_tool

from config import settings
from medicine.filter import recommend as recommend_medicines
from session_state import MedLinkUserData
from workflows.base import SHARED_STYLE, MedLinkAgent

logger = logging.getLogger("medlink.workflow")

INSTRUCTIONS = f"""\
You are MedLink, on a health helpline call. You understand the caller's problem
now. Explain what is likely going on and what to do, the way a kind doctor
would - a conversation, not a list read out.

{SHARED_STYLE}

# Explaining and advising
- Tell them what this most likely is and why, in plain words, tied to what
  they told you. Be honest that you cannot examine them.
- Call `get_medicine_guidance` once with the main symptom. Read it back in
  their language - never change a name, dose or warning, never add one.
- Share simple home care that fits a rural home (rest, fluids, ORS, food).
- Say what NOT to do as well as what to do - the advice below names what makes
  this worse, and callers act on it.
- Tell them clearly which signs mean they must see a doctor.
- Answer follow-up questions properly - food, drink, work, how long it takes.
  "See a doctor" is not an answer to "can I drink tea".
- If they ask about someone else, say you would need to ask about that person
  first - never pass this medicine on.
- Only call `end_call` once they have no more questions.

# Absolute rules
- If the tool says no medicine is appropriate, do NOT suggest one anyway -
  explain kindly and advise a doctor or pharmacist.
- Never mention antibiotics, injections, or anything needing a prescription.
- Before ending, gently convey: {settings.disclaimer}
"""


class RecommendAgent(MedLinkAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(instructions=INSTRUCTIONS, **kwargs)

    async def on_enter(self) -> None:
        causes = ", ".join(self.data.possible_causes)
        likely = (
            f"\n\n# What you concluded it might be (most likely first)\n{causes}"
            if causes
            else ""
        )
        await self.session.generate_reply(
            instructions=(
                f"{self._context_block()}{likely}\n\n"
                "Continue naturally - don't greet again. Let them know you've "
                "understood, then explain what this most likely is and why, in "
                "a warm, simple way. Then call get_medicine_guidance."
            )
        )

    @function_tool
    async def get_medicine_guidance(
        self, context: RunContext[MedLinkUserData], symptom: str
    ) -> str:
        """Get safe over-the-counter guidance for the caller's symptom.

        This applies every safety filter (prescription-only, age, pregnancy,
        contraindications, drug interactions, how long it has lasted). Read out
        what it returns. Never suggest a medicine it did not give you.

        Args:
            symptom: The main symptom in English, e.g. "fever", "loose motions".
        """
        data = context.userdata
        # Feed everything the caller described into the contraindication check,
        # not just formally recorded conditions.
        data.patient.reported_symptoms = [
            *(x for x in [data.chief_complaint] if x),
            *data.answers.values(),
        ]
        # BM25 + filtering is CPU work; keep it off the voice event loop.
        result = await asyncio.to_thread(
            recommend_medicines,
            symptom,
            data.patient,
            allowed_classes=data.allowed_otc_classes,
        )

        data.recommendations = [r.structured for r in result.recommendations]
        if result.escalate:
            data.urgency = "clinic"

        logger.info(
            "medicine guidance",
            extra={
                "call_id": data.call_id,
                "symptom": symptom,
                "recommended": [r.entry_id for r in result.recommendations],
                "rejected": result.rejected,
            },
        )
        return result.to_spoken()

    @function_tool
    async def end_call(
        self, context: RunContext[MedLinkUserData], disposition: str = "self_care"
    ) -> str:
        """Close the call warmly once the caller has what they need.

        Args:
            disposition: One of self_care, clinic.
        """
        data = context.userdata
        data.disposition = disposition
        return (
            "Reassure them briefly, remind them to see a doctor if it gets worse, "
            "thank them for calling, and say goodbye."
        )
