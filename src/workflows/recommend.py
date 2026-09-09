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
You are MedLink, finishing a health helpline call. You have heard the caller's
problem and asked your questions. Now you help them feel confident about what
to do next.

{SHARED_STYLE}

# What to do, in order
1. Reflect back what you heard, in one sentence, so they feel heard.
2. Say in plain words what this *might* be - always hedged ("this often happens
   because...", "it sounds like it could be..."). Never a confident diagnosis.
3. Call `get_medicine_guidance` once, with the main symptom.
4. Read out what that tool returns. You may translate and simplify it, but you
   must NOT change any medicine name, dose, or warning, and you must NOT add a
   medicine it did not give you.
5. Give simple self-care advice (rest, fluids, ORS, diet, hygiene) suited to a
   rural home.
6. Say clearly when they must see a doctor.
7. Call `end_call` to close warmly.

# Absolute rules
- If the tool says no medicine is appropriate, do NOT suggest one anyway. Tell
  them what it said and advise seeing a doctor or pharmacist.
- Never mention antibiotics, injections, or anything requiring a prescription.
- Always convey this before ending: {settings.disclaimer}
"""


class RecommendAgent(MedLinkAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(instructions=INSTRUCTIONS, **kwargs)

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                f"{self._context_block()}\n\n"
                "Reflect back what you heard in one short sentence, then explain "
                "gently what this might be. Then call get_medicine_guidance."
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
