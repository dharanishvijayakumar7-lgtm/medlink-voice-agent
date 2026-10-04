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
from knowledge.triage_kb import get_triage_kb
from medicine.filter import recommend as recommend_medicines
from session_state import MedLinkUserData
from workflows.base import SHARED_STYLE, MedLinkAgent, reply_language_note

logger = logging.getLogger("medlink.workflow")

INSTRUCTIONS = f"""\
You are MedLink, on a health helpline call. You understand the caller's problem
now. Explain what is likely going on and what to do, the way a kind doctor
would - a conversation, not a list read out.

{SHARED_STYLE}

# Explaining and advising
- Tell them what this most likely is and why, in plain words, tied to what
  they told you. Be honest that you cannot examine them.
- The medicine guidance for their main problem is given to you. Read it back
  in their language - never change a name, dose or warning, never add one.
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


# Demo mode: one short answer in a fixed shape, then the call ends. Everything
# the caller must hear is still here - what to do, what not to do, when to see a
# doctor, and the disclaimer - just said once and briefly. The safety rules are
# the same, because the medicine guidance still comes from the filtered pipeline.
DEMO_INSTRUCTIONS = f"""\
You are MedLink, on a health helpline call. You understand the caller's problem
now. Give your answer once, briefly, and then end the call.

{SHARED_STYLE}

# Your answer: about 60-80 words, four parts, in this order
1. What it looks like - ONE sentence, the likely cause in everyday words.
   Say "this looks like" or "this is most likely", never claim to be certain.
2. What to do - two short actions. The medicine guidance you were given is
   part of this: never change a name, dose or warning, and never add one.
3. What not to do - one short line.
4. When to see a doctor - one short line, then: {settings.disclaimer}

# How to say it
- Plain, everyday words, short sentences, warm but brief.
- Do NOT list possibilities, do not repeat their answers back to them, and do
  not ask whether they need anything else.
- Straight after this answer, call `end_call`. If they do ask something, answer
  it in one short sentence, then close.

# Absolute rules
- If the guidance says no medicine is appropriate, do NOT suggest one anyway -
  say so kindly and point them to a doctor or pharmacist.
- Never mention antibiotics, injections, or anything needing a prescription.
"""


class RecommendAgent(MedLinkAgent):
    def __init__(self, **kwargs) -> None:
        instructions = DEMO_INSTRUCTIONS if settings.demo_mode else INSTRUCTIONS
        super().__init__(instructions=instructions, **kwargs)

    async def on_enter(self) -> None:
        # The medicine guidance is fetched here, before the agent says anything,
        # rather than left for the model to request mid-reply. Left to the model,
        # it fetched first and read the result straight out - a real test call
        # went from questions to "take paracetamol" without ever saying what the
        # problem probably was. With everything in hand up front, one reply can
        # follow the order a caring doctor would use.
        guidance = await self._guidance(self._main_symptom())
        shared = (
            f"{self._context_block()}{self._likely_causes()}\n\n"
            "# Medicine guidance - already safety-checked for this caller\n"
            f"{guidance}\n\n"
            f"{reply_language_note(self.data)}\n"
        )
        if settings.demo_mode:
            # Short mode. Some of what a caller normally gets asked was never
            # asked, so where something is unknown, say what you can and lean on
            # "see a doctor if it does not settle" rather than filling the gap
            # with a guess.
            await self.session.generate_reply(
                instructions=(
                    shared + "Give the four-part answer now, about 60-80 words: "
                    "what it looks like, what to do (with the guidance above), "
                    "what not to do, when to see a doctor, then the disclaimer. "
                    "Don't greet again and don't read it as a numbered list. "
                    "Where you were not able to ask about something, do not "
                    "guess - say to see a doctor if it does not settle. Then "
                    "call end_call."
                )
            )
            return
        await self.session.generate_reply(
            instructions=(
                shared + "Now speak to them warmly, in this order:\n"
                "1. What this most likely is, and WHY - tied to what they told "
                "you. Say honestly that you cannot examine them.\n"
                "2. What to do: the home care, then the medicine guidance above "
                "(names, doses and warnings exactly as given; none if it gives "
                "none).\n"
                "3. What NOT to do, from the approved advice.\n"
                "4. Which signs mean they must see a doctor.\n"
                "5. Ask whether they have any questions.\n"
                "Don't greet again, and don't read it out as a numbered list - "
                "say it the way you would to someone sitting beside you."
            )
        )

    def _likely_causes(self) -> str:
        causes = ", ".join(self.data.possible_causes)
        if not causes:
            return ""
        text = f"\n\n# What you concluded it might be (most likely first)\n{causes}"
        if self.data.possible_causes_reasoning:
            text += f"\nWhy: {self.data.possible_causes_reasoning}"
        return text

    def _main_symptom(self) -> str:
        """What to look medicine up for: the matched presentation and the complaint.

        Both, because each misses things alone. The presentation name ("Cut,
        wound or minor injury") is tidy but narrow; the caller's own complaint
        ("deep cut on my leg") carries the words a medicine is listed under.
        Which medicines are even eligible is still set by the presentation.
        """
        parts = []
        entry_id = self.data.triage_entry_id
        if entry_id and (entry := get_triage_kb().by_id.get(entry_id)):
            parts.append(entry.presentation)
        if self.data.chief_complaint:
            parts.append(self.data.chief_complaint)
        return " ".join(parts)

    async def _guidance(self, symptom: str) -> str:
        """Run the safety-filtered medicine lookup and record what it offered."""
        data = self.data
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
    async def get_medicine_guidance(
        self, context: RunContext[MedLinkUserData], symptom: str
    ) -> str:
        """Get safe over-the-counter guidance for a DIFFERENT symptom.

        Guidance for the caller's main problem was already fetched when you took
        over - do not call this for it again. Use it only when the caller raises
        another symptom and asks what they can take. It applies every safety
        filter (prescription-only, age, pregnancy, contraindications, drug
        interactions, duration). Never suggest a medicine it did not give you.

        Args:
            symptom: The symptom in English, e.g. "fever", "loose motions".
        """
        return await self._guidance(symptom)

    @function_tool
    async def end_call(
        self, context: RunContext[MedLinkUserData], disposition: str = "self_care"
    ) -> str:
        """Close the call warmly once the caller has what they need.

        Args:
            disposition: One of self_care, clinic.
        """
        data = context.userdata
        if data.disposition:
            # Called twice. Saying goodbye again just makes the agent ramble -
            # a demo call ended with three goodbyes in a row.
            # Safe to say out loud, because the model sometimes reads a tool
            # result verbatim - one demo call ended on "the call has already
            # been closed", spoken to the caller.
            return "Take care."
        data.disposition = disposition
        if settings.demo_mode:
            # The advice already carried the doctor line and the disclaimer.
            return (
                "(Private note, not to be said aloud) Say one short goodbye "
                "line, nothing else."
            )
        return (
            "Reassure them briefly, remind them to see a doctor if it gets worse, "
            "thank them for calling, and say goodbye."
        )
