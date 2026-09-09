"""Escalate: emergencies and anything too serious for self-care.

Reached two ways:
  * `routing.should_escalate` after triage scoring, or
  * the deterministic red-flag hook in `workflows.base`, which force-hands the
    session here and suppresses the LLM's reply entirely.

For a red flag the first thing the caller hears is fixed, pre-written text from
`data/redflags.yaml` - not model output. Provider matching, call transfer, SMS
and consented doctor handoff arrive in P2.
"""

from __future__ import annotations

import logging

from livekit.agents import RunContext, function_tool

from config import settings
from safety.redflags import format_redflag_advice
from session_state import MedLinkUserData
from workflows.base import SHARED_STYLE, MedLinkAgent

logger = logging.getLogger("medlink.workflow")

INSTRUCTIONS = f"""\
You are MedLink. This call needs real medical care, possibly urgently. Stay calm
and steady - the caller may be frightened.

{SHARED_STYLE}

# What to do
1. If safety advice has already been spoken, do NOT repeat it word for word.
   Check they understood and ask if someone is with them.
2. Be direct but kind about the fact that this needs a doctor or hospital.
   Do not minimise it, and do not dramatise it.
3. Offer practical help getting there: who can take them, how far the nearest
   clinic is, whether to call an ambulance on {settings.ambulance_number}.
4. Ask permission before arranging anything on their behalf, then call
   `record_consent`.
5. Answer simple questions about what to do while waiting (position, fluids,
   keeping warm) - but nothing that delays them getting help.

# Absolute rules
- NEVER suggest any medicine here. Not even paracetamol.
- NEVER suggest waiting to see if it improves.
- If they are unsure, err toward going to a hospital.
- The emergency number is {settings.emergency_number}; ambulance is
  {settings.ambulance_number}.
"""


class EscalateAgent(MedLinkAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(instructions=INSTRUCTIONS, **kwargs)

    async def on_enter(self) -> None:
        data: MedLinkUserData = self.data
        data.escalated = True
        data.disposition = data.disposition or "urgent"

        if data.red_flag is not None:
            # Deterministic, pre-reviewed wording - spoken before any LLM output.
            advice = format_redflag_advice(
                data.red_flag,
                emergency_number=settings.emergency_number,
                ambulance_number=settings.ambulance_number,
            )
            if data.red_flag.is_emergency:
                data.disposition = "emergency"
            logger.warning(
                "escalating on red flag",
                extra={"call_id": data.call_id, "category": data.red_flag.category_id},
            )
            await self.session.say(advice)
            await self.session.generate_reply(
                instructions=(
                    f"{self._context_block()}\n\n"
                    "You have just spoken the urgent safety advice. Now, in one or "
                    "two short sentences, check they heard it and ask whether "
                    "someone is with them who can take them for help."
                )
            )
            return

        await self.session.generate_reply(
            instructions=(
                f"{self._context_block()}\n\n"
                "Explain kindly but clearly that this needs to be seen by a doctor "
                "soon, and ask how far their nearest clinic or health centre is."
            )
        )

    @function_tool
    async def record_consent(
        self,
        context: RunContext[MedLinkUserData],
        wants_help_reaching_care: bool,
        may_share_summary_with_doctor: bool = False,
    ) -> str:
        """Record what the caller agreed to before anything is arranged for them.

        Args:
            wants_help_reaching_care: They accepted help finding a doctor/clinic.
            may_share_summary_with_doctor: They agreed we may pass their symptom
                summary to that doctor.
        """
        data = context.userdata
        data.consent_share_doctor = may_share_summary_with_doctor
        logger.info(
            "consent recorded",
            extra={
                "call_id": data.call_id,
                "help": wants_help_reaching_care,
                "share": may_share_summary_with_doctor,
            },
        )
        if not wants_help_reaching_care:
            return (
                "Respect that. Remind them of the warning signs that mean they must "
                "go immediately, and that they can call back any time."
            )
        # Provider lookup / booking / SMS / SIP transfer land here in P2.
        return (
            "Tell them the nearest government health centre or hospital is the right "
            f"place, and that they can call {settings.ambulance_number} for an "
            "ambulance. Confirm someone can go with them."
        )

    @function_tool
    async def end_call(self, context: RunContext[MedLinkUserData]) -> str:
        """Close the call once the caller knows what to do."""
        context.userdata.disposition = context.userdata.disposition or "urgent"
        return (
            "Repeat the single most important next step in one sentence, tell them "
            "to go now, and say goodbye warmly."
        )
