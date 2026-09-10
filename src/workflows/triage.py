"""Triage: ask the fewest targeted follow-ups needed to route the call safely.

Question selection is guided, not scripted: the agent is handed a short list of
*candidate* questions (from the triage KB in P1.5, falling back to core slots)
and picks the most informative unanswered one. Severity scoring is computed in
`routing.assess` - never by the model.
"""

from __future__ import annotations

import logging

from livekit.agents import ChatContext, ChatMessage, RunContext, function_tool

from config import settings
from knowledge.triage_kb import apply_to_session
from session_state import MedLinkUserData
from workflows import routing
from workflows.base import SHARED_STYLE, MedLinkAgent

logger = logging.getLogger("medlink.workflow")

INSTRUCTIONS = f"""\
You are MedLink, continuing a health helpline call. The caller has told you
their main problem. Now you gather just enough detail to judge how serious it is.

{SHARED_STYLE}

# How to question
- Ask ONE question per turn, then wait for the answer.
- Ask the FEWEST questions needed - at most {settings.max_followup_questions}
  in total. Never interrogate.
- Prefer the suggested questions you are given; they are chosen for this
  specific symptom. Skip any the caller has already answered.
- After each answer, call `record_answer` with a short slot name and what they said.
- Ask about pregnancy, ongoing conditions, or current medicines ONLY if it is
  relevant to what they described - then call `record_patient_context`.
- When you know how long it has been going on and how bad it is, call
  `finish_questions`. Do not keep asking out of habit.

# Never
- Never name a medicine here.
- Never state a diagnosis. You may say "this sounds like it could be ..." only
  after `finish_questions` has routed the call.
"""


class TriageAgent(MedLinkAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(instructions=INSTRUCTIONS, **kwargs)

    async def on_enter(self) -> None:
        suggestions = self.data.next_questions(limit=3)
        hint = "\n".join(f"- {q}" for q in suggestions)
        await self.session.generate_reply(
            instructions=(
                f"{self._context_block()}\n\n"
                f"# Suggested next questions (pick the single most useful one)\n{hint}\n\n"
                "Ask that one question now, in one short sentence. Do not greet again."
            )
        )

    async def on_turn(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Refresh the candidate-question hint each turn (KB-driven from P1.5)."""
        suggestions = self.data.next_questions(limit=3)
        if suggestions:
            turn_ctx.add_message(
                role="assistant",
                content=(
                    "Questions still worth asking, most useful first: "
                    + "; ".join(suggestions)
                    + f". You have asked {self.data.questions_asked} of "
                    f"{settings.max_followup_questions} allowed."
                ),
            )

    @function_tool
    async def record_answer(
        self, context: RunContext[MedLinkUserData], slot: str, answer: str
    ) -> str:
        """Record the caller's answer to a follow-up question.

        Args:
            slot: Short label for what was asked - one of: duration, severity,
                location, associated, history, or another short lowercase word.
            answer: What the caller said, summarised in English.
        """
        data = context.userdata
        data.record_answer(slot, answer)
        data.questions_asked += 1

        # Opportunistically parse duration into days for the safety filters.
        if slot == "duration":
            data.patient.symptom_duration_days = _parse_duration_days(answer)

        if routing.has_enough_information(data):
            return "Recorded. You now have enough to assess - call finish_questions."
        return "Recorded. Ask the next most useful question."

    @function_tool
    async def record_patient_context(
        self,
        context: RunContext[MedLinkUserData],
        is_pregnant: bool | None = None,
        is_breastfeeding: bool | None = None,
        known_conditions: list[str] | None = None,
        current_medications: list[str] | None = None,
        allergies: list[str] | None = None,
        past_medical_issues: list[str] | None = None,
    ) -> str:
        """Record safety-relevant background about the person who is unwell.

        Only call this for details the caller actually volunteered or confirmed.
        Never guess or infer any of it. These directly gate which medicines are
        safe to suggest.

        Args:
            is_pregnant: Only if the caller said so.
            is_breastfeeding: Only if the caller said so.
            known_conditions: Ongoing conditions they mentioned, e.g. diabetes.
            current_medications: Medicines they say they are ALREADY taking.
            allergies: Drug or other allergies they mentioned.
            past_medical_issues: Relevant past illnesses, surgery or admissions.
        """
        data = context.userdata
        if is_pregnant is not None:
            data.patient.is_pregnant = is_pregnant
        if is_breastfeeding is not None:
            data.patient.is_breastfeeding = is_breastfeeding
        if known_conditions:
            data.patient.known_conditions.extend(known_conditions)
            data.medical_history.extend(
                {"kind": "condition", "detail": c} for c in known_conditions
            )
        if current_medications:
            data.patient.current_medications.extend(current_medications)
        if allergies:
            data.medical_history.extend(
                {"kind": "allergy", "detail": a} for a in allergies
            )
        if past_medical_issues:
            data.medical_history.extend(
                {"kind": "past_issue", "detail": p} for p in past_medical_issues
            )
        return "Recorded."

    @function_tool
    async def record_caller_identity(
        self,
        context: RunContext[MedLinkUserData],
        name: str | None = None,
        gender: str | None = None,
    ) -> str:
        """Record the caller's name or gender, ONLY if they stated it themselves.

        Never ask for these and never infer them - not from the voice, not from
        the name. Call this only when the caller has volunteered the detail.

        Args:
            name: The name they gave for the person who is unwell.
            gender: Only if explicitly stated, e.g. "male", "female".
        """
        data = context.userdata
        if name and name.strip():
            data.patient_name = name.strip()[:128]
        if gender and gender.strip():
            data.patient_gender = gender.strip()[:16]
        return "Recorded."

    @function_tool
    async def finish_questions(self, context: RunContext[MedLinkUserData]):
        """Finish questioning and route the call based on how serious it is.

        Call this once you know roughly how long the problem has lasted and how
        severe it is - or sooner if the caller cannot answer more.
        """
        data = context.userdata
        # Re-run the KB now that the answers are in, so presentation-specific
        # modifiers (fever over 5 days, blood in stool, ...) count toward severity.
        apply_to_session(data)
        severity, urgency = routing.assess(data)
        logger.info(
            "triage assessed",
            extra={
                "call_id": data.call_id,
                "triage_entry": data.triage_entry_id,
                "severity": severity,
                "urgency": urgency,
                "questions": data.questions_asked,
            },
        )

        if routing.should_escalate(data):
            from workflows.escalate import EscalateAgent

            return (
                EscalateAgent(chat_ctx=self.chat_ctx),
                "This needs proper medical attention - let me help with that.",
            )

        from workflows.recommend import RecommendAgent

        return (
            RecommendAgent(chat_ctx=self.chat_ctx),
            "Thank you, that's enough for me to help.",
        )


def _parse_duration_days(text: str) -> int | None:
    """Best-effort '3 days' / 'two weeks' / 'since morning' -> days."""
    import re

    lowered = text.casefold()
    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    if any(
        w in lowered
        for w in ("today", "this morning", "since morning", "just now", "few hours")
    ):
        return 0
    if "yesterday" in lowered:
        return 1
    match = re.search(r"(\d+)", lowered)
    number = (
        int(match.group(1))
        if match
        else next((v for w, v in words.items() if w in lowered), None)
    )
    if number is None:
        return None
    if "week" in lowered:
        return number * 7
    if "month" in lowered:
        return number * 30
    if "hour" in lowered:
        return 0
    return number
