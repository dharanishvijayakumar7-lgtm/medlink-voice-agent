"""Triage: ask the fewest targeted follow-ups needed to route the call safely.

Question selection is guided, not scripted: the agent is handed a short list of
*candidate* questions (from the triage KB in P1.5, falling back to core slots)
and picks the most informative unanswered one. Severity scoring is computed in
`routing.assess` - never by the model.
"""

from __future__ import annotations

import logging
import re

from livekit.agents import ChatContext, ChatMessage, RunContext, function_tool

from knowledge.triage_kb import apply_to_session
from session_state import SLOT_QUESTIONS, MedLinkUserData
from workflows import routing
from workflows.base import SHARED_STYLE, MedLinkAgent, reply_language_note
from workflows.intake import _NO_NAME

logger = logging.getLogger("medlink.workflow")

INSTRUCTIONS = f"""\
You are MedLink, in a health helpline call. The caller has told you what is
wrong. Now understand their situation the way a good doctor would on the phone.

{SHARED_STYLE}

# Understanding what is going on
- Ask about what matters for THIS problem: how long, how bad, and any worrying
  signs that go with it. Let their answers lead your next question.
- ONE question per turn. Never two or three in one breath.
- Keep acknowledging them. A caring word matters more than speed.
- Never re-ask something they already told you. If they give several details
  at once, take them all in.
- Quietly save what you learn with `record_answer`. Save each thing once.
- Ask about regular medicines, health conditions or pregnancy only if it
  matters here, in one gentle question - then `record_patient_context`.
- If they share their name or gender themselves, save it with
  `record_caller_identity`. Never ask for or guess these.
- Never say a warning sign is absent unless you asked and they answered. If you
  did not ask, you do not know.
- Like a good doctor, stop asking once you know how long, how bad, and whether
  anything worrying is present. Then call `finish_questions` with up to 3
  `possible_causes` (most likely first) and a one-line `reasoning`.

# Not yet
- Don't name any medicine, and don't give your conclusion - you'll explain
  what it might be and what to do right after `finish_questions`.
"""


class TriageAgent(MedLinkAgent):
    def __init__(self, **kwargs) -> None:
        super().__init__(instructions=INSTRUCTIONS, **kwargs)

    async def on_enter(self) -> None:
        hint = "\n".join(f"- {q}" for q in open_questions(self.data))
        await self.session.generate_reply(
            instructions=(
                f"{self._context_block()}\n\n"
                "# Things a doctor would want to know about this problem\n"
                f"{hint}\n\n"
                "Continue the conversation naturally - don't greet again. If you "
                "haven't yet shown you understand how they feel, do that first in "
                "a few words. Then ask the one thing you most need to know. "
                f"{reply_language_note(self.data)}"
            )
        )

    async def on_turn(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Keep the clinically useful unknowns in view, KB-driven from P1.5.

        Deliberately no question count and no "ask the next question" order: those
        made the agent run through a checklist instead of listening. Once the
        essentials are known the note says so, so there is a natural point to stop
        asking and start explaining.
        """
        if routing.has_enough_information(self.data):
            note = (
                "(Private note, not to be said aloud) You know enough now. Respond "
                "to what they just said, then move on to explaining by calling "
                "finish_questions - don't start new questions."
            )
        else:
            note = (
                "(Private note, not to be said aloud) Still worth knowing, if they "
                "haven't already said: " + "; ".join(open_questions(self.data)) + ". "
                "Respond to what they just said first, and ask at most one thing."
            )
        turn_ctx.add_message(role="assistant", content=note)

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
        slot = canonical_slot(slot)
        # Count a question once. The model re-saves earlier answers on later turns,
        # which used to inflate the count toward the limit.
        if slot not in data.answers:
            data.questions_asked += 1
        data.record_answer(slot, answer)

        # Opportunistically parse duration into days for the safety filters.
        if slot == "duration":
            data.patient.symptom_duration_days = _parse_duration_days(answer)

        # Neutral results: "Ask the next most useful question" used to push the
        # model straight into another question after every single answer.
        if routing.has_enough_information(data):
            return "Saved. You likely know enough now - wrap up when it feels natural."
        return "Saved."

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
        # The model sometimes passes the string "null" for a detail it doesn't
        # have; treat that as absent rather than saving it as the value.
        name = (name or "").strip()
        if name and name.casefold() not in _NO_NAME:
            data.patient_name = name[:128]
        gender = (gender or "").strip()
        if gender and gender.casefold() not in _NO_NAME:
            data.patient_gender = gender[:16]
        return "Recorded."

    @function_tool
    async def finish_questions(
        self,
        context: RunContext[MedLinkUserData],
        possible_causes: list[str] | None = None,
        reasoning: str | None = None,
    ):
        """Finish questioning and route the call based on how serious it is.

        Call this once you know roughly how long the problem has lasted and how
        severe it is - or sooner if the caller cannot answer more.

        Args:
            possible_causes: Up to 3 things this could be, in plain words, most
                likely first. For the clinic's records only, not a diagnosis.
            reasoning: One line on why, based on what the caller said.
        """
        data = context.userdata
        # Optional on purpose: routing must still work if the model omits them.
        data.possible_causes = clean_possible_causes(possible_causes)
        reason = (reasoning or "").strip()
        data.possible_causes_reasoning = (
            reason[:300] if data.possible_causes and reason.casefold() not in _NO_NAME else None
        )

        # Safety gate, enforced in code rather than hoped for in the prompt: no
        # explaining or medicine until the warning-sign check has happened. The
        # model gets told what is missing and simply asks one more question.
        credit_volunteered_answers(data)
        if not routing.has_enough_information(data) and data.finish_refusals < MAX_REFUSALS:
            data.finish_refusals += 1
            still = "; ".join(open_questions(data))
            return (
                "Not yet - you haven't checked this yet: " + still + ". "
                "Acknowledge what they said, then ask the most important one "
                "naturally. Call finish_questions again once you know."
            )
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

        # Hand off without a scripted line - see IntakeAgent.record_complaint.
        if routing.should_escalate(data):
            from workflows.escalate import EscalateAgent

            return EscalateAgent(chat_ctx=self.chat_ctx)

        from workflows.recommend import RecommendAgent

        return RecommendAgent(chat_ctx=self.chat_ctx)


# Phrases that mark a question as asking for an essential slot, so an already
# answered slot isn't asked again under a different wording.
_SLOT_PHRASES: dict[str, tuple[str, ...]] = {
    "duration": ("how long", "since when", "how many days", "when did it start"),
    "severity": ("how bad", "how severe", "how strong", "how much pain"),
}

# The model picks its own slot labels, and `finish_questions` gates on the three
# in routing.ESSENTIAL_SLOTS by exact name. When it saved the warning-sign
# answer as "associated_symptoms" or "other_symptoms", the gate never opened:
# finish_questions kept replying "Not yet", the call never reached the advice
# stage, and a child with diarrhoea was given no ORS, no zinc and no danger
# signs. Names are folded onto the canonical three here.
#
# "symptoms" on its own is deliberately NOT a marker for `associated` - the
# model uses it for the chief complaint too, and letting that satisfy the gate
# would skip the warning-sign check this gate exists to enforce.
_SLOT_CANON: dict[str, tuple[str, ...]] = {
    "duration": ("duration", "how_long", "howlong", "since_when", "time_since"),
    "severity": ("severity", "severe", "how_bad", "howbad", "intensity", "pain_level"),
    "associated": (
        "associated",
        "accompany",
        "accompanying",
        "other_symptom",
        "othersymptom",
        "related_symptom",
        "warning",
        "red_flag",
        "redflag",
    ),
}


def canonical_slot(slot: str) -> str:
    """Fold a slot label the model invented onto the name the gate checks."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", slot.casefold()).strip("_")
    for canonical, markers in _SLOT_CANON.items():
        if any(marker in cleaned for marker in markers):
            return canonical
    return cleaned or slot


def _asks_for(question: str, slot: str) -> bool:
    q = question.casefold()
    return any(phrase in q for phrase in _SLOT_PHRASES.get(slot, ()))


# Words that carry no meaning when deciding whether a question was answered.
_QUESTION_STOP = frozenset(
    ["is", "are", "was", "were", "does", "did", "do", "has", "have", "had", "any", "some", "the", "a", "an", "you", "your", "yours", "she", "he", "it", "they", "them", "there", "here", "been", "being", "with", "and", "or", "of", "to", "in", "on", "at", "for", "from", "that", "this", "these", "those", "how", "what", "when", "where", "which", "who", "why", "since", "also", "able", "about", "along", "still", "very", "much", "more", "than", "been", "get", "got", "go", "going"]
)
# How much of a question's meaning must already appear in what the caller said
# before it counts as answered. Two words minimum, so a single incidental match
# ("fever" in an unrelated sentence) cannot silence a real question.
_COVERED_RATIO = 0.6
_COVERED_MIN_WORDS = 2
# How callers describe severity without being asked. Stemmed, like everything
# else compared against what they said.
_SEVERITY_WORDS = frozenset(
    {"mild", "moder", "sever", "bad", "wors", "slight", "terribl", "unbear",
     "manag", "bearabl", "strong", "intens"}
)


def _stem(word: str) -> str:
    """Crude suffix trim, so word forms match across a question and an answer.

    Both sides go through it, so it only has to be consistent, not linguistically
    correct. Without it "loose motions" did not answer "how many times have you
    passed motion", and "she is drinking water" did not answer "are you able to
    drink".
    """
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _meaningful(text: str) -> set[str]:
    return {
        _stem(w)
        for w in re.findall(r"[a-z]+", text.casefold())
        if len(w) > 3 and w not in _QUESTION_STOP
    }


def _already_answered(question: str, said: set[str]) -> bool:
    """True when the caller has already covered what this question asks.

    `answers` only holds what the model chose to record under a slot. Callers
    volunteer far more than that in one breath, and being asked it again reads
    as not listening - which is exactly what cost a real call its advice.
    """
    asked = _meaningful(question)
    if len(asked) < _COVERED_MIN_WORDS:
        return False
    overlap = asked & said
    return (
        len(overlap) >= _COVERED_MIN_WORDS
        and len(overlap) >= round(len(asked) * _COVERED_RATIO)
    )


def open_questions(data: MedLinkUserData) -> list[str]:
    """What is still worth asking, most clinically useful first.

    The triage KB's questions come first: they are written for this specific
    problem (for a headache: sudden onset, fever with neck stiffness, weakness or
    trouble seeing). Any that ask for something already answered are dropped -
    `MedLinkUserData.next_questions` never removed them, so a known duration was
    asked for again. Generic essential questions fill in only where the KB has no
    question covering that slot.
    """
    # Everything the caller has said: the complaint, what was recorded, and
    # their own words. A question they already covered is dropped.
    spoken = " ".join([data.chief_complaint or "", *data.answers.values(), *data.heard])
    said = _meaningful(spoken)

    answered = set(data.answers)
    # A slot counts as covered the moment the caller says it, not when the model
    # gets round to recording it. Someone who opens with "a headache for two
    # days, quite bad" was still asked how long it had been going on, because
    # nothing had been written to `answers` yet.
    if _parse_duration_days(spoken) is not None:
        answered.add("duration")
    if _SEVERITY_WORDS & said:
        answered.add("severity")
    kb = [
        q
        for q in data.candidate_questions
        if not any(_asks_for(q, slot) for slot in answered)
        and not _already_answered(q, said)
    ]
    generic = [
        SLOT_QUESTIONS[slot]
        for slot in routing.ESSENTIAL_SLOTS
        if slot not in answered and not any(_asks_for(q, slot) for q in kb)
    ]
    return (kb + generic)[:3]


# The gate sends the agent back for more at most this often. It exists to make
# sure the warning-sign question gets asked; once it has been, the call moves on.
MAX_REFUSALS = 1

# Words that show a caller is telling us whether something else is going on:
# "no blood, no vomiting", "नहीं, कोई बुखार नहीं".
_ANSWER_NEGATORS = frozenset(
    {"no", "not", "none", "nothing", "never", "without",
     "नहीं", "नही", "ना", "இல்லை", "లేదు", "ಇಲ್ಲ", "ഇല്ല"}
)


def credit_volunteered_answers(data: MedLinkUserData) -> None:
    """Fill the essential slots from what the caller has already said.

    The gate checks `answers`, which only holds what the model chose to record.
    A mother who opened with "since this morning, five times, no blood, no
    vomiting" had given duration and the warning-sign answer, but nothing was
    recorded - so the gate made the agent ask how long it had been going on.

    Duration and severity are credited from anything the caller said. The
    warning-sign slot needs more, because it is the safety check: the caller
    must have addressed at least two of the symptoms this presentation's own
    questions ask about, with a negation or confirmation ("no blood, no
    vomiting").
    """
    said = " ".join([data.chief_complaint or "", *data.heard])
    if "duration" not in data.answers and _parse_duration_days(said) is not None:
        data.record_answer("duration", _caller_said(data, _parse_duration_days))
        data.patient.symptom_duration_days = _parse_duration_days(said)
    words = _meaningful(said)
    if "severity" not in data.answers and _describes_severity(said):
        data.record_answer("severity", _caller_said(data, _describes_severity))
    if "associated" not in data.answers:
        asked_about = _meaningful(" ".join(data.candidate_questions))
        raw = {w.casefold() for w in re.findall(r"\w+", said)}
        if len(asked_about & words) >= 2 and raw & _ANSWER_NEGATORS:
            data.record_answer(
                "associated",
                _caller_said(data, lambda t: _meaningful(t) & asked_about),
            )


# "Five times since morning" is how people say how bad loose motions or
# vomiting are, and it is the measure a clinician would ask for anyway.
_HOW_OFTEN = re.compile(
    r"\b(\d+|two|three|four|five|six|seven|eight|nine|ten|many|several)\s+times\b",
    re.IGNORECASE,
)


def _describes_severity(text: str) -> bool:
    return bool(_SEVERITY_WORDS & _meaningful(text) or _HOW_OFTEN.search(text))


def _caller_said(data: MedLinkUserData, relevant) -> str:
    """The caller's own words that carry the answer, for the record."""
    for text in data.heard:
        if relevant(text):
            return f"(caller) {text[:200]}"
    return f"(caller) {(data.chief_complaint or '')[:200]}"


MAX_POSSIBLE_CAUSES = 3


def clean_possible_causes(causes: list[str] | None) -> list[str]:
    """Trim the model's suggestions to a short, de-duplicated, sane list.

    Drops placeholder strings ("null", "unknown") the model emits for "none",
    keeps order (most likely first), and caps the count and each entry's length.
    """
    out: list[str] = []
    seen: set[str] = set()
    for cause in causes or []:
        if not isinstance(cause, str):
            continue
        text = " ".join(cause.split())[:80]
        key = text.casefold()
        if not text or key in _NO_NAME or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) == MAX_POSSIBLE_CAUSES:
            break
    return out


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
