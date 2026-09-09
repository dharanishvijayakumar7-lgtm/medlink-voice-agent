"""Per-call session state shared across the MedLink workflow agents.

Attached to the LiveKit `AgentSession` as typed `userdata`, so every agent and
tool can read/write the same picture of the call. Deliberately plain data: the
routing decisions that consume it live in `workflows/routing.py` and are pure
functions so they can be unit-tested without a live session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from config import DEFAULT_LANGUAGE_CODE
from medicine.filter import PatientContext
from safety.redflags import RedFlagHit

# Core follow-up slots. The triage KB (P1.5) supplies presentation-specific
# questions on top of these; these are the minimum needed to triage safely.
CORE_SLOTS: tuple[str, ...] = (
    "duration",  # how long has this been going on
    "severity",  # how bad is it (mild / moderate / severe)
    "location",  # where exactly
    "associated",  # other symptoms alongside it
    "history",  # happened before / relevant conditions
)

SLOT_QUESTIONS: dict[str, str] = {
    "duration": "How long have you been feeling this?",
    "severity": "How bad is it right now - mild, moderate, or severe?",
    "location": "Where exactly do you feel it?",
    "associated": "Are you having any other problems along with this, like fever?",
    "history": "Have you had this before, or do you have any ongoing health problems?",
}


@dataclass
class MedLinkUserData:
    """Everything known about the current call."""

    # --- call identity (populated from SIP / persisted in P1.6) ---
    call_id: str = field(default_factory=lambda: str(uuid4()))
    caller_phone: str | None = None
    user_id: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    channel: str = "web"  # web | pstn | console
    is_returning_caller: bool = False
    previous_summary: str | None = None

    # --- language ---
    language: str = DEFAULT_LANGUAGE_CODE

    # --- clinical picture ---
    chief_complaint: str | None = None
    patient: PatientContext = field(default_factory=PatientContext)
    answers: dict[str, str] = field(default_factory=dict)
    questions_asked: int = 0

    # --- triage (KB-driven from P1.5) ---
    triage_entry_id: str | None = None
    candidate_questions: list[str] = field(default_factory=list)
    # None => unrestricted OTC search; set() => explicitly no OTC is appropriate.
    allowed_otc_classes: set[str] | None = None
    severity_score: int = 0
    urgency: str = "unknown"  # unknown | self_care | clinic | urgent | emergency

    # --- safety ---
    red_flag: RedFlagHit | None = None
    emergency_handled: bool = False

    # --- outcome ---
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    consent_store: bool | None = None
    consent_share_doctor: bool | None = None
    escalated: bool = False
    disposition: str | None = None

    # ------------------------------------------------------------------
    def record_answer(self, slot: str, value: str) -> None:
        if value and value.strip():
            self.answers[slot] = value.strip()

    def unanswered_slots(self) -> list[str]:
        return [s for s in CORE_SLOTS if s not in self.answers]

    def answered_count(self) -> int:
        return len(self.answers)

    def next_questions(self, limit: int = 3) -> list[str]:
        """Questions still worth asking - KB candidates first, then core slots."""
        out = [q for q in self.candidate_questions if q not in self.answers.values()]
        out += [SLOT_QUESTIONS[s] for s in self.unanswered_slots()]
        return out[:limit]

    def clinical_summary(self) -> str:
        """Compact English summary for the LLM, the DB, and doctor handoff."""
        parts = [f"Chief complaint: {self.chief_complaint or 'not yet stated'}."]
        if self.patient.age_years is not None:
            who = "child" if self.patient.is_for_child else "adult"
            parts.append(f"Patient: {who}, age {self.patient.age_years}.")
        if self.patient.is_pregnant:
            parts.append("Pregnant.")
        if self.patient.known_conditions:
            parts.append(f"Conditions: {', '.join(self.patient.known_conditions)}.")
        if self.patient.current_medications:
            parts.append(
                f"Current medicines: {', '.join(self.patient.current_medications)}."
            )
        for slot, answer in self.answers.items():
            parts.append(f"{slot.capitalize()}: {answer}.")
        if self.red_flag:
            parts.append(f"RED FLAG: {self.red_flag.category_id}.")
        if self.urgency != "unknown":
            parts.append(
                f"Assessed urgency: {self.urgency} (score {self.severity_score})."
            )
        if self.recommendations:
            names = ", ".join(r.get("generic_name", "?") for r in self.recommendations)
            parts.append(f"Medicines discussed: {names}.")
        return " ".join(parts)
