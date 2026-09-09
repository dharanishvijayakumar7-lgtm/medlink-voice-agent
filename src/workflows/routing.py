"""Pure routing / triage-scoring decisions for the MedLink workflow.

Kept free of LiveKit types so the safety-relevant logic can be unit-tested
directly, without a live session or an LLM.
"""

from __future__ import annotations

from config import settings
from session_state import MedLinkUserData

# Urgency levels, least to most severe.
SELF_CARE = "self_care"
CLINIC = "clinic"
URGENT = "urgent"
EMERGENCY = "emergency"

_SEVERITY_WORDS = {
    "severe": 4,
    "unbearable": 4,
    "worst": 4,
    "very bad": 4,
    "terrible": 3,
    "bad": 2,
    "moderate": 2,
    "medium": 2,
    "mild": 0,
    "slight": 0,
    "little": 0,
}

# Durations that push a presentation beyond safe self-care.
_LONG_DURATION_DAYS = 7


def _severity_from_words(text: str) -> int:
    lowered = text.casefold()
    return max(
        (score for word, score in _SEVERITY_WORDS.items() if word in lowered), default=0
    )


def compute_severity(ud: MedLinkUserData) -> int:
    """Score 0-10. Deterministic; the LLM never sets this directly."""
    if ud.red_flag is not None:
        return 10 if ud.red_flag.is_emergency else max(settings.severity_urgent, 6)

    score = 0
    if "severity" in ud.answers:
        score += _severity_from_words(ud.answers["severity"])

    # Presentation-specific modifiers from the triage KB (e.g. fever over 5 days,
    # blood in stool, exertional chest burning).
    score += ud.kb_severity_bonus

    duration = ud.patient.symptom_duration_days
    if duration is not None and duration > _LONG_DURATION_DAYS:
        score += 2

    # Vulnerable groups warrant a lower threshold for referral.
    age = ud.patient.age_years
    if age is not None and (age < 5 or age > 65):
        score += 2
    if ud.patient.is_pregnant:
        score += 2
    if ud.patient.known_conditions:
        score += 1

    return min(score, 10)


def decide_urgency(severity: int) -> str:
    if severity >= settings.severity_emergency:
        return EMERGENCY
    if severity >= settings.severity_urgent:
        return URGENT
    if severity >= 3:
        return CLINIC
    return SELF_CARE


def should_escalate(ud: MedLinkUserData) -> bool:
    """True when the call must go to the escalation flow rather than OTC advice."""
    if ud.red_flag is not None and ud.red_flag.is_emergency:
        return True
    return ud.urgency in (URGENT, EMERGENCY)


def has_enough_information(ud: MedLinkUserData) -> bool:
    """Stop interrogating once we can triage safely, or we've asked enough.

    Minimum viable picture: how long, and how bad. Anything beyond that is a
    bonus - the goal is the fewest questions that still make the call safe.
    """
    if ud.questions_asked >= settings.max_followup_questions:
        return True
    return "duration" in ud.answers and "severity" in ud.answers


def assess(ud: MedLinkUserData) -> tuple[int, str]:
    """Score and classify the call, writing both back onto the session state."""
    severity = compute_severity(ud)
    urgency = decide_urgency(severity)
    ud.severity_score = severity
    ud.urgency = urgency
    return severity, urgency


def next_stage(ud: MedLinkUserData) -> str:
    """Which workflow agent should hold the session right now."""
    if should_escalate(ud):
        return "escalate"
    if not ud.chief_complaint:
        return "intake"
    if not has_enough_information(ud):
        return "triage"
    return "recommend"
