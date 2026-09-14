"""Behavior lock for workflow routing, triage scoring and session state.

These are the safety-relevant decisions that must NOT be left to the LLM, so
they are pure functions and tested directly without a live session.
"""

import pytest

from safety.redflags import detect_redflag
from session_state import CORE_SLOTS, MedLinkUserData
from workflows import routing
from workflows.triage import _parse_duration_days


def _ud(**kwargs) -> MedLinkUserData:
    ud = MedLinkUserData()
    for key, value in kwargs.items():
        setattr(ud, key, value)
    return ud


# --------------------------------------------------------------- severity ---


def test_no_information_is_low_severity():
    assert routing.compute_severity(_ud()) == 0
    assert routing.decide_urgency(0) == routing.SELF_CARE


def test_emergency_red_flag_pins_severity_to_max():
    ud = _ud(red_flag=detect_redflag("I have crushing chest pain"))
    assert ud.red_flag is not None
    assert routing.compute_severity(ud) == 10
    assert routing.decide_urgency(10) == routing.EMERGENCY
    assert routing.should_escalate(ud)


def test_urgent_red_flag_is_serious_but_not_emergency():
    ud = _ud(red_flag=detect_redflag("I have not passed urine all day"))
    assert ud.red_flag is not None
    assert not ud.red_flag.is_emergency
    severity = routing.compute_severity(ud)
    assert 5 <= severity < 10


def test_severity_words_raise_the_score():
    mild = _ud()
    mild.record_answer("severity", "it is mild")
    bad = _ud()
    bad.record_answer("severity", "the pain is severe")
    assert routing.compute_severity(bad) > routing.compute_severity(mild)


def test_vulnerable_groups_lower_the_referral_threshold():
    adult = _ud()
    adult.patient.age_years = 30
    infant = _ud()
    infant.patient.age_years = 2
    elderly = _ud()
    elderly.patient.age_years = 75
    pregnant = _ud()
    pregnant.patient.age_years = 28
    pregnant.patient.is_pregnant = True

    base = routing.compute_severity(adult)
    assert routing.compute_severity(infant) > base
    assert routing.compute_severity(elderly) > base
    assert routing.compute_severity(pregnant) > base


def test_long_duration_raises_severity():
    short = _ud()
    short.patient.symptom_duration_days = 1
    long = _ud()
    long.patient.symptom_duration_days = 30
    assert routing.compute_severity(long) > routing.compute_severity(short)


def test_severity_is_capped_at_ten():
    ud = _ud()
    ud.patient.age_years = 80
    ud.patient.is_pregnant = True
    ud.patient.symptom_duration_days = 90
    ud.patient.known_conditions = ["diabetes", "heart disease"]
    ud.record_answer("severity", "unbearable and severe")
    assert routing.compute_severity(ud) <= 10


def test_assess_writes_back_to_session_state():
    ud = _ud(red_flag=detect_redflag("she is unconscious and not responding"))
    severity, urgency = routing.assess(ud)
    assert ud.severity_score == severity == 10
    assert ud.urgency == urgency == routing.EMERGENCY


# ------------------------------------------------------- question stopping ---


def test_needs_more_questions_when_nothing_is_known():
    assert not routing.has_enough_information(_ud(chief_complaint="fever"))


def test_duration_and_severity_alone_are_not_enough():
    """A simulated call proved this unsafe: a caller who gave both at once was
    handed a paracetamol dose with no check for fever, vomiting or vision."""
    ud = _ud(chief_complaint="headache")
    ud.record_answer("duration", "since this morning")
    ud.record_answer("severity", "moderate")
    assert not routing.has_enough_information(ud)


def test_warning_sign_check_completes_the_picture():
    ud = _ud(chief_complaint="fever")
    ud.record_answer("duration", "two days")
    ud.record_answer("severity", "moderate")
    ud.record_answer("associated", "no vomiting, no rash")
    assert routing.has_enough_information(ud)


def test_question_cap_stops_the_interrogation():
    ud = _ud(chief_complaint="fever", questions_asked=99)
    assert routing.has_enough_information(ud)


# ------------------------------------------------------------ next_stage ---


def test_stage_intake_until_complaint_known():
    assert routing.next_stage(_ud()) == "intake"


def test_stage_triage_after_complaint():
    assert routing.next_stage(_ud(chief_complaint="loose motions")) == "triage"


def test_stage_recommend_once_enough_is_known():
    ud = _ud(chief_complaint="loose motions")
    ud.record_answer("duration", "since this morning")
    ud.record_answer("severity", "mild")
    ud.record_answer("associated", "no blood, no fever")
    assert routing.next_stage(ud) == "recommend"


def test_stage_escalate_overrides_everything_on_red_flag():
    ud = _ud(chief_complaint="cough")
    ud.record_answer("duration", "two days")
    ud.record_answer("severity", "mild")
    ud.red_flag = detect_redflag("I cannot breathe")
    assert routing.next_stage(ud) == "escalate"


def test_escalation_never_routes_to_medicine():
    ud = _ud(chief_complaint="chest discomfort", urgency=routing.URGENT)
    assert routing.should_escalate(ud)
    assert routing.next_stage(ud) != "recommend"


# --------------------------------------------------------- session state ---


def test_unanswered_slots_shrink_as_answers_arrive():
    ud = _ud()
    assert set(ud.unanswered_slots()) == set(CORE_SLOTS)
    ud.record_answer("duration", "3 days")
    assert "duration" not in ud.unanswered_slots()


def test_blank_answers_are_ignored():
    ud = _ud()
    ud.record_answer("duration", "   ")
    assert "duration" not in ud.answers


def test_next_questions_prefers_kb_candidates():
    ud = _ud(candidate_questions=["Does the pain spread to your arm?"])
    questions = ud.next_questions(limit=3)
    assert questions[0] == "Does the pain spread to your arm?"


def test_clinical_summary_includes_the_safety_relevant_facts():
    ud = _ud(chief_complaint="fever")
    ud.patient.age_years = 4
    ud.patient.is_for_child = True
    ud.patient.known_conditions = ["asthma"]
    ud.record_answer("duration", "three days")
    ud.red_flag = detect_redflag("the baby is not drinking milk")
    summary = ud.clinical_summary()
    assert "fever" in summary
    assert "child" in summary
    assert "asthma" in summary
    assert "RED FLAG" in summary


# ------------------------------------------------------- duration parsing ---


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("since this morning", 0),
        ("yesterday", 1),
        ("3 days", 3),
        ("two weeks", 14),
        ("1 month", 30),
        ("a few hours", 0),
        ("no idea", None),
    ],
)
def test_parse_duration_days(text, expected):
    assert _parse_duration_days(text) == expected
