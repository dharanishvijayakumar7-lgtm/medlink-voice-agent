"""Behavior lock for the triage knowledge base and its effect on the session."""

import pytest

from knowledge.triage_kb import apply_to_session, get_triage_kb, score_modifiers
from medicine.formulary import get_formulary
from session_state import MedLinkUserData
from workflows import routing


@pytest.fixture
def kb():
    return get_triage_kb()


def _ud(**kwargs) -> MedLinkUserData:
    ud = MedLinkUserData()
    for key, value in kwargs.items():
        setattr(ud, key, value)
    return ud


# ------------------------------------------------------------- integrity ---


def test_kb_loads_and_has_entries(kb):
    assert len(kb.entries) >= 15


def test_every_allowed_otc_class_exists_in_the_formulary(kb):
    """A KB entry must never authorise a therapeutic class we cannot dispense."""
    known = {e.therapeutic_class for e in get_formulary().entries}
    for entry in kb.entries:
        unknown = set(entry.otc_categories_allowed) - known
        assert not unknown, f"{entry.id} allows unknown classes: {unknown}"


def test_every_entry_has_questions_and_referral_criteria(kb):
    for entry in kb.entries:
        assert entry.candidate_questions, f"{entry.id} has no candidate questions"
        assert entry.refer_when, f"{entry.id} has no refer_when"
        assert entry.aliases, f"{entry.id} has no aliases"


# -------------------------------------------------------------- matching ---


@pytest.mark.parametrize(
    ("complaint", "expected_id"),
    [
        ("I have had a fever since two days", "fever"),
        ("bad cough and cold with phlegm", "cough_cold"),
        ("loose motions since morning", "diarrhoea"),
        ("my stomach is running", "diarrhoea"),
        ("burning in chest after food, acidity", "acidity"),
        ("a round itchy patch on my leg, looks like ringworm", "fungal_skin"),
        ("chest pain when I walk", "chest_pain"),
        ("I cannot breathe properly", "breathlessness"),
        ("burning urine since yesterday", "urinary_symptoms"),
        ("cut my hand on a blade", "wound_cut"),
        ("bukhar hai do din se", "fever"),
        ("pet dard ho raha hai", "abdominal_pain"),
    ],
)
def test_complaints_match_the_right_entry(kb, complaint, expected_id):
    entry = kb.match(complaint)
    assert entry is not None, f"no match for {complaint!r}"
    assert entry.id == expected_id


def test_unrelated_complaint_matches_nothing(kb):
    assert kb.match("my mobile phone is not charging") is None


def test_empty_complaint_matches_nothing(kb):
    assert kb.match("") is None
    assert kb.match("   ") is None


def test_longer_alias_wins_over_shorter(kb):
    # "chest pain" must beat a bare "pain" style match.
    assert kb.match("chest pain").id == "chest_pain"


# --------------------------------------------------- OTC authorisation ---


def test_serious_presentations_authorise_no_otc_medicine(kb):
    for entry_id in (
        "chest_pain",
        "breathlessness",
        "urinary_symptoms",
        "constipation",
    ):
        entry = kb.by_id[entry_id]
        assert entry.otc_categories_allowed == []
        assert not entry.allows_otc
        assert kb.allowed_otc_classes(entry) == set()


def test_no_kb_match_leaves_otc_search_unrestricted(kb):
    assert kb.allowed_otc_classes(None) is None


def test_fever_authorises_only_safe_classes(kb):
    allowed = kb.allowed_otc_classes(kb.by_id["fever"])
    assert "analgesic_antipyretic" in allowed
    assert "nsaid" not in allowed  # not first-line for undifferentiated fever


# ------------------------------------------------------------- modifiers ---


def test_structural_modifiers_are_evaluated(kb):
    entry = kb.by_id["fever"]
    young = _ud()
    young.patient.age_years = 2
    adult = _ud()
    adult.patient.age_years = 30
    assert score_modifiers(entry, young) > score_modifiers(entry, adult)


def test_duration_modifier_is_evaluated(kb):
    entry = kb.by_id["fever"]
    short = _ud()
    short.patient.symptom_duration_days = 1
    long = _ud()
    long.patient.symptom_duration_days = 9
    assert score_modifiers(entry, long) > score_modifiers(entry, short)


def test_keyword_modifiers_read_the_answers(kb):
    entry = kb.by_id["diarrhoea"]
    plain = _ud()
    plain.record_answer("associated", "just watery motions")
    bloody = _ud()
    bloody.record_answer("associated", "there is blood in stool")
    assert score_modifiers(entry, bloody) > score_modifiers(entry, plain)


def test_denied_symptoms_add_no_severity(kb):
    """From a simulated call: "No fever, no neck stiffness" scored
    with_neck_stiffness (+8) and turned a mild tension headache into an
    emergency. The agent now asks exactly this question on most headache calls."""
    ud = _ud(chief_complaint="moderate headache since this morning")
    ud.record_answer("duration", "since this morning, built up slowly")
    ud.record_answer("associated", "No fever, no neck stiffness, no vomiting")
    ud.record_answer("history", "first time having a headache like this")
    assert score_modifiers(kb.by_id["headache"], ud) == 0


def test_affirmed_symptom_still_adds_severity(kb):
    ud = _ud(chief_complaint="headache")
    ud.record_answer("associated", "yes, fever and neck stiffness since last night")
    assert score_modifiers(kb.by_id["headache"], ud) > 0


def test_a_denial_in_one_answer_does_not_cancel_another(kb):
    """Each answer is its own clause: "no fever" must not negate a later answer."""
    ud = _ud(chief_complaint="headache")
    ud.record_answer("associated", "no fever")
    ud.record_answer("history", "neck stiffness started today")
    assert score_modifiers(kb.by_id["headache"], ud) > 0


def test_mild_headache_with_denied_warning_signs_is_not_an_emergency(kb):
    ud = _ud(chief_complaint="moderate headache since this morning")
    ud.record_answer("duration", "since this morning")
    ud.record_answer("severity", "moderate")
    ud.record_answer("associated", "No fever, no neck stiffness, no vomiting")
    apply_to_session(ud)
    _, urgency = routing.assess(ud)
    assert urgency not in (routing.URGENT, routing.EMERGENCY)


# ------------------------------------------------- session integration ---


def test_apply_to_session_populates_guidance():
    ud = _ud()
    entry = apply_to_session(ud, "loose motions since this morning")
    assert entry is not None and entry.id == "diarrhoea"
    assert ud.triage_entry_id == "diarrhoea"
    assert ud.candidate_questions
    assert ud.allowed_otc_classes == {
        "oral_rehydration",
        "diarrhoea_adjunct",
        "antidiarrheal",
    }
    assert ud.self_care_advice
    assert ud.refer_when


def test_apply_to_session_blocks_otc_for_chest_pain():
    ud = _ud()
    apply_to_session(ud, "I have chest pain")
    assert ud.allowed_otc_classes == set()  # explicitly: no medicine


def test_apply_to_session_is_idempotent():
    ud = _ud()
    ud.patient.age_years = 2
    apply_to_session(ud, "fever for two days")
    first = ud.kb_severity_bonus
    apply_to_session(ud, "fever for two days")
    assert ud.kb_severity_bonus == first  # recomputed, never accumulated


def test_no_match_clears_the_severity_bonus():
    ud = _ud(kb_severity_bonus=5)
    assert apply_to_session(ud, "my phone is broken") is None
    assert ud.kb_severity_bonus == 0


def test_kb_bonus_feeds_the_severity_score():
    ud = _ud()
    ud.patient.age_years = 2
    ud.patient.symptom_duration_days = 9
    apply_to_session(ud, "fever")
    severity, urgency = routing.assess(ud)
    assert ud.kb_severity_bonus > 0
    assert severity >= ud.kb_severity_bonus
    assert urgency in (routing.URGENT, routing.EMERGENCY, routing.CLINIC)
