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


# ------------------------------------------- native script (audit finding) ---
# A Hindi caller matched no KB entry at all, which silently removed the curated
# questions, self-care text, referral advice and severity scoring - and left
# allowed_otc_classes as None, meaning *unrestricted*. An antacid was recommended
# for a urinary infection that way. Two causes: no native-script aliases, and a
# tokenizer that split Indic words at their vowel marks.


@pytest.mark.parametrize(
    "complaint,expected",
    [
        ("मुझे बुखार है", "fever"),
        ("मुझे सिर दर्द है", "headache"),
        ("मुझे पेशाब में जलन है", "urinary_symptoms"),
        ("सीने में दर्द", "chest_pain"),
        ("सांस लेने में तकलीफ", "breathlessness"),
        ("எனக்கு காய்ச்சல்", "fever"),
        ("வயிற்றுப்போக்கு", "diarrhoea"),
        ("மார்பு வலி", "chest_pain"),
        ("నాకు జ్వరం", "fever"),
        ("కడుపు నొప్పి", "abdominal_pain"),
        ("ನನಗೆ ಜ್ವರ", "fever"),
        ("ಹೊಟ್ಟೆ ನೋವು", "abdominal_pain"),
        ("എനിക്ക് പനി", "fever"),
        ("വയറിളക്കം", "diarrhoea"),
    ],
)
def test_a_native_script_complaint_matches_its_entry(complaint, expected):
    from knowledge.triage_kb import get_triage_kb

    match = get_triage_kb().match(complaint)
    assert match is not None, f"{complaint} matched nothing"
    assert match.id == expected


def test_indic_words_survive_tokenisation():
    r"""`\w` does not cover Indic vowel signs, so the old pattern split words at
    every matra: "मुझे बुखार है" became ['म','झ','ब','ख','र','ह']."""
    from knowledge.triage_kb import _tokenize

    assert _tokenize("मुझे बुखार है") == ["मुझे", "बुखार", "है"]
    assert _tokenize("எனக்கு காய்ச்சல்") == ["எனக்கு", "காய்ச்சல்"]


def test_a_complaint_that_matches_nothing_forbids_every_medicine():
    """Fail closed. A miss used to leave allowed_otc_classes as None, which the
    formulary reads as "no restriction" - every medicine became eligible for a
    complaint nothing was understood about."""
    from knowledge.triage_kb import apply_to_session
    from session_state import MedLinkUserData

    ud = MedLinkUserData(call_id="t", caller_phone=None, channel="web")
    assert apply_to_session(ud, "my hair is greying and I feel unlucky") is None
    assert ud.allowed_otc_classes == set()


def test_the_urinary_complaint_that_returned_an_antacid():
    """The exact regression, end to end, in the language that broke it."""
    from knowledge.triage_kb import apply_to_session
    from medicine.filter import recommend
    from session_state import MedLinkUserData

    ud = MedLinkUserData(call_id="t", caller_phone=None, channel="web")
    entry = apply_to_session(ud, "मुझे पेशाब में जलन है")
    assert entry is not None and entry.id == "urinary_symptoms"
    result = recommend("burning urine", ud.patient, allowed_classes=ud.allowed_otc_classes)
    assert result.recommendations == []


def test_every_entry_says_what_not_to_do():
    """Callers act on the don'ts as much as the dos - putting ash on a wound,
    stopping food during loose motions, walking off chest pain. Seven entries
    carried no "do not" at all, so the agent had nothing to warn them with.
    """
    import re

    from knowledge.triage_kb import get_triage_kb

    negative = re.compile(r"\b(do not|don't|avoid|never)\b", re.IGNORECASE)
    missing = [
        e.id for e in get_triage_kb().entries if not negative.search(e.self_care or "")
    ]
    assert not missing, f"no what-not-to-do for: {missing}"


# ------------------------------- severity modifiers (audit finding) ---------
# `_modifier_matches` fell back to "does every word of the key appear ANYWHERE
# in the transcript". That made `no_urine` fire on a child who was passing urine
# normally - it took "no" from "no blood" and "urine" from "still passing urine"
# - scoring +6 and pushing a mild case toward an ambulance.


def _assess(complaint: str, age: int, answers: dict[str, str]):
    from knowledge.triage_kb import apply_to_session
    from session_state import MedLinkUserData
    from workflows import routing

    ud = MedLinkUserData(call_id="t", caller_phone=None, channel="web")
    ud.chief_complaint = complaint
    ud.patient.age_years = age
    ud.patient.is_for_child = age < 12
    apply_to_session(ud, complaint)
    for slot, text in answers.items():
        ud.record_answer(slot, text)
    apply_to_session(ud)
    severity, _ = routing.assess(ud)
    return severity, routing.should_escalate(ud)


def test_a_well_hydrated_child_with_mild_diarrhoea_is_not_escalated():
    severity, escalate = _assess(
        "my daughter has loose motions",
        6,
        {
            "duration": "Since this morning.",
            "severity": "About five times today.",
            "associated": "No blood, no vomiting, she is drinking water "
            "and still passing urine.",
        },
    )
    assert not escalate, f"escalated at severity {severity}"


@pytest.mark.parametrize(
    "associated",
    [
        "There is blood in the stool and she cannot keep water down.",
        "She has passed no urine since yesterday.",
    ],
)
def test_a_child_with_a_real_danger_sign_is_still_escalated(associated):
    """The fix must not have blunted the modifiers it was meant to keep."""
    severity, escalate = _assess(
        "my daughter has loose motions",
        6,
        {"duration": "Two days.", "severity": "Many times.", "associated": associated},
    )
    assert escalate, f"not escalated, severity {severity}"


def test_a_modifier_phrase_still_matches_with_a_filler_word():
    """"blood in stool" has to match "blood in THE stool"."""
    severity, _ = _assess(
        "loose motions",
        30,
        {
            "duration": "Two days.",
            "severity": "Many times.",
            "associated": "I saw blood in the stool this morning.",
        },
    )
    assert severity >= 5, "blood in the stool did not score"


def test_a_modifier_phrase_cannot_reach_across_a_comma():
    """"no vomiting, urine is fine" must not read as "no urine"."""
    severity, escalate = _assess(
        "my daughter has loose motions",
        6,
        {
            "duration": "Since this morning.",
            "severity": "Three times.",
            "associated": "No vomiting, urine is fine.",
        },
    )
    assert not escalate, f"escalated at severity {severity}"


def test_the_hindi_full_stop_does_not_stick_to_the_last_word():
    """The danda sits inside the Indic range kept as word characters."""
    from knowledge.triage_kb import _tokenize
    from medicine.formulary import _tokenize as formulary_tokenize

    assert _tokenize("मुझे बुखार है।")[-1] == "है"
    assert formulary_tokenize("मुझे बुखार है।")[-1] == "है"


# ----------------------------------------------- matching (test calls) ---


@pytest.mark.parametrize(
    "complaint,expected",
    [
        # "burn" used to match inside "burning" and "burns".
        ("burning in my chest after meals", "acidity"),
        ("it usually burns after I eat", "acidity"),
        ("I burnt my hand on the stove", "burn"),
        ("hot water burn on my arm", "burn"),
        ("burning urine", "urinary_symptoms"),
        # An emergency mentioned second still decides the presentation.
        ("acidity and chest pain", "chest_pain"),
        ("fever and chest pain", "chest_pain"),
        ("burning chest and pain spreading to left arm", "chest_pain"),
        ("I have gas and cannot breathe properly", "breathlessness"),
        ("बुखार और सीने में दर्द", "chest_pain"),
        # ...but ordinary pairs still go to the first complaint.
        ("fever and body pain", "fever"),
    ],
)
def test_complaints_reach_the_right_presentation(complaint, expected):
    from knowledge.triage_kb import get_triage_kb

    match = get_triage_kb().match(complaint)
    assert match is not None and match.id == expected, getattr(match, "id", None)


def test_filler_words_do_not_make_a_match():
    """Acidity's aliases say "burning in my chest"; "my" must not match acidity."""
    from knowledge.triage_kb import get_triage_kb

    assert get_triage_kb().match("my hair is greying and I feel unlucky") is None


@pytest.mark.parametrize(
    "complaint,expected",
    [
        # A denied symptom must not decide the presentation.
        ("मुझे दो दिन से सिर में दर्द है, पर बुखार या उल्टी जैसी कोई दिक्कत नहीं है।", "headache"),
        ("सिर में दर्द है, बुखार नहीं", "headache"),
        ("bukhar nahi hai, sir dard hai", "headache"),
        ("காய்ச்சல் இல்லை, தலைவலி இருக்கிறது", "headache"),
        ("no fever but I have a headache", "headache"),
        ("I don't have fever, just a headache", "headache"),
        ("fever, no chest pain", "fever"),
        ("I can't breathe properly", "breathlessness"),
    ],
)
def test_denied_symptoms_do_not_decide_the_presentation(complaint, expected):
    from knowledge.triage_kb import get_triage_kb

    match = get_triage_kb().match(complaint)
    assert match is not None and match.id == expected, getattr(match, "id", None)


def test_a_complaint_made_only_of_denials_matches_nothing():
    from knowledge.triage_kb import get_triage_kb

    assert get_triage_kb().match("no chest pain, no fever") is None


@pytest.mark.parametrize(
    "complaint,expected",
    [
        # A word in the middle of a phrase no longer breaks it.
        ("दो दिनों से सिर में हल्का दर्द है", "headache"),
        ("pain in my chest", "chest_pain"),
        ("I cannot properly breathe", "breathlessness"),
    ],
)
def test_a_phrase_still_matches_with_a_word_inside_it(complaint, expected):
    from knowledge.triage_kb import get_triage_kb

    match = get_triage_kb().match(complaint)
    assert match is not None and match.id == expected


def test_a_phrase_is_never_assembled_across_a_comma():
    from knowledge.triage_kb import _alias_pattern
    from safety.redflags import _normalize_clauses

    assert not _alias_pattern("chest pain").search(_normalize_clauses("chest, pain"))
    assert not _alias_pattern("सिर में दर्द").search(_normalize_clauses("सिर में, दर्द"))


def test_the_problem_is_found_in_what_the_caller_said_later():
    """The complaint the model recorded never named the symptom; the caller did,
    two answers later."""
    from knowledge.triage_kb import apply_to_session
    from session_state import MedLinkUserData

    ud = MedLinkUserData(call_id="t", caller_phone=None, channel="web")
    ud.chief_complaint = "symptoms for about a week, worse with spicy food"
    ud.heard += [
        "It started about a week ago and gets worse when I eat spicy food.",
        "Yes, it's for me. I'm 40.",
        "It's moderate, just a burning feeling in my chest after meals.",
    ]
    entry = apply_to_session(ud)
    assert entry is not None and entry.id == "acidity"
    assert ud.allowed_otc_classes == {"antacid"}
