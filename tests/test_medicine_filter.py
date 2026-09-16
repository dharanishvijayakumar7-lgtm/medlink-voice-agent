"""Behavior lock for the OTC medicine safety pipeline."""

import pytest

from medicine.filter import PatientContext, recommend
from medicine.formulary import get_formulary


@pytest.fixture
def fm():
    return get_formulary()


def _ids(result):
    return {r.entry_id for r in result.recommendations}


def test_simple_fever_returns_paracetamol(fm):
    result = recommend(
        "high fever and body pain since yesterday",
        PatientContext(age_years=30, symptom_duration_days=1),
        allowed_classes={"analgesic_antipyretic"},
        formulary=fm,
    )
    assert result.has_medicine
    assert "paracetamol_tab_500" in _ids(result)
    spoken = result.to_spoken().lower()
    assert "paracetamol" in spoken
    assert "not a doctor" in spoken  # disclaimer always present


def test_recommendation_text_is_built_from_structured_fields(fm):
    result = recommend(
        "acidity and heartburn after food",
        PatientContext(age_years=40),
        allowed_classes={"antacid"},
        formulary=fm,
    )
    assert result.has_medicine
    rec = result.recommendations[0]
    entry = fm.by_id[rec.entry_id]
    # dose text must be verbatim from the formulary, not invented
    assert entry.adult_dose in rec.spoken_text
    assert "available at any pharmacy" in rec.spoken_text.lower()


def test_empty_allowed_classes_means_no_otc(fm):
    result = recommend(
        "chest discomfort",
        PatientContext(age_years=55),
        allowed_classes=set(),
        formulary=fm,
    )
    assert not result.has_medicine
    assert result.escalate
    assert "doctor" in result.no_medicine_reason.lower()


# test_redflag_symptom_never_gets_a_medicine was removed with the red-flag layer.
# An emergency presentation now reaches the normal recommendation path.


def test_no_confident_match_gives_safe_fallback(fm):
    result = recommend(
        "my hair has been falling a lot lately",
        PatientContext(age_years=25),
        formulary=fm,
    )
    assert not result.has_medicine
    assert result.no_medicine_reason is not None
    assert not result.escalate


def test_age_filter_blocks_underage(fm):
    # Ibuprofen entry has min_age_years 16
    result = recommend(
        "strong sprain pain in my ankle",
        PatientContext(age_years=10, is_for_child=True),
        allowed_classes={"nsaid"},
        formulary=fm,
    )
    assert "ibuprofen_tab_400" not in _ids(result)
    assert any("age" in reason for _, reason in result.rejected)


def test_pregnancy_avoid_is_rejected(fm):
    result = recommend(
        "bad sprain pain and swelling",
        PatientContext(age_years=28, is_pregnant=True),
        allowed_classes={"nsaid"},
        formulary=fm,
    )
    assert "ibuprofen_tab_400" not in _ids(result)
    assert any("pregnan" in reason for _, reason in result.rejected)


def test_pregnancy_caution_keeps_medicine_but_adds_caution(fm):
    result = recommend(
        "sneezing and itchy runny nose from allergy",
        PatientContext(age_years=26, is_pregnant=True),
        allowed_classes={"antihistamine"},
        formulary=fm,
    )
    assert result.has_medicine
    assert any(
        "pregnant" in c.lower() for r in result.recommendations for c in r.cautions
    )


def test_contraindication_matches_disclosed_condition(fm):
    # Ibuprofen contraindicated with stomach ulcer / GI bleeding history
    result = recommend(
        "body pain and mild fever not settling",
        PatientContext(
            age_years=45,
            known_conditions=["stomach ulcer"],
            symptom_duration_days=1,
        ),
        allowed_classes={"nsaid"},
        formulary=fm,
    )
    assert "ibuprofen_tab_400" not in _ids(result)
    assert any("contraindicated" in reason for _, reason in result.rejected)


def test_reported_symptoms_are_checked_against_contraindications(fm):
    """A symptom the caller described must block a drug, not just a formal
    'known condition'. Loperamide is dangerous in dysentery."""
    query = "control diarrhoea stop loose motion frequent watery stools"
    bloody = recommend(
        query,
        PatientContext(
            age_years=30,
            symptom_duration_days=1,
            reported_symptoms=["blood in the stool"],
        ),
        allowed_classes={"antidiarrheal"},
        formulary=fm,
    )
    assert "loperamide_2" not in _ids(bloody)
    assert any("contraindicated" in reason for _, reason in bloody.rejected)


def test_fever_with_diarrhoea_blocks_loperamide(fm):
    result = recommend(
        "control diarrhoea stop loose motion frequent watery stools",
        PatientContext(
            age_years=30,
            symptom_duration_days=1,
            reported_symptoms=["loose motions with fever"],
        ),
        allowed_classes={"antidiarrheal"},
        formulary=fm,
    )
    assert "loperamide_2" not in _ids(result)


def test_uncomplicated_diarrhoea_still_allows_loperamide(fm):
    """The safety check must not be so broad that it blocks valid use."""
    result = recommend(
        "control diarrhoea stop loose motion frequent watery stools",
        PatientContext(
            age_years=30,
            symptom_duration_days=1,
            reported_symptoms=["watery stools only"],
        ),
        allowed_classes={"antidiarrheal"},
        formulary=fm,
    )
    assert "loperamide_2" in _ids(result)


def test_interaction_with_current_medication_is_rejected(fm):
    result = recommend(
        "strong body pain and sprain",
        PatientContext(
            age_years=50,
            current_medications=["warfarin"],
            symptom_duration_days=1,
        ),
        allowed_classes={"nsaid"},
        formulary=fm,
    )
    assert "ibuprofen_tab_400" not in _ids(result)
    assert any("interaction" in reason for _, reason in result.rejected)


def test_duration_beyond_self_care_limit_triggers_escalation(fm):
    result = recommend(
        "headache every day",
        PatientContext(age_years=35, symptom_duration_days=20),
        allowed_classes={"analgesic_antipyretic"},
        formulary=fm,
    )
    assert result.escalate
    assert "paracetamol_tab_500" not in _ids(result)


def test_only_truly_otc_entries_are_recommended(fm):
    # Every entry that survives must be genuinely non-prescription in India.
    result = recommend(
        "loose motions since this morning",
        PatientContext(age_years=30, symptom_duration_days=0),
        formulary=fm,
    )
    for rec in result.recommendations:
        assert fm.by_id[rec.entry_id].is_truly_otc


def test_max_results_is_respected(fm):
    result = recommend(
        "cough cold blocked nose sneezing sore throat",
        PatientContext(age_years=30),
        formulary=fm,
    )
    assert len(result.recommendations) <= 2


def test_known_names_populated_for_allow_list_gate(fm):
    names = fm.known_names()
    assert "paracetamol" in names
    assert "cetirizine" in names
    assert len(names) > 20


def test_never_recommends_two_products_with_the_same_ingredient(fm):
    """Double-dosing paracetamol is a classic accidental overdose route."""
    result = recommend(
        "fever and body pain",
        PatientContext(age_years=34, symptom_duration_days=2),
        allowed_classes={"analgesic_antipyretic"},
        formulary=fm,
    )
    ingredients = [
        i.name.casefold()
        for rec in result.recommendations
        for i in fm.by_id[rec.entry_id].active_ingredients
    ]
    assert len(ingredients) == len(set(ingredients)), (
        f"same active ingredient offered twice: {ingredients}"
    )


# ----------------------------------------------------- audit regressions ---
# Found by the end-to-end audit. Each of these returned the wrong thing before.


def test_a_urinary_complaint_does_not_return_an_antacid(fm):
    """"burning urine" matched this antacid through the single word "burning",
    which reaches it via the lay term "burning in chest after food"."""
    hits = fm.search("burning urine")
    assert [e.id for e, _ in hits] == []


def test_a_bare_fever_query_returns_the_adult_tablet(fm):
    """The paediatric syrup used to win: its indications repeat "children fever",
    which scores higher on BM25 than the tablet's single "fever"."""
    hits = fm.search("fever", limit=1)
    assert hits and hits[0][0].id == "paracetamol_tab_500"


def test_a_child_fever_query_returns_the_syrup(fm):
    hits = fm.search("my child has fever", limit=1)
    assert hits and hits[0][0].id == "paracetamol_syrup_250"


def test_a_medicine_asked_for_by_name_resolves(fm):
    """Callers ask by name. BM25 scored "paracetamol" 1.89 and "crocin" 2.35,
    both under the old 2.5 threshold, so neither resolved at all."""
    for query in ("paracetamol", "crocin", "dolo 650"):
        hits = fm.search(query, limit=1)
        assert hits and hits[0][0].id == "paracetamol_tab_500", query


def test_ors_resolves_to_rehydration_salts_not_zinc(fm):
    hits = fm.search("ORS", limit=1)
    assert hits and hits[0][0].id == "ors_who"


def test_an_unrelated_complaint_still_matches_nothing(fm):
    """The phrase gate must not have loosened anything."""
    for query in ("hair falling", "my hair is greying", "I cannot sleep"):
        assert fm.search(query) == [], query


def test_a_child_with_diarrhoea_is_offered_zinc_with_ors(fm):
    """WHO guidance, and the KB's own self-care text, pair zinc with ORS. Zinc
    was unreachable: its indication read "diarrhoea in children (given alongside
    ORS)", a clinical note rather than an indication, so relevance demanded the
    words "given" and "alongside" in the caller's complaint.
    """
    from knowledge.triage_kb import apply_to_session
    from medicine.filter import recommend
    from session_state import MedLinkUserData

    ud = MedLinkUserData(call_id="t", caller_phone=None, channel="web")
    ud.patient.age_years = 6
    ud.patient.is_for_child = True
    apply_to_session(ud, "loose motions")
    got = {
        r.entry_id
        for r in recommend(
            "my child has loose motions",
            ud.patient,
            allowed_classes=ud.allowed_otc_classes,
        ).recommendations
    }
    assert {"zinc_dispersible_20", "ors_who"} <= got


def test_a_plural_complaint_matches_a_singular_indication(fm):
    """"loose motions" must reach an entry listing "loose motion"."""
    assert fm.search("loose motion", limit=1)
    assert fm.search("loose motions", limit=1)


def test_a_dose_limit_that_is_a_sentence_is_not_prefixed():
    """Eight of twenty entries hold a sentence in max_daily_dose, and the caller
    heard "Do not exceed As much as needed to replace losses"."""
    from medicine.filter import _build_recommendation
    from medicine.formulary import get_formulary
    from session_state import PatientContext

    spoken = _build_recommendation(
        get_formulary().by_id["ors_who"], PatientContext(), []
    ).spoken_text
    assert "Do not exceed As much as needed" not in spoken
    assert "As much as needed to replace losses" in spoken


def test_a_dose_limit_that_is_a_quantity_still_says_do_not_exceed():
    from medicine.filter import _build_recommendation
    from medicine.formulary import get_formulary
    from session_state import PatientContext

    spoken = _build_recommendation(
        get_formulary().by_id["paracetamol_tab_500"], PatientContext(), []
    ).spoken_text
    assert "Do not exceed 4000 mg/day" in spoken
