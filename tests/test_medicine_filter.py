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


def test_redflag_symptom_never_gets_a_medicine(fm):
    result = recommend(
        "crushing chest pain spreading to my arm",
        PatientContext(age_years=60),
        formulary=fm,
    )
    assert not result.has_medicine
    assert result.escalate
    assert "urgent" in result.no_medicine_reason.lower()


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
