"""Behaviour lock for the Firestore call-summary export.

The document builders are pure, so the exact shape the mobile app receives is
verified here without Firebase. The live write is exercised separately against a
real project.
"""

from datetime import datetime, timedelta, timezone

import pytest

import firestore_export as fx
from config import settings
from session_state import MedLinkUserData


def _full_call() -> MedLinkUserData:
    ud = MedLinkUserData(caller_phone="+919876543210", channel="pstn")
    ud.started_at = datetime(2026, 9, 14, 9, 0, tzinfo=timezone.utc)
    ud.language = "ta-IN"
    ud.patient_name = "Priya"
    ud.patient_gender = "female"
    ud.patient.age_years = 32
    ud.chief_complaint = "fever and body pain"
    ud.answers = {"duration": "2 days", "severity": "moderate"}
    ud.patient.known_conditions = ["diabetes"]
    ud.urgency = "clinic"
    ud.disposition = "self_care"
    ud.recommendations = [{"generic_name": "Paracetamol", "adult_dose": "500 mg"}]
    return ud


ENDED = datetime(2026, 9, 14, 9, 4, 30, tzinfo=timezone.utc)


# ---------------------------------------------------------------- patient id ---


@pytest.mark.parametrize(
    "raw",
    ["+919876543210", "919876543210", "09876543210", "9876543210", "+91 98765 43210"],
)
def test_every_indian_format_lands_on_one_patient_document(raw):
    assert fx.patient_id(raw) == "+919876543210"


@pytest.mark.parametrize("withheld", [None, "", "anonymous", "unknown", "+"])
def test_withheld_caller_id_has_no_patient_document(withheld):
    assert fx.patient_id(withheld) is None


# ----------------------------------------------------------------- documents ---


def test_call_document_carries_the_structured_summary():
    doc = fx.build_call_document(_full_call(), ENDED)

    assert doc["phone"] == "+919876543210"
    assert doc["channel"] == "pstn"
    assert doc["duration_sec"] == 270
    assert doc["patient"] == {"name": "Priya", "age_years": 32, "gender": "female"}
    assert doc["chief_complaint"] == "fever and body pain"
    assert doc["answers"] == {"duration": "2 days", "severity": "moderate"}
    assert doc["urgency"] == "clinic"
    assert doc["medicines_discussed"] == [
        {"generic_name": "Paracetamol", "adult_dose": "500 mg"}
    ]
    assert doc["schema_version"] == fx.SCHEMA_VERSION


def test_unknown_placeholders_never_reach_the_app():
    """The session uses the string "unknown"; the app must see an absent field."""
    ud = MedLinkUserData(caller_phone="+919876543210")
    doc = fx.build_call_document(ud, ud.started_at + timedelta(seconds=5))

    assert "urgency" not in doc
    assert "unknown" not in str(doc)
    # Empty containers are omitted rather than sent as [] / {}.
    for key in ("answers", "medicines_discussed", "known_conditions", "patient"):
        assert key not in doc


def test_false_flags_are_kept_because_false_is_meaningful():
    doc = fx.build_call_document(MedLinkUserData(), ENDED)
    assert doc["escalated"] is False
    assert doc["returning_caller"] is False


def test_summary_text_reads_as_english_from_captured_fields():
    text = fx.build_summary_text(_full_call())
    assert text.startswith("Priya (female, 32) called in Tamil about: fever and body pain.")
    assert "Duration: 2 days. Severity: moderate." in text
    assert "Known conditions: diabetes." in text
    assert "Assessed urgency: clinic." in text
    assert "Medicines discussed: Paracetamol." in text


def test_summary_text_skips_what_was_never_captured():
    assert fx.build_summary_text(MedLinkUserData()) == (
        "Caller called; no complaint was recorded."
    )


def test_patient_fields_never_blank_out_earlier_details():
    """merge=True + omitted blanks => a later call can't erase a known name."""
    ud = MedLinkUserData(caller_phone="+919876543210")
    doc = fx.build_call_document(ud, ENDED)
    fields = fx.build_patient_fields(ud, doc)
    assert "name" not in fields
    assert "gender" not in fields
    assert fields["phone"] == "+919876543210"


# ---------------------------------------------------------------- assessment ---


def test_assessment_keeps_curated_and_ai_suggested_apart():
    ud = _full_call()
    ud.triage_entry_id = "headache"
    ud.self_care_advice = "Rest,   drink water."
    ud.refer_when = "Worst headache of your life."
    ud.possible_causes = ["tension headache", "dehydration"]
    ud.possible_causes_reasoning = "Started after a long day with little water."

    assessment = fx.build_call_document(ud, ENDED)["assessment"]
    assert assessment["category"]  # resolved from the curated triage KB
    assert assessment["self_care_advice"] == "Rest, drink water."
    assert assessment["see_doctor_if"] == "Worst headache of your life."
    assert assessment["possible_causes"] == ["tension headache", "dehydration"]
    assert assessment["possible_causes_note"] == fx.POSSIBLE_CAUSES_NOTE


def test_no_ai_causes_means_no_not_a_diagnosis_note():
    ud = _full_call()
    ud.triage_entry_id = "headache"
    assessment = fx.build_call_document(ud, ENDED)["assessment"]
    assert "possible_causes" not in assessment
    assert "possible_causes_note" not in assessment


def test_unknown_kb_entry_does_not_break_the_document():
    ud = _full_call()
    ud.triage_entry_id = "no-such-entry"
    doc = fx.build_call_document(ud, ENDED)
    assert "category" not in doc.get("assessment", {})


def test_summary_text_labels_ai_causes_as_not_a_diagnosis():
    ud = _full_call()
    ud.possible_causes = ["tension headache"]
    assert (
        "Possible causes (AI-suggested, not a diagnosis): tension headache."
        in fx.build_summary_text(ud)
    )


# ---------------------------------------------------------------- write plan ---


def _plan(pid="+919876543210", *, call_exists=False, patient_exists=False):
    ud = _full_call()
    doc = fx.build_call_document(ud, ENDED)
    return fx.plan_writes(
        pid,
        doc,
        fx.build_patient_fields(ud, doc),
        call_exists=call_exists,
        patient_exists=patient_exists,
    ), doc


def test_first_call_creates_the_call_and_the_patient():
    writes, doc = _plan()
    assert [w.path for w in writes] == [
        ("patients", "+919876543210", "calls", doc["call_id"]),
        ("patients", "+919876543210"),
    ]
    patient = writes[1]
    assert patient.merge is True
    assert patient.data["total_calls"] is fx.INCREMENT_TOTAL_CALLS
    assert patient.data["first_seen_at"] == doc["started_at"]


def test_returning_patient_keeps_their_first_seen_date():
    writes, _ = _plan(patient_exists=True)
    assert "first_seen_at" not in writes[1].data
    assert writes[1].data["total_calls"] is fx.INCREMENT_TOTAL_CALLS


def test_re_exporting_the_same_call_writes_nothing():
    """A shutdown callback that runs twice must not double-count total_calls."""
    writes, _ = _plan(call_exists=True)
    assert writes == []


def test_withheld_caller_goes_to_unidentified_calls():
    writes, doc = _plan(pid=None)
    assert [w.path for w in writes] == [("unidentified_calls", doc["call_id"])]


# ------------------------------------------------------------------- export ---


async def test_export_is_off_unless_enabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_firestore_export", False)
    assert await fx.export_call(_full_call()) is False


async def test_export_skips_when_credentials_are_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "enable_firestore_export", True)
    monkeypatch.setattr(settings, "firebase_credentials_path", tmp_path / "absent.json")
    assert await fx.export_call(_full_call()) is False


async def test_export_respects_the_consent_gate(monkeypatch, tmp_path):
    creds = tmp_path / "key.json"
    creds.write_text("{}")
    monkeypatch.setattr(settings, "enable_firestore_export", True)
    monkeypatch.setattr(settings, "firebase_credentials_path", creds)
    monkeypatch.setattr(settings, "require_consent", True)
    ud = _full_call()
    ud.consent_store = None
    assert await fx.export_call(ud) is False


async def test_export_commits_the_built_documents(monkeypatch, tmp_path):
    creds = tmp_path / "key.json"
    creds.write_text("{}")
    monkeypatch.setattr(settings, "enable_firestore_export", True)
    monkeypatch.setattr(settings, "firebase_credentials_path", creds)
    monkeypatch.setattr(settings, "require_consent", False)

    seen = {}

    def fake_commit(pid, call_doc, patient_fields):
        seen.update(pid=pid, call_doc=call_doc, patient_fields=patient_fields)
        return True

    monkeypatch.setattr(fx, "_commit", fake_commit)
    assert await fx.export_call(_full_call()) is True
    assert seen["pid"] == "+919876543210"
    assert seen["call_doc"]["summary_text"].startswith("Priya")
    assert seen["patient_fields"]["last_urgency"] == "clinic"


async def test_a_firestore_failure_never_raises(monkeypatch, tmp_path):
    """export_call runs in the shutdown callback; it must swallow errors."""
    creds = tmp_path / "key.json"
    creds.write_text("{}")
    monkeypatch.setattr(settings, "enable_firestore_export", True)
    monkeypatch.setattr(settings, "firebase_credentials_path", creds)
    monkeypatch.setattr(settings, "require_consent", False)

    def boom(*_):
        raise RuntimeError("firestore down")

    monkeypatch.setattr(fx, "_commit", boom)
    assert await fx.export_call(_full_call()) is False
