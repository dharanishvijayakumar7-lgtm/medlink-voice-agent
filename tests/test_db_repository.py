"""Behavior lock for call/patient history persistence.

Runs against SQLite (via aiosqlite) so the suite needs no Docker or Postgres.
The models avoid PG-specific column types precisely so this works.
"""

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select

from config import DEFAULT_LANGUAGE_CODE, settings
from db import repository as repo
from db import session as db_session
from db.crypto import hash_phone, normalise_phone
from db.models import (
    AuditLog,
    Call,
    CallAnswer,
    Consent,
    Escalation,
    MedicalHistory,
    Message,
    Symptom,
    TriageAssessment,
    User,
)
from db.models import Medication as MedRow
from safety.redflags import detect_redflag
from session_state import MedLinkUserData

PHONE = "+91 98765 43210"


@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "enable_db", True)
    monkeypatch.setattr(
        settings, "database_url", f"sqlite+aiosqlite:///{tmp_path / 'medlink.db'}"
    )
    monkeypatch.setattr(settings, "phone_hash_key", "unit-test-hash-key")
    monkeypatch.setattr(
        settings, "field_encryption_key", Fernet.generate_key().decode()
    )
    from db import crypto

    crypto._fernet.cache_clear()
    db_session.reset_for_tests()
    await db_session.create_all()
    yield
    await db_session.get_engine().dispose()
    db_session.reset_for_tests()
    crypto._fernet.cache_clear()


def _ud(**kwargs) -> MedLinkUserData:
    ud = MedLinkUserData(caller_phone=PHONE, channel="pstn")
    for key, value in kwargs.items():
        setattr(ud, key, value)
    return ud


async def _rows(model):
    async with db_session.session_scope() as session:
        return list((await session.execute(select(model))).scalars().all())


# ------------------------------------------------------------- call start ---


async def test_start_call_creates_user_and_call(db):
    ud = _ud()
    await repo.start_call(ud)

    users = await _rows(User)
    calls = await _rows(Call)
    assert len(users) == 1
    assert len(calls) == 1
    assert calls[0].session_id == ud.call_id
    assert calls[0].channel == "pstn"
    assert ud.user_id is not None


async def test_raw_phone_number_is_never_stored(db):
    await repo.start_call(_ud())
    users = await _rows(User)
    digits = normalise_phone(PHONE)
    assert users[0].phone_hash == hash_phone(PHONE)
    assert digits not in users[0].phone_hash
    # No consent yet, so not even the encrypted copy exists.
    assert users[0].phone_enc is None


async def test_returning_caller_is_recognised(db):
    first = _ud()
    await repo.start_call(first)
    await repo.finish_call(first)

    second = _ud()
    await repo.start_call(second)

    assert second.is_returning_caller is True
    assert second.user_id == first.user_id
    assert len(await _rows(User)) == 1


async def test_first_time_caller_is_not_marked_returning(db):
    ud = _ud()
    await repo.start_call(ud)
    assert ud.is_returning_caller is False
    assert ud.previous_summary is None


async def test_previous_summary_only_recalled_with_consent(db):
    first = _ud(chief_complaint="fever for two days", consent_store=True)
    await repo.start_call(first)
    await repo.record_consent(first, "store", True)
    first.consent_store = True
    await repo.finish_call(first)

    second = _ud()
    await repo.start_call(second)
    assert second.previous_summary is not None
    assert "fever" in second.previous_summary


async def test_previous_summary_withheld_without_consent(db):
    first = _ud(chief_complaint="fever for two days")
    await repo.start_call(first)
    await repo.finish_call(first)  # consent_store defaults False

    second = _ud()
    await repo.start_call(second)
    assert second.previous_summary is None


async def test_anonymous_caller_creates_a_call_but_no_user(db):
    ud = MedLinkUserData(caller_phone=None, channel="web")
    await repo.start_call(ud)
    assert len(await _rows(User)) == 0
    assert len(await _rows(Call)) == 1


# ------------------------------------------------------------ consent gate ---


async def test_turns_are_not_stored_without_consent(db):
    ud = _ud()
    await repo.start_call(ud)
    await repo.record_turn(ud, "user", "I have a fever")
    assert await _rows(Message) == []


async def test_turns_are_stored_with_consent(db):
    ud = _ud(consent_store=True)
    await repo.start_call(ud)
    ud.consent_store = True
    await repo.record_turn(ud, "user", "I have a fever", language="hi-IN")
    messages = await _rows(Message)
    assert len(messages) == 1
    assert messages[0].text_original == "I have a fever"
    assert messages[0].language == "hi-IN"


async def test_consent_is_logged_and_mirrored_onto_the_user(db):
    ud = _ud()
    await repo.start_call(ud)
    await repo.record_consent(ud, "store", True)

    consents = await _rows(Consent)
    users = await _rows(User)
    assert len(consents) == 1
    assert consents[0].kind == "store" and consents[0].granted is True
    assert users[0].consent_store is True
    # Consent unlocks storing the reversible copy for callbacks.
    assert users[0].phone_enc is not None


async def test_clinical_detail_withheld_without_consent(db):
    ud = _ud(chief_complaint="loose motions")
    ud.record_answer("duration", "two days")
    await repo.start_call(ud)
    await repo.finish_call(ud)

    calls = await _rows(Call)
    assert calls[0].chief_complaint is None
    assert calls[0].summary_en is None
    assert await _rows(CallAnswer) == []


async def test_dev_bypass_stores_clinical_detail_without_consent(db, monkeypatch):
    """MEDLINK_REQUIRE_CONSENT=false is the local-development escape hatch.

    It exists so there is data to inspect before the spoken consent flow is
    built. Consent must still be the default (see conftest); this pins that the
    bypass genuinely works when deliberately enabled.
    """
    monkeypatch.setattr(settings, "require_consent", False)
    ud = _ud(chief_complaint="loose motions")
    ud.record_answer("duration", "two days")
    await repo.start_call(ud)
    await repo.record_turn(ud, "user", "I have loose motions", "en-IN")
    await repo.finish_call(ud)

    calls = await _rows(Call)
    assert calls[0].chief_complaint == "loose motions"
    assert calls[0].summary_en is not None
    assert len(await _rows(CallAnswer)) == 1
    assert len(await _rows(Message)) == 1


async def test_clinical_detail_stored_with_consent(db):
    ud = _ud(chief_complaint="loose motions", consent_store=True)
    ud.record_answer("duration", "two days")
    await repo.start_call(ud)
    ud.consent_store = True
    await repo.finish_call(ud)

    calls = await _rows(Call)
    answers = await _rows(CallAnswer)
    assert calls[0].chief_complaint == "loose motions"
    assert calls[0].summary_en is not None
    assert len(answers) == 1 and answers[0].slot == "duration"


# --------------------------------------------------------------- call end ---


async def test_triage_outcome_is_always_recorded(db):
    """Operational/clinical-safety record is kept even without content consent."""
    ud = _ud(urgency="urgent", severity_score=7, triage_entry_id="fever")
    await repo.start_call(ud)
    await repo.finish_call(ud)

    assessments = await _rows(TriageAssessment)
    calls = await _rows(Call)
    assert len(assessments) == 1
    assert assessments[0].urgency == "urgent"
    assert assessments[0].severity_score == 7
    assert calls[0].ended_at is not None


async def test_emergency_red_flag_is_persisted(db):
    ud = _ud(escalated=True, urgency="emergency")
    ud.red_flag = detect_redflag("I have crushing chest pain")
    await repo.start_call(ud)
    await repo.finish_call(ud)

    calls = await _rows(Call)
    assessments = await _rows(TriageAssessment)
    escalations = await _rows(Escalation)
    assert calls[0].is_emergency is True
    assert assessments[0].red_flag_category == "cardiac"
    assert len(escalations) == 1


async def test_medicine_recommendations_are_persisted_with_audit_detail(db):
    ud = _ud(
        recommendations=[
            {
                "id": "paracetamol_tab_500",
                "generic_name": "Paracetamol",
                "adult_dose": "1 tablet every 4 to 6 hours",
            }
        ]
    )
    await repo.start_call(ud)
    await repo.finish_call(ud)

    rows = await _rows(MedRow)
    assert len(rows) == 1
    assert rows[0].formulary_id == "paracetamol_tab_500"
    assert rows[0].dose_text == "1 tablet every 4 to 6 hours"
    assert rows[0].details["generic_name"] == "Paracetamol"


async def test_medicine_source_distinguishes_advice_from_what_the_caller_takes(db):
    """The safety-critical distinction: our suggestion is never a prescription."""
    ud = _ud(
        consent_store=True,
        chief_complaint="headache",
        recommendations=[{"id": "paracetamol_tab_500", "generic_name": "Paracetamol"}],
    )
    ud.patient.current_medications = ["Cetirizine"]
    await repo.start_call(ud)
    ud.consent_store = True
    await repo.finish_call(ud)

    by_source = {row.source: row for row in await _rows(MedRow)}
    assert by_source["ai_recommended"].generic_name == "Paracetamol"
    assert by_source["patient_reported"].generic_name == "Cetirizine"
    # Nothing on this path may ever claim to be a doctor's prescription.
    assert "doctor_prescribed" not in by_source


async def test_symptom_row_records_only_what_was_asked(db):
    """Unasked fields stay NULL rather than being invented."""
    ud = _ud(consent_store=True, chief_complaint="headache since yesterday")
    ud.record_answer("duration", "since yesterday")
    ud.record_answer("severity", "moderate")
    await repo.start_call(ud)
    ud.consent_store = True
    await repo.finish_call(ud)

    rows = await _rows(Symptom)
    assert len(rows) == 1
    assert rows[0].symptom == "headache since yesterday"
    assert rows[0].duration == "since yesterday"
    assert rows[0].severity == "moderate"
    assert rows[0].onset is None  # never asked, so never guessed


async def test_no_symptom_row_without_a_complaint(db):
    ud = _ud(consent_store=True)
    await repo.start_call(ud)
    ud.consent_store = True
    await repo.finish_call(ud)
    assert await _rows(Symptom) == []


async def test_medical_history_is_not_duplicated_across_calls(db):
    first = _ud(consent_store=True, chief_complaint="headache")
    first.medical_history = [
        {"kind": "condition", "detail": "migraine"},
        {"kind": "allergy", "detail": "penicillin"},
    ]
    await repo.start_call(first)
    first.consent_store = True
    await repo.finish_call(first)

    second = _ud(consent_store=True, chief_complaint="headache again")
    second.medical_history = [{"kind": "condition", "detail": "migraine"}]
    await repo.start_call(second)
    second.consent_store = True
    await repo.finish_call(second)

    rows = await _rows(MedicalHistory)
    assert {(r.kind, r.detail) for r in rows} == {
        ("condition", "migraine"),
        ("allergy", "penicillin"),
    }
    assert len(rows) == 2  # migraine was offered twice, stored once


async def test_demographics_stay_null_unless_the_caller_gave_them(db):
    ud = _ud(consent_store=True, chief_complaint="headache")
    await repo.start_call(ud)
    ud.consent_store = True
    await repo.finish_call(ud)

    users = await _rows(User)
    assert users[0].name is None
    assert users[0].gender is None
    assert users[0].age_years is None


async def test_demographics_are_stored_when_volunteered(db):
    ud = _ud(consent_store=True, chief_complaint="headache")
    ud.patient_name = "Rahul"
    ud.patient_gender = "male"
    ud.patient.age_years = 29
    await repo.start_call(ud)
    ud.consent_store = True
    await repo.finish_call(ud)

    users = await _rows(User)
    assert users[0].name == "Rahul"
    assert users[0].gender == "male"
    assert users[0].age_years == 29


async def test_writes_are_skipped_when_the_call_row_is_missing(db):
    """If start_call never landed, dependent writes must not throw FK errors.

    Otherwise every turn of a live call fills the log with integrity-error
    tracebacks, which is what happens when the calls row disappears mid-session.
    """
    ud = _ud(consent_store=True, chief_complaint="headache")
    # Deliberately skip start_call, so no calls row exists.
    assert ud.call_row_ready is False

    await repo.record_turn(ud, "user", "I have a headache")
    await repo.finish_call(ud)

    assert await _rows(Message) == []
    assert await _rows(Call) == []


async def test_preferred_language_is_recorded_but_does_not_steer_the_next_call(db):
    """One Hindi turn used to make every later call open in Hindi.

    The agent now always greets in English and only changes language when the
    caller asks, so the stored preference is a record, not an instruction.
    """
    first = _ud(language="ta-IN")
    await repo.start_call(first)
    await repo.finish_call(first)

    second = _ud()
    await repo.start_call(second)
    assert second.language == DEFAULT_LANGUAGE_CODE

    users = await _rows(User)
    assert users[0].preferred_language == "ta-IN"


# ---------------------------------------------------------------- privacy ---


async def test_delete_caller_data_removes_everything(db):
    ud = _ud(chief_complaint="fever", consent_store=True)
    await repo.start_call(ud)
    ud.consent_store = True
    await repo.record_turn(ud, "user", "I have a fever")
    await repo.finish_call(ud)

    removed = await repo.delete_caller_data(PHONE)
    assert removed == 1
    assert await _rows(User) == []
    assert await _rows(Call) == []
    assert await _rows(Message) == []


async def test_delete_unknown_caller_is_a_no_op(db):
    assert await repo.delete_caller_data("+91 90000 00000") == 0


async def test_purge_old_messages_respects_the_window(db):
    ud = _ud(consent_store=True)
    await repo.start_call(ud)
    ud.consent_store = True
    await repo.record_turn(ud, "user", "recent message")

    assert await repo.purge_old_messages(days=30) == 0
    assert len(await _rows(Message)) == 1
    # A zero-day window makes everything stale.
    assert await repo.purge_old_messages(days=0) == 1
    assert await _rows(Message) == []


async def test_actions_are_audited(db):
    ud = _ud()
    await repo.start_call(ud)
    await repo.finish_call(ud)
    actions = {row.action for row in await _rows(AuditLog)}
    assert "call_started" in actions
    assert "call_finished" in actions


# ------------------------------------------------------ graceful degradation ---


async def test_all_operations_are_no_ops_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_db", False)
    ud = _ud()
    # None of these may raise, and none may block the call.
    await repo.start_call(ud)
    await repo.record_turn(ud, "user", "hello")
    await repo.record_consent(ud, "store", True)
    await repo.finish_call(ud)
    assert await repo.delete_caller_data(PHONE) == 0
    assert await repo.purge_old_messages() == 0


async def test_database_failure_does_not_break_the_call(monkeypatch):
    """An unreachable database must be swallowed, not raised at the caller."""
    monkeypatch.setattr(settings, "enable_db", True)
    monkeypatch.setattr(
        settings, "database_url", "postgresql+asyncpg://nobody@127.0.0.1:1/nope"
    )
    monkeypatch.setattr(settings, "phone_hash_key", "unit-test-hash-key")
    monkeypatch.setattr(settings, "db_connect_timeout", 0.5)
    db_session.reset_for_tests()

    ud = _ud()
    await repo.start_call(ud)  # must not raise
    await repo.finish_call(ud)
    assert ud.user_id is None  # nothing was loaded, call continues regardless

    db_session.reset_for_tests()


# ------------------------------------------------------ identity linking ---
# Regression locks from an audit: a withheld SIP caller ID ("anonymous",
# "unknown", "+") normalised to "" and hashed to one shared value, so every
# such caller collapsed into a single patient record and the next one was
# greeted as a returning caller and read back the previous caller's summary.


@pytest.mark.parametrize(
    "withheld",
    ["anonymous", "Anonymous", "unknown", "restricted", "private", "+", "abc", "", "  "],
)
def test_withheld_caller_ids_are_not_identities(withheld):
    from db.crypto import is_usable_phone

    assert not is_usable_phone(withheld)


@pytest.mark.parametrize("withheld", ["anonymous", "unknown", "+", ""])
def test_hash_phone_refuses_unusable_ids(monkeypatch, withheld):
    """Must raise rather than return a hash two callers could share."""
    from config import settings as cfg
    from db.crypto import hash_phone

    monkeypatch.setattr(cfg, "phone_hash_key", "k" * 32)
    with pytest.raises(ValueError, match="unusable caller ID"):
        hash_phone(withheld)


def test_real_numbers_still_hash_to_one_identity(monkeypatch):
    """The formats a real Indian SIP caller arrives in must still converge."""
    from config import settings as cfg
    from db.crypto import hash_phone, is_usable_phone

    monkeypatch.setattr(cfg, "phone_hash_key", "k" * 32)
    variants = [
        "+919876543210",
        "919876543210",
        "09876543210",
        "9876543210",
        "+91 98765 43210",
        "+91-98765-43210",
    ]
    assert all(is_usable_phone(v) for v in variants)
    assert len({hash_phone(v) for v in variants}) == 1


async def test_anonymous_caller_is_not_linked_to_a_user(db):
    """No user row, so no returning-caller recall across two withheld callers."""
    first = MedLinkUserData(caller_phone="anonymous", channel="pstn")
    await repo.start_call(first)
    second = MedLinkUserData(caller_phone="anonymous", channel="pstn")
    await repo.start_call(second)

    assert first.user_id is None
    assert second.user_id is None
    assert second.is_returning_caller is False
    assert second.previous_summary is None
