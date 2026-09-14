"""Export a structured summary of each finished call to Firestore.

The separate mobile app reads call history from Firestore, so when a call ends
the agent writes one document per call under the caller's phone number:

    patients/{+91XXXXXXXXXX}                  one per caller: who they are + rollups
    patients/{+91XXXXXXXXXX}/calls/{call_id}  one per call: the structured summary
    unidentified_calls/{call_id}              calls with a withheld/unusable caller ID

Postgres (`db/repository.py`) stays the agent's own history - returning-caller
recall still reads it. Firestore is purely the app-facing copy.

Everything here runs after the caller has hung up, from the session shutdown
callback, so it never touches call latency. A failure is logged and swallowed:
losing one app-side summary must not break worker shutdown.

The document builders are pure functions of `MedLinkUserData`, so the shape of
what the app receives is testable without Firebase.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import PROJECT_ROOT, SUPPORTED_LANGUAGES, settings
from db.crypto import is_usable_phone, normalise_phone
from db.repository import may_store_content
from session_state import MedLinkUserData

logger = logging.getLogger("medlink.firestore")

# Bump when the document shape changes in a way the app has to handle.
SCHEMA_VERSION = 1

_LANGUAGE_NAMES = {code: name.capitalize() for name, code in SUPPORTED_LANGUAGES.items()}

# Replaced with `firestore.Increment(1)` at commit time. Kept as a plain marker so
# the write plan below stays a pure function that tests can inspect.
INCREMENT_TOTAL_CALLS = object()


# ------------------------------------------------------------------ builders ---


def patient_id(phone: str | None) -> str | None:
    """E.164 id for the patient document, or None for a withheld caller ID.

    Indian numbers arrive as 9876543210, +919876543210, 09876543210 and so on;
    `normalise_phone` reduces all of them to the 10-digit subscriber number, so
    every format lands on the same `+91...` document.
    """
    if not is_usable_phone(phone):
        return None
    digits = normalise_phone(phone or "")
    return f"+91{digits}" if len(digits) == 10 else f"+{digits}"


def _compact(value: Any) -> Any:
    """Drop empty values so the app sees absent fields, not "", [] or {}."""
    if isinstance(value, dict):
        out = {k: _compact(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if v not in (None, "", [], {})}
    if isinstance(value, list):
        return [v for v in (_compact(i) for i in value) if v not in (None, "", [], {})]
    return value


def _known(value: str | None) -> str | None:
    """The session uses the literal "unknown" as a placeholder; the app gets null."""
    return None if value in (None, "", "unknown") else value


def build_summary_text(ud: MedLinkUserData) -> str:
    """A short, readable English summary built only from captured fields.

    Deterministic - no LLM - so it costs nothing and cannot invent details.
    Clauses with missing data are skipped rather than filled with placeholders.
    """
    patient = ud.patient
    age = patient.age_years

    if ud.patient_name:
        extras = [x for x in (ud.patient_gender, None if age is None else str(age)) if x]
        who = f"{ud.patient_name} ({', '.join(extras)})" if extras else ud.patient_name
    elif age is not None:
        who = f"{'Child' if patient.is_for_child or age < 12 else 'Adult'}, {age}"
    elif patient.is_for_child:
        who = "A caller asking about a child"
    else:
        who = "Caller"

    language = _LANGUAGE_NAMES.get(ud.language)
    sentences = []
    if ud.chief_complaint:
        lang = f" in {language}" if language else ""
        sentences.append(f"{who} called{lang} about: {ud.chief_complaint.strip()}.")
    else:
        sentences.append(f"{who} called; no complaint was recorded.")

    detail = [
        f"{slot.capitalize()}: {ud.answers[slot]}"
        for slot in ("duration", "severity", "location", "associated", "history")
        if ud.answers.get(slot)
    ]
    if detail:
        sentences.append(". ".join(detail) + ".")

    if patient.known_conditions:
        sentences.append(f"Known conditions: {', '.join(patient.known_conditions)}.")
    if patient.current_medications:
        sentences.append(
            f"Currently taking: {', '.join(patient.current_medications)}."
        )
    if category := assessment_category(ud):
        sentences.append(f"Category: {category}.")
    if ud.possible_causes:
        sentences.append(
            f"Possible causes (AI-suggested, not a diagnosis): "
            f"{', '.join(ud.possible_causes)}."
        )
    if urgency := _known(ud.urgency):
        sentences.append(f"Assessed urgency: {urgency.replace('_', ' ')}.")
    names = [r.get("generic_name") for r in ud.recommendations if r.get("generic_name")]
    if names:
        sentences.append(f"Medicines discussed: {', '.join(names)}.")
    if ud.escalated:
        sentences.append("Advised to seek medical care.")
    return " ".join(sentences)


POSSIBLE_CAUSES_NOTE = (
    "Suggested by AI from the caller's own description. Not a medical diagnosis "
    "and not confirmed by a clinician."
)


def assessment_category(ud: MedLinkUserData) -> str | None:
    """The curated triage-KB presentation this call matched, e.g. "Headache"."""
    if not ud.triage_entry_id:
        return None
    try:
        from knowledge.triage_kb import get_triage_kb

        entry = get_triage_kb().by_id.get(ud.triage_entry_id)
    except Exception:  # a KB load problem must not lose the rest of the summary
        logger.debug("triage KB unavailable for category lookup", exc_info=True)
        return None
    return entry.presentation if entry else None


def build_assessment(ud: MedLinkUserData) -> dict[str, Any]:
    """What it might be, from two sources kept deliberately separate.

    `category`, `self_care_advice` and `see_doctor_if` come from the curated
    triage knowledge base - reviewed content, nothing invented. `possible_causes`
    is the LLM's suggestion and always carries a not-a-diagnosis note, so the app
    can show the two differently.
    """
    return {
        "category": assessment_category(ud),
        "self_care_advice": " ".join(ud.self_care_advice.split()) or None,
        "see_doctor_if": " ".join(ud.refer_when.split()) or None,
        "possible_causes": list(ud.possible_causes),
        "possible_causes_reasoning": ud.possible_causes_reasoning,
        "possible_causes_note": POSSIBLE_CAUSES_NOTE if ud.possible_causes else None,
    }


def build_call_document(ud: MedLinkUserData, ended_at: datetime) -> dict[str, Any]:
    """One call, as the mobile app receives it."""
    patient = ud.patient
    duration = max(0, int((ended_at - ud.started_at).total_seconds()))
    doc = {
        "schema_version": SCHEMA_VERSION,
        "call_id": ud.call_id,
        "phone": patient_id(ud.caller_phone),
        "channel": ud.channel,
        "language": ud.language,
        "started_at": ud.started_at,
        "ended_at": ended_at,
        "duration_sec": duration,
        "returning_caller": ud.is_returning_caller,
        "patient": {
            "name": ud.patient_name,
            "age_years": patient.age_years,
            "gender": ud.patient_gender,
            "is_for_child": patient.is_for_child or None,
            "is_pregnant": patient.is_pregnant or None,
        },
        "chief_complaint": ud.chief_complaint,
        "symptom": ud.structured_symptom(),
        "answers": dict(ud.answers),
        "known_conditions": list(patient.known_conditions),
        "current_medications": list(patient.current_medications),
        "medical_history": list(ud.medical_history),
        "medicines_discussed": [
            {
                "generic_name": r.get("generic_name"),
                "brand_names": r.get("brand_names"),
                "adult_dose": r.get("adult_dose"),
                "paediatric_dose": r.get("paediatric_dose"),
                "max_daily_dose": r.get("max_daily_dose"),
                "duration_limit_days": r.get("duration_limit_days"),
            }
            for r in ud.recommendations
        ],
        "triage_entry_id": ud.triage_entry_id,
        "assessment": build_assessment(ud),
        "severity_score": ud.severity_score or None,
        "urgency": _known(ud.urgency),
        "disposition": ud.disposition,
        "escalated": ud.escalated,
        "summary_text": build_summary_text(ud),
    }
    # `escalated` and `returning_caller` are meaningful as False, so keep them.
    compacted = _compact(doc)
    compacted.setdefault("escalated", ud.escalated)
    compacted.setdefault("returning_caller", ud.is_returning_caller)
    return compacted


def build_patient_fields(ud: MedLinkUserData, call_doc: dict[str, Any]) -> dict[str, Any]:
    """Rollup fields merged onto `patients/{phone}` after every call.

    Only fields actually known are sent, so a later call with a blank name never
    erases a name captured on an earlier one (the write uses merge=True).
    """
    return _compact(
        {
            "phone": call_doc.get("phone"),
            "name": ud.patient_name,
            "age_years": ud.patient.age_years,
            "gender": ud.patient_gender,
            "preferred_language": ud.language,
            "last_seen_at": call_doc["ended_at"],
            "last_call_at": call_doc["started_at"],
            "last_call_id": ud.call_id,
            "last_summary_text": call_doc.get("summary_text"),
            "last_urgency": call_doc.get("urgency"),
            "last_category": (call_doc.get("assessment") or {}).get("category"),
        }
    )


# ---------------------------------------------------------------- write plan ---


@dataclass(frozen=True)
class Write:
    """One Firestore `set`. `path` alternates collection and document ids."""

    path: tuple[str, ...]
    data: dict[str, Any]
    merge: bool = False


def plan_writes(
    pid: str | None,
    call_doc: dict[str, Any],
    patient_fields: dict[str, Any],
    *,
    call_exists: bool,
    patient_exists: bool,
) -> list[Write]:
    """Decide what to write, given what is already in Firestore.

    Idempotent by construction: if this call's document already exists (the
    shutdown callback ran twice, or an export is retried) nothing is written,
    so `total_calls` is never double-counted.
    """
    if call_exists:
        return []
    call_id = call_doc["call_id"]
    if pid is None:
        return [Write(("unidentified_calls", call_id), call_doc)]

    patient = dict(patient_fields)
    patient["total_calls"] = INCREMENT_TOTAL_CALLS
    if not patient_exists:
        patient["first_seen_at"] = call_doc["started_at"]
    return [
        Write(("patients", pid, "calls", call_id), call_doc),
        Write(("patients", pid), patient, merge=True),
    ]


# -------------------------------------------------------------------- writer ---

_client_lock = threading.Lock()
_client: Any = None


def _credentials_path() -> Path:
    path = settings.firebase_credentials_path
    return path if path.is_absolute() else PROJECT_ROOT / path


def _get_client():
    """Initialise Firebase once, under its own app name so it can't clash."""
    global _client
    with _client_lock:
        if _client is None:
            import firebase_admin
            from firebase_admin import credentials, firestore

            try:
                app = firebase_admin.get_app("medlink")
            except ValueError:
                app = firebase_admin.initialize_app(
                    credentials.Certificate(str(_credentials_path())), name="medlink"
                )
            _client = firestore.client(app=app)
        return _client


def _ref(db, path: tuple[str, ...]):
    ref = db.collection(path[0]).document(path[1])
    for i in range(2, len(path), 2):
        ref = ref.collection(path[i]).document(path[i + 1])
    return ref


def _commit(pid: str | None, call_doc: dict[str, Any], patient_fields: dict[str, Any]) -> bool:
    """Run the write plan inside one Firestore transaction. Blocking."""
    from firebase_admin import firestore

    db = _get_client()
    call_path = (
        ("unidentified_calls", call_doc["call_id"])
        if pid is None
        else ("patients", pid, "calls", call_doc["call_id"])
    )
    call_ref = _ref(db, call_path)
    patient_ref = _ref(db, ("patients", pid)) if pid else None

    @firestore.transactional
    def run(transaction) -> bool:
        # Firestore transactions require every read before any write.
        call_exists = call_ref.get(transaction=transaction).exists
        patient_exists = bool(
            patient_ref and patient_ref.get(transaction=transaction).exists
        )
        writes = plan_writes(
            pid,
            call_doc,
            patient_fields,
            call_exists=call_exists,
            patient_exists=patient_exists,
        )
        for write in writes:
            data = {
                k: (firestore.Increment(1) if v is INCREMENT_TOTAL_CALLS else v)
                for k, v in write.data.items()
            }
            transaction.set(_ref(db, write.path), data, merge=write.merge)
        return bool(writes)

    return run(db.transaction())


async def export_call(ud: MedLinkUserData) -> bool:
    """Write this call's summary to Firestore. Returns True if anything was written.

    Never raises: the caller is a shutdown callback.
    """
    if not settings.enable_firestore_export:
        return False
    if not _credentials_path().is_file():
        logger.warning(
            "MEDLINK_ENABLE_FIRESTORE is on but no credentials file at %s - "
            "skipping export",
            _credentials_path(),
        )
        return False
    # Same consent gate as Postgres clinical content.
    if not may_store_content(ud):
        logger.info("no consent to store content - skipping Firestore export")
        return False

    try:
        ended_at = datetime.now(timezone.utc)
        call_doc = build_call_document(ud, ended_at)
        pid = call_doc.get("phone")
        written = await asyncio.wait_for(
            asyncio.to_thread(
                _commit, pid, call_doc, build_patient_fields(ud, call_doc)
            ),
            timeout=settings.firestore_timeout,
        )
    except Exception:
        logger.exception("Firestore export failed", extra={"call_id": ud.call_id})
        return False

    logger.info(
        "call summary exported to Firestore",
        extra={
            "call_id": ud.call_id,
            "patient": pid or "unidentified",
            "written": written,
        },
    )
    return written
