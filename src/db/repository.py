"""Call and patient history persistence.

Two rules govern everything here:

1. **A database problem must never break a call.** Every public coroutine is
   wrapped so failures are logged and swallowed. If persistence is disabled or
   unreachable the agent carries on and the caller notices nothing.

2. **Consent gates clinical content.** The operational ``calls`` row (timings,
   triage outcome, whether we escalated) is always written - it carries no
   identifying content and we need it to run and audit the service. The
   caller's words, their filled triage slots, their chief complaint and their
   decrypted number are stored only once ``consent_store`` is true.

Writes from the voice loop go through :func:`fire_and_forget` so the caller is
never waiting on the database.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import delete, desc, select

from config import settings
from db import session as db_session
from db.crypto import encrypt, hash_phone
from db.models import (
    SOURCE_AI_RECOMMENDED,
    SOURCE_PATIENT_REPORTED,
    AuditLog,
    Call,
    CallAnswer,
    Consent,
    Escalation,
    MedicalHistory,
    Medication,
    Message,
    Symptom,
    TriageAssessment,
    User,
)
from session_state import MedLinkUserData

logger = logging.getLogger("medlink.db")

_BACKGROUND_TASKS: set[asyncio.Task] = set()


def fire_and_forget(coro: Awaitable[Any]) -> None:
    """Run a write in the background without blocking the voice loop."""
    task = asyncio.ensure_future(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


def may_store_content(ud: MedLinkUserData) -> bool:
    """May we persist this call's clinical content (words, complaint, summary)?

    True when the caller consented, or when consent is deliberately relaxed for
    local development (``MEDLINK_REQUIRE_CONSENT=false``). Operational rows -
    timings, urgency, triage outcome - are always written and never gated.
    """
    return ud.consent_store or not settings.require_consent


async def _safe(factory: Callable[[], Awaitable[Any]], what: str) -> Any:
    """Run a persistence op, or skip it entirely when the DB is disabled.

    Takes a factory rather than a coroutine so nothing is constructed - and
    therefore nothing is left un-awaited - when persistence is off.
    """
    if not db_session.is_enabled():
        return None
    try:
        return await factory()
    except Exception:
        # Persistence is best-effort; a failure here must not reach the caller.
        logger.exception("persistence failed: %s", what)
        return None


# --------------------------------------------------------------- call start ---


async def start_call(ud: MedLinkUserData) -> None:
    """Create the call row and, for a known number, load prior context."""
    await _safe(lambda: _start_call(ud), "start_call")


async def _start_call(ud: MedLinkUserData) -> None:
    async with db_session.session_scope() as session:
        user: User | None = None

        if ud.caller_phone:
            phone_hash = hash_phone(ud.caller_phone)
            user = (
                await session.execute(select(User).where(User.phone_hash == phone_hash))
            ).scalar_one_or_none()

            if user is None:
                user = User(phone_hash=phone_hash)
                session.add(user)
                await session.flush()
            else:
                ud.is_returning_caller = True
                user.last_seen_at = datetime.now(timezone.utc)
                if user.preferred_language:
                    ud.language = user.preferred_language
                # Only recall what they previously agreed we could keep.
                if user.consent_store or not settings.require_consent:
                    ud.previous_summary = await _previous_summary(session, user.id)

            ud.user_id = str(user.id)
            ud.consent_store = user.consent_store

        session.add(
            Call(
                id=UUID(ud.call_id),
                user_id=user.id if user else None,
                session_id=ud.call_id,
                channel=ud.channel,
                started_at=ud.started_at,
                language=ud.language,
            )
        )
        session.add(
            AuditLog(
                call_id=UUID(ud.call_id),
                action="call_started",
                detail={"channel": ud.channel, "returning": ud.is_returning_caller},
            )
        )
        ud.call_row_ready = True


async def _previous_summary(session, user_id: UUID) -> str | None:
    """One-line recap of this caller's last completed call."""
    previous = (
        await session.execute(
            select(Call)
            .where(Call.user_id == user_id, Call.summary_en.is_not(None))
            .order_by(desc(Call.started_at))
            .limit(1)
        )
    ).scalar_one_or_none()
    if previous is None:
        return None
    when = previous.started_at.strftime("%d %B")
    return f"On {when} they called about: {previous.summary_en}"


# -------------------------------------------------------------- during call ---


async def record_turn(
    ud: MedLinkUserData, role: str, text: str, language: str | None = None
) -> None:
    """Store one conversational turn. Gated by :func:`may_store_content`."""
    if not text or not ud.call_row_ready or not may_store_content(ud):
        return
    await _safe(lambda: _record_turn(ud, role, text, language), "record_turn")


async def _record_turn(
    ud: MedLinkUserData, role: str, text: str, language: str | None
) -> None:
    async with db_session.session_scope() as session:
        session.add(
            Message(
                call_id=UUID(ud.call_id),
                role=role,
                text_original=text,
                language=language or ud.language,
            )
        )


async def record_consent(
    ud: MedLinkUserData, kind: str, granted: bool, method: str = "voice"
) -> None:
    """Log a consent decision and mirror it onto the user record."""
    await _safe(lambda: _record_consent(ud, kind, granted, method), "record_consent")


async def _record_consent(
    ud: MedLinkUserData, kind: str, granted: bool, method: str
) -> None:
    async with db_session.session_scope() as session:
        session.add(
            Consent(
                user_id=UUID(ud.user_id) if ud.user_id else None,
                call_id=UUID(ud.call_id),
                kind=kind,
                granted=granted,
                method=method,
            )
        )
        session.add(
            AuditLog(
                call_id=UUID(ud.call_id),
                action=f"consent_{kind}",
                detail={"granted": granted, "method": method},
            )
        )
        if ud.user_id:
            user = await session.get(User, UUID(ud.user_id))
            if user is not None:
                if kind == "store":
                    user.consent_store = granted
                    if granted and ud.caller_phone:
                        user.phone_enc = encrypt(ud.caller_phone)
                elif kind == "share_doctor":
                    user.consent_share_doctor = granted


# ----------------------------------------------------------------- call end ---


async def finish_call(ud: MedLinkUserData) -> None:
    """Write the outcome of the call. Safe to call exactly once, at shutdown."""
    if not ud.call_row_ready:
        logger.warning("no call row for %s - skipping finish_call", ud.call_id)
        return
    await _safe(lambda: _finish_call(ud), "finish_call")


async def _finish_call(ud: MedLinkUserData) -> None:
    async with db_session.session_scope() as session:
        call = await session.get(Call, UUID(ud.call_id))
        if call is None:
            return

        # Always-safe operational fields.
        call.ended_at = datetime.now(timezone.utc)
        call.language = ud.language
        call.severity_score = ud.severity_score
        call.urgency = ud.urgency
        call.disposition = ud.disposition
        call.escalated = ud.escalated
        call.questions_asked = ud.questions_asked
        call.triage_entry_id = ud.triage_entry_id
        call.is_emergency = bool(ud.red_flag and ud.red_flag.is_emergency)

        # Clinical detail only with consent.
        user_uuid = UUID(ud.user_id) if ud.user_id else None
        if may_store_content(ud):
            call.chief_complaint = ud.chief_complaint
            call.summary_en = ud.clinical_summary()
            for slot, answer in ud.answers.items():
                session.add(CallAnswer(call_id=call.id, slot=slot, answer=answer))

            symptom = ud.structured_symptom()
            if symptom is not None:
                session.add(Symptom(call_id=call.id, user_id=user_uuid, **symptom))

            # Background belongs to the patient, not the call, so it is only
            # written for an identified caller. Skip anything already on file.
            if user_uuid is not None and ud.medical_history:
                known = set(
                    (
                        await session.execute(
                            select(MedicalHistory.kind, MedicalHistory.detail).where(
                                MedicalHistory.user_id == user_uuid
                            )
                        )
                    ).all()
                )
                for item in ud.medical_history:
                    if (item["kind"], item["detail"]) in known:
                        continue
                    session.add(
                        MedicalHistory(
                            user_id=user_uuid,
                            call_id=call.id,
                            kind=item["kind"],
                            detail=item["detail"],
                        )
                    )

            # Medicines the caller says they are ALREADY on - never our advice.
            for name in ud.patient.current_medications:
                session.add(
                    Medication(
                        call_id=call.id,
                        user_id=user_uuid,
                        source=SOURCE_PATIENT_REPORTED,
                        generic_name=name[:128],
                    )
                )

            # Demographics, only where the caller actually gave them.
            if user_uuid is not None:
                user = await session.get(User, user_uuid)
                if user is not None:
                    if ud.patient_name:
                        user.name = ud.patient_name
                    if ud.patient_gender:
                        user.gender = ud.patient_gender
                    if ud.patient.age_years is not None:
                        user.age_years = ud.patient.age_years

        session.add(
            TriageAssessment(
                call_id=call.id,
                triage_entry_id=ud.triage_entry_id,
                severity_score=ud.severity_score,
                urgency=ud.urgency,
                red_flag_category=ud.red_flag.category_id if ud.red_flag else None,
                red_flag_term=ud.red_flag.matched_term if ud.red_flag else None,
            )
        )

        # Anything the agent suggested is ALWAYS ai_recommended. Nothing on this
        # path may ever be written as a doctor's prescription.
        for rec in ud.recommendations:
            session.add(
                Medication(
                    call_id=call.id,
                    user_id=UUID(ud.user_id) if ud.user_id else None,
                    source=SOURCE_AI_RECOMMENDED,
                    formulary_id=rec.get("id"),
                    generic_name=rec.get("generic_name", "unknown"),
                    dose_text=rec.get("adult_dose"),
                    indication=ud.chief_complaint,
                    details=rec,
                )
            )

        if ud.escalated:
            session.add(
                Escalation(
                    call_id=call.id,
                    reason=ud.red_flag.category_id if ud.red_flag else ud.urgency,
                    action="advice",
                    consent_ts=(
                        datetime.now(timezone.utc) if ud.consent_share_doctor else None
                    ),
                )
            )

        if ud.user_id and ud.language:
            user = await session.get(User, UUID(ud.user_id))
            if user is not None:
                user.preferred_language = ud.language

        session.add(
            AuditLog(
                call_id=call.id,
                action="call_finished",
                detail={
                    "urgency": ud.urgency,
                    "escalated": ud.escalated,
                    "stored_content": may_store_content(ud),
                    "consent_store": ud.consent_store,
                },
            )
        )


# ------------------------------------------------------- privacy operations ---


async def delete_caller_data(phone: str) -> int:
    """Erase everything for one caller. Returns the number of calls removed."""
    result = await _safe(lambda: _delete_caller_data(phone), "delete_caller_data")
    return result or 0


async def _delete_caller_data(phone: str) -> int:
    async with db_session.session_scope() as session:
        user = (
            await session.execute(
                select(User).where(User.phone_hash == hash_phone(phone))
            )
        ).scalar_one_or_none()
        if user is None:
            return 0

        calls = list(
            (await session.execute(select(Call).where(Call.user_id == user.id)))
            .scalars()
            .all()
        )
        for call in calls:
            await session.delete(call)  # cascades to messages/answers/etc.
        await session.execute(delete(Consent).where(Consent.user_id == user.id))
        await session.delete(user)

        session.add(
            AuditLog(action="caller_data_deleted", detail={"calls": len(calls)})
        )
        return len(calls)


async def purge_old_messages(days: int | None = None) -> int:
    """Retention job: drop transcripts older than the retention window.

    De-identified assessments are kept for quality metrics.
    """
    result = await _safe(lambda: _purge_old_messages(days), "purge_old_messages")
    return result or 0


async def _purge_old_messages(days: int | None) -> int:
    window = days if days is not None else settings.history_retention_days
    cutoff = datetime.now(timezone.utc) - timedelta(days=window)
    async with db_session.session_scope() as session:
        stale = list(
            (await session.execute(select(Message).where(Message.ts < cutoff)))
            .scalars()
            .all()
        )
        for message in stale:
            await session.delete(message)
        session.add(
            AuditLog(
                action="messages_purged",
                detail={"count": len(stale), "older_than_days": window},
            )
        )
        return len(stale)
