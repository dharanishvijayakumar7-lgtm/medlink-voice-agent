"""Relational schema for MedLink call and patient history.

Normalised rather than one wide table, so symptoms, assessment, medicines,
escalations and consents can each be queried and retained independently.

Column types are deliberately portable (``Uuid``, ``JSON``, ``LargeBinary`` -
no ARRAY/JSONB/vector), so the same models run on PostgreSQL in production and
SQLite in the test suite. Retrieval is lexical, so no vector column is needed.

Privacy: no raw phone number is stored anywhere. ``phone_hash`` is an HMAC used
for returning-caller lookup; ``phone_enc`` is Fernet ciphertext, used only when
a number genuinely has to be read back.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    """A caller, identified only by a hash of their phone number."""

    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    phone_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    phone_enc: Mapped[bytes | None] = mapped_column(LargeBinary, default=None)
    preferred_language: Mapped[str | None] = mapped_column(String(16), default=None)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    # Consent gates what we keep beyond the bare operational record.
    consent_store: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_share_doctor: Mapped[bool] = mapped_column(Boolean, default=False)

    calls: Mapped[list[Call]] = relationship(back_populates="user")


class Call(Base):
    """One conversation. Always written; clinical detail depends on consent."""

    __tablename__ = "calls"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id"), default=None, index=True
    )
    session_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    channel: Mapped[str] = mapped_column(String(16), default="web")  # pstn|web|console
    direction: Mapped[str] = mapped_column(String(16), default="inbound")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    language: Mapped[str | None] = mapped_column(String(16), default=None)

    chief_complaint: Mapped[str | None] = mapped_column(Text, default=None)
    triage_entry_id: Mapped[str | None] = mapped_column(String(64), default=None)
    severity_score: Mapped[int] = mapped_column(Integer, default=0)
    urgency: Mapped[str] = mapped_column(String(16), default="unknown")
    disposition: Mapped[str | None] = mapped_column(String(32), default=None)
    is_emergency: Mapped[bool] = mapped_column(Boolean, default=False)
    escalated: Mapped[bool] = mapped_column(Boolean, default=False)
    questions_asked: Mapped[int] = mapped_column(Integer, default=0)
    summary_en: Mapped[str | None] = mapped_column(Text, default=None)

    user: Mapped[User | None] = relationship(back_populates="calls")
    messages: Mapped[list[Message]] = relationship(
        back_populates="call", cascade="all, delete-orphan"
    )
    answers: Mapped[list[CallAnswer]] = relationship(
        back_populates="call", cascade="all, delete-orphan"
    )
    assessments: Mapped[list[TriageAssessment]] = relationship(
        back_populates="call", cascade="all, delete-orphan"
    )
    recommendations: Mapped[list[MedicineRecommendationRow]] = relationship(
        back_populates="call", cascade="all, delete-orphan"
    )
    escalations: Mapped[list[Escalation]] = relationship(
        back_populates="call", cascade="all, delete-orphan"
    )


class Message(Base):
    """A single conversational turn. Only stored with consent_store."""

    __tablename__ = "messages"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    call_id: Mapped[UUID] = mapped_column(ForeignKey("calls.id"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    language: Mapped[str | None] = mapped_column(String(16), default=None)
    text_original: Mapped[str] = mapped_column(Text)
    # Normalised English so doctor handoff and later retrieval work across
    # languages and code-mixed turns.
    text_en_gloss: Mapped[str | None] = mapped_column(Text, default=None)

    call: Mapped[Call] = relationship(back_populates="messages")


class CallAnswer(Base):
    """A filled triage slot (duration, severity, associated symptoms, ...)."""

    __tablename__ = "call_answers"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    call_id: Mapped[UUID] = mapped_column(ForeignKey("calls.id"), index=True)
    slot: Mapped[str] = mapped_column(String(32))
    answer: Mapped[str] = mapped_column(Text)

    call: Mapped[Call] = relationship(back_populates="answers")


class TriageAssessment(Base):
    """The deterministic triage outcome - never model-authored."""

    __tablename__ = "triage_assessments"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    call_id: Mapped[UUID] = mapped_column(ForeignKey("calls.id"), index=True)
    triage_entry_id: Mapped[str | None] = mapped_column(String(64), default=None)
    severity_score: Mapped[int] = mapped_column(Integer, default=0)
    urgency: Mapped[str] = mapped_column(String(16), default="unknown")
    red_flag_category: Mapped[str | None] = mapped_column(String(64), default=None)
    red_flag_term: Mapped[str | None] = mapped_column(String(128), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    call: Mapped[Call] = relationship(back_populates="assessments")


class MedicineRecommendationRow(Base):
    """What was actually suggested, plus the structured record behind it."""

    __tablename__ = "medicine_recommendations"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    call_id: Mapped[UUID] = mapped_column(ForeignKey("calls.id"), index=True)
    formulary_id: Mapped[str] = mapped_column(String(64))
    generic_name: Mapped[str] = mapped_column(String(128))
    dose_text: Mapped[str | None] = mapped_column(Text, default=None)
    # Audit trail: the structured fields the spoken advice was built from.
    details: Mapped[dict | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    call: Mapped[Call] = relationship(back_populates="recommendations")


class Provider(Base):
    """A doctor, clinic or health centre we can direct or transfer a caller to."""

    __tablename__ = "providers"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(32), default="phc")  # phc|clinic|hospital
    languages: Mapped[list | None] = mapped_column(JSON, default=None)
    phone_enc: Mapped[bytes | None] = mapped_column(LargeBinary, default=None)
    area: Mapped[str | None] = mapped_column(String(128), default=None)
    on_call: Mapped[bool] = mapped_column(Boolean, default=False)


class Escalation(Base):
    """A consented handoff toward real medical care."""

    __tablename__ = "escalations"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    call_id: Mapped[UUID] = mapped_column(ForeignKey("calls.id"), index=True)
    reason: Mapped[str | None] = mapped_column(Text, default=None)
    action: Mapped[str] = mapped_column(String(32), default="advice")
    provider_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("providers.id"), default=None
    )
    consent_ts: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    outcome: Mapped[str | None] = mapped_column(String(64), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    call: Mapped[Call] = relationship(back_populates="escalations")


class Consent(Base):
    """Every consent decision, logged separately from the flags on User."""

    __tablename__ = "consents"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    user_id: Mapped[UUID | None] = mapped_column(ForeignKey("users.id"), default=None)
    call_id: Mapped[UUID | None] = mapped_column(ForeignKey("calls.id"), default=None)
    kind: Mapped[str] = mapped_column(String(32))  # store | share_doctor | callback
    granted: Mapped[bool] = mapped_column(Boolean)
    method: Mapped[str] = mapped_column(String(16), default="voice")
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AuditLog(Base):
    """Append-only record of anything that touched or shared personal data."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    actor: Mapped[str] = mapped_column(String(64), default="agent")
    call_id: Mapped[UUID | None] = mapped_column(Uuid, default=None)
    action: Mapped[str] = mapped_column(String(64))
    detail: Mapped[dict | None] = mapped_column(JSON, default=None)


Index("ix_calls_started_at", Call.started_at)
Index("ix_audit_ts", AuditLog.ts)
