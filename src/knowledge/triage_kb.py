"""Retrieval over the curated triage knowledge base.

Deliberately **lexical**, not embedding-based:

  * The KB is a few dozen curated entries, each carrying the actual words
    callers use in six languages. Enumerated aliases beat a small multilingual
    embedding model on short symptom phrases, and cost nothing.
  * No model download, no GPU, no per-query API call - which matters because
    this project runs on free infrastructure only.

Matching order: longest alias hit wins (most specific), then BM25 over the
entry text as a fallback, then nothing. Returning `None` is a valid, safe
outcome - the caller then falls back to the core follow-up slots.

Note: aliases currently cover English plus romanized Indic. Native-script
aliases can be added freely; the primary path normalizes the complaint to
English before lookup (see `IntakeAgent.record_complaint`).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from config import settings
from safety.redflags import CLAUSE_SENTINEL, _is_negated, _normalize_clauses, normalize

_TOKEN_RE = re.compile(r"[^\w]+", re.UNICODE)
# BM25 fallback needs a clear signal before it overrides "no match".
_BM25_MIN_SCORE = 3.0


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.split(normalize(text)) if t]


class TriageEntry(BaseModel):
    id: str
    presentation: str
    aliases: dict[str, list[str]] = Field(default_factory=dict)
    red_flags: list[str] = Field(default_factory=list)
    candidate_questions: list[str] = Field(default_factory=list)
    severity_modifiers: dict[str, int] = Field(default_factory=dict)
    # [] means no OTC medicine is ever appropriate for this presentation.
    otc_categories_allowed: list[str] = Field(default_factory=list)
    self_care: str = ""
    refer_when: str = ""
    source: str = ""

    @property
    def allows_otc(self) -> bool:
        return bool(self.otc_categories_allowed)

    def all_aliases(self) -> list[str]:
        return [a for terms in self.aliases.values() for a in terms]

    def search_text(self) -> str:
        return " ".join([self.presentation, *self.all_aliases()])


class TriageKB:
    def __init__(self, entries: list[TriageEntry]):
        self.entries = entries
        self.by_id: dict[str, TriageEntry] = {e.id: e for e in entries}
        # (normalized alias, entry) sorted longest-first so "chest pain" beats "pain".
        self._aliases: list[tuple[str, TriageEntry]] = sorted(
            (
                (normalize(alias), entry)
                for entry in entries
                for alias in entry.all_aliases()
                if normalize(alias)
            ),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )
        docs = [_tokenize(e.search_text()) for e in entries]
        self._bm25 = BM25Okapi(docs) if docs else None

    @classmethod
    def from_yaml(cls, path: str | Path) -> TriageKB:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        entries = [TriageEntry.model_validate(e) for e in raw.get("entries", [])]
        ids = [e.id for e in entries]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate triage entry ids: {sorted(dupes)}")
        return cls(entries)

    def match(self, complaint: str) -> TriageEntry | None:
        """Best entry for a free-text complaint, or None if nothing is confident."""
        if not complaint or not complaint.strip():
            return None
        text = normalize(complaint)

        # 1. Alias match. Callers lead with their main concern, so the alias
        #    appearing EARLIEST wins ("fever and body pain" -> fever, not
        #    body_ache); ties break toward the longer, more specific alias.
        best: tuple[int, int, TriageEntry] | None = None
        for alias, entry in self._aliases:
            position = text.find(alias)
            if position == -1:
                continue
            candidate = (position, -len(alias), entry)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is not None:
            return best[2]

        # 2. BM25 fallback for phrasings the aliases missed.
        if self._bm25 is not None:
            tokens = _tokenize(complaint)
            if tokens:
                scores = self._bm25.get_scores(tokens)
                best = max(range(len(scores)), key=lambda i: scores[i])
                if scores[best] >= _BM25_MIN_SCORE:
                    return self.entries[best]
        return None

    def allowed_otc_classes(self, entry: TriageEntry | None) -> set[str] | None:
        """Classes the medicine filter may consider.

        `None` -> unrestricted search (no KB match, fall back to generic filters).
        `set()` -> KB says explicitly that no OTC medicine is appropriate.
        """
        if entry is None:
            return None
        return set(entry.otc_categories_allowed)


def score_modifiers(entry: TriageEntry, ud) -> int:
    """Extra severity points from this presentation's modifiers.

    Structural keys (`age_under_5`, `duration_over_5_days`, `pregnant`) are
    evaluated against the session state; everything else is matched as keywords
    against what the caller has told us so far.
    """
    if entry is None:
        return 0

    # Each answer is its own clause, so "no" in one answer can't negate the next.
    parts = [*ud.answers.values(), ud.chief_complaint or ""]
    clauses = _normalize_clauses(" . ".join(p for p in parts if p))
    flat = clauses.replace(CLAUSE_SENTINEL, " ")

    total = 0
    for key, points in entry.severity_modifiers.items():
        if _modifier_matches(key, ud, flat, clauses):
            total += points
    return total


def _affirmed(words: list[str], flat: str, clauses: str) -> bool:
    """True if the words occur, as a phrase, at least once without a negator.

    Reuses the red-flag module's negation scope (clause breaks, punctuation,
    3-token lookback), which was tested against real negated phrasings.
    """
    pattern = r"\b" + r"\s+".join(re.escape(w) for w in words) + r"\b"
    return any(not _is_negated(clauses, m.start()) for m in re.finditer(pattern, flat))


def _modifier_matches(key: str, ud, flat: str, clauses: str) -> bool:
    if match := re.fullmatch(r"age_under_(\d+)", key):
        age = ud.patient.age_years
        return age is not None and age < int(match.group(1))
    if match := re.fullmatch(r"age_over_(\d+)", key):
        age = ud.patient.age_years
        return age is not None and age > int(match.group(1))
    if match := re.fullmatch(r"duration_over_(\d+)_days", key):
        days = ud.patient.symptom_duration_days
        return days is not None and days > int(match.group(1))
    if key == "pregnant":
        return ud.patient.is_pregnant
    if key.startswith("age_over_") and key.endswith("_new_change"):
        age = ud.patient.age_years
        return age is not None and age > 45

    # Keyword modifiers. Match the whole phrase first ("blood in stool"), then
    # fall back to requiring every word. Short qualifiers such as "no" are kept:
    # dropping them would turn "no_urine" into a match on any mention of urine.
    #
    # Negation-aware: this used to be a plain substring test, so "No fever, no
    # neck stiffness" scored with_neck_stiffness (+8) and turned a mild tension
    # headache into an emergency. A phrase that is itself negative ("no urine")
    # still matches, because the negator is part of the phrase, not before it.
    words = [w for w in key.removeprefix("with_").split("_") if w]
    if not words:
        return False
    if _affirmed(words, flat, clauses):
        return True
    return all(_affirmed([w], flat, clauses) for w in words)


def apply_to_session(ud, complaint: str | None = None) -> TriageEntry | None:
    """Match the KB against the complaint and write its guidance onto the session.

    Sets the candidate follow-up questions, the OTC categories the medicine
    filter is allowed to consider, the self-care / referral text, and the
    presentation-specific severity bonus. Safe to call repeatedly - the bonus is
    recomputed, never accumulated.
    """
    kb = get_triage_kb()
    entry = kb.match(complaint or ud.chief_complaint or "")

    if entry is None:
        ud.kb_severity_bonus = 0
        return None

    ud.triage_entry_id = entry.id
    ud.candidate_questions = list(entry.candidate_questions)
    ud.allowed_otc_classes = kb.allowed_otc_classes(entry)
    ud.self_care_advice = entry.self_care
    ud.refer_when = entry.refer_when
    ud.kb_severity_bonus = score_modifiers(entry, ud)
    return entry


@lru_cache(maxsize=2)
def get_triage_kb(path: str | None = None) -> TriageKB:
    return TriageKB.from_yaml(path or settings.triage_kb_path)
