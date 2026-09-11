"""Deterministic, multilingual emergency red-flag detection.

Runs on every transcribed user turn *before* the LLM. If a red flag matches, the
session is force-routed to escalation no matter what the model would have said.

Design notes:
  * Bias toward over-triage. Missing an emergency is unacceptable; a false
    positive only costs the user a "please get checked" message.
  * No network, no LLM, microsecond latency. Safe to call on the hot path.
  * The lexicon lives in ``data/redflags.yaml`` so clinicians / native speakers
    can extend it without touching code.
  * ASCII terms match on word boundaries (so "no" doesn't hit inside "another").
    Non-ASCII (Indic-script) terms are substring-matched on NFKC-normalized text.
  * A deliberately narrow negation guard skips "no chest pain" style phrases.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from config import DATA_DIR

# Sentence-ending punctuation closes a clause, so it has to survive
# normalization as a marker instead of becoming plain whitespace. Without it,
# "no fever, chest pain" reads as a single clause and the negation swallows the
# emergency.
# \x01 specifically, not one of \x1c-\x1f: Python's str.split() treats those as
# whitespace, which silently ate the sentinel and let negation cross the comma.
CLAUSE_SENTINEL = "\x01"
_CLAUSE_PUNCT_RE = re.compile(r"[.,;:!?]+")
_PUNCT_RE = re.compile(r"['\"`()\[\]{}/\\|~*_<>@#%^&+=‘’“”-]+")  # noqa: RUF001
_WS_RE = re.compile(r"\s+")

# Last token before a match being one of these => treat the red flag as negated.
# Kept strictly to words that only ever negate. "na" (Hindi discourse filler,
# "mujhe na chest pain ho raha hai") and "ondu" (Kannada for "one") were removed:
# both are ordinary speech, and treating them as negators silently suppressed
# real emergencies.
_NEGATORS = {
    "no",
    "not",
    "dont",
    "doesnt",
    "didnt",
    "without",
    "never",
    "nil",
    "nahi",
    "nahin",
    "illa",
    "illai",
    "ledu",
}
# Words that close the clause a negator belongs to; a negator before one of
# these no longer negates what comes after ("I did not sleep AND now chest pain").
# The sentinel stands in for punctuation, which closes a clause just as hard.
_CLAUSE_BREAKS = {
    "and",
    "but",
    "so",
    "then",
    "because",
    "yet",
    "however",
    CLAUSE_SENTINEL,
}
# How many tokens before a match to scan for a negator. Deliberately short: a
# real negation sits close to what it negates ("no chest pain", "I have not had
# any chest pain" = 3). Scanning further back mostly catches a negator belonging
# to an earlier clause ("I have no doubt this is chest pain"), and a missed
# emergency costs far more than an extra "please get checked".
_NEG_LOOKBACK_TOKENS = 3


def _normalize_clauses(text: str) -> str:
    """Like `normalize`, but clause-ending punctuation becomes a sentinel token.

    Only `detect_redflag` uses this, to tell where one clause stops and the next
    begins. Replacing the sentinel with a space yields a string of *identical
    length*, so a match offset found in one is valid in the other - that is what
    lets terms be matched on the flat text while negation is judged on this one.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _CLAUSE_PUNCT_RE.sub(f" {CLAUSE_SENTINEL} ", text)
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Lowercase, NFKC-fold, strip punctuation, collapse whitespace.

    Indic combining marks are preserved (they carry meaning); only latin
    punctuation is removed.
    """
    flat = _normalize_clauses(text).replace(CLAUSE_SENTINEL, " ")
    return _WS_RE.sub(" ", flat).strip()


@dataclass(frozen=True)
class RedFlagHit:
    category_id: str
    priority: str  # "emergency" | "urgent"
    matched_term: str
    advice_template: str

    @property
    def is_emergency(self) -> bool:
        return self.priority == "emergency"


@dataclass(frozen=True)
class _CompiledCategory:
    id: str
    priority: str
    advice_template: str
    ascii_patterns: tuple[tuple[re.Pattern[str], str], ...]
    unicode_terms: tuple[str, ...]


def _compile_category(raw: dict) -> _CompiledCategory:
    ascii_patterns: list[tuple[re.Pattern[str], str]] = []
    unicode_terms: list[str] = []
    for term in raw.get("terms", []):
        norm = normalize(str(term))
        if not norm:
            continue
        if norm.isascii():
            # Escape, then allow flexible whitespace between words.
            pattern = r"\b" + r"\s+".join(re.escape(w) for w in norm.split()) + r"\b"
            ascii_patterns.append((re.compile(pattern), term))
        else:
            unicode_terms.append(norm)
    return _CompiledCategory(
        id=str(raw["id"]),
        priority=str(raw.get("priority", "emergency")),
        advice_template=" ".join(str(raw.get("advice", "")).split()),
        ascii_patterns=tuple(ascii_patterns),
        unicode_terms=tuple(unicode_terms),
    )


@lru_cache(maxsize=4)
def _load_categories(path: str) -> tuple[_CompiledCategory, ...]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return tuple(_compile_category(c) for c in data.get("categories", []))


def _is_negated(haystack: str, start: int) -> bool:
    """True if a negator token appears just before ``start`` in the same clause.

    Scans the last few tokens before the match. A negator counts only if no
    clause-break word ("and", "but", ...) sits between it and the match.
    """
    tokens = haystack[max(0, start - 80) : start].split()
    for tok in reversed(tokens[-_NEG_LOOKBACK_TOKENS:]):
        if tok in _CLAUSE_BREAKS:
            return False
        if tok in _NEGATORS:
            return True
    return False


def detect_redflag(
    text: str, *, redflags_path: str | Path | None = None
) -> RedFlagHit | None:
    """Return the highest-priority red flag in ``text``, or ``None``.

    Categories are evaluated in file order, so list the most time-critical ones
    (cardiac, breathing, stroke) first in ``redflags.yaml``.
    """
    if not text or not text.strip():
        return None
    path = str(redflags_path or (DATA_DIR / "redflags.yaml"))
    # Terms are matched against `flat` so a phrase still matches across a comma
    # ("chest, pain"); negation is judged against `clauses`, where that comma is
    # a hard stop. The two are the same length, so offsets carry between them.
    clauses = _normalize_clauses(text)
    flat = clauses.replace(CLAUSE_SENTINEL, " ")

    for category in _load_categories(path):
        for pattern, original in category.ascii_patterns:
            m = pattern.search(flat)
            if m and not _is_negated(clauses, m.start()):
                return RedFlagHit(
                    category.id, category.priority, original, category.advice_template
                )
        for term in category.unicode_terms:
            idx = flat.find(term)
            if idx != -1 and not _is_negated(clauses, idx):
                return RedFlagHit(
                    category.id, category.priority, term, category.advice_template
                )
    return None


def format_redflag_advice(
    hit: RedFlagHit, *, emergency_number: str = "112", ambulance_number: str = "108"
) -> str:
    return hit.advice_template.format(
        emergency=emergency_number, ambulance=ambulance_number
    )
