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
# A comma is a softer break than a full stop, and negation carries across it:
# "no vomiting blood, black stools, or chest pain" denies all three. Treating a
# comma as a hard clause end scored every item after the first as PRESENT, which
# turned an ordinary acidity call into "urgent" on the strength of the caller
# saying they had NOT had black stools.
LIST_SENTINEL = "\x02"
_CLAUSE_PUNCT_RE = re.compile(r"[.;:!?]+")
_LIST_PUNCT_RE = re.compile(r",+")
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
# A negated list can run longer than a plain negation, so the scan across list
# items is allowed further - but only through list-like tokens (see _is_negated).
_NEG_LIST_LOOKBACK_TOKENS = 20
# Separators that keep a list going rather than starting a new statement.
_LIST_JOINERS = {"or", "and", "nor"}
# A verb here means a new assertion has begun, so an earlier negator no longer
# applies: "no fever, I HAVE chest pain".
_ASSERTIONS = {
    "have", "has", "had", "having", "is", "are", "was", "were", "am",
    "got", "getting", "get", "feel", "feeling", "felt", "started", "starting",
}


def flatten(clauses: str) -> str:
    """Drop the clause markers, keeping every offset where it was."""
    return clauses.replace(CLAUSE_SENTINEL, " ").replace(LIST_SENTINEL, " ")


def _normalize_clauses(text: str) -> str:
    """Like `normalize`, but clause-ending punctuation becomes a sentinel token.

    Only `detect_redflag` uses this, to tell where one clause stops and the next
    begins. Replacing the sentinel with a space yields a string of *identical
    length*, so a match offset found in one is valid in the other - that is what
    lets terms be matched on the flat text while negation is judged on this one.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _CLAUSE_PUNCT_RE.sub(f" {CLAUSE_SENTINEL} ", text)
    text = _LIST_PUNCT_RE.sub(f" {LIST_SENTINEL} ", text)
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Lowercase, NFKC-fold, strip punctuation, collapse whitespace.

    Indic combining marks are preserved (they carry meaning); only latin
    punctuation is removed.
    """
    flat = flatten(_normalize_clauses(text))
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
    """True if a negator applies to the match at ``start``.

    Two ways that happens:

    1. A negator sits within the last few tokens, in the same clause. Kept
       short on purpose - a real negation sits close to what it negates, and
       treating a distant one as binding would suppress a real emergency.
    2. The match is an item in a negated list: "no vomiting blood, black
       stools, or chest pain". The negator sits next to the first item only, but
       it denies all of them. Scanning back across the commas is allowed only
       while every token in between is list-like - no full stop, no contrastive
       conjunction, and no verb that would start a fresh assertion. That keeps
       "no fever, I have chest pain" un-negated.

       It also requires a real list joiner ("or", "and", "nor") somewhere in the
       sentence. Without one, two comma-separated phrases are more likely to be
       a contrast than a list: "no fever, chest pain" almost certainly means the
       caller HAS chest pain, and treating that as denied would swallow an
       emergency. A missed emergency costs far more than an over-triage.
    """
    tokens = haystack[max(0, start - 400) : start].split()

    for tok in reversed(tokens[-_NEG_LOOKBACK_TOKENS:]):
        # A comma ends this tight scan too. Carrying a negation past one is the
        # list case below, and only with a joiner to prove it is a list.
        if tok in _CLAUSE_BREAKS or tok == LIST_SENTINEL:
            break
        if tok in _NEGATORS:
            return True

    if not _sentence_has_joiner(haystack, start):
        return False

    crossed_list_break = False
    for tok in reversed(tokens[-_NEG_LIST_LOOKBACK_TOKENS:]):
        if tok == LIST_SENTINEL:
            crossed_list_break = True
            continue
        if tok in _NEGATORS:
            # Only ever extend the scope through an actual list.
            return crossed_list_break
        if tok in _LIST_JOINERS:
            continue
        if tok == CLAUSE_SENTINEL or tok in _CLAUSE_BREAKS or tok in _ASSERTIONS:
            return False
        # An ordinary word: another item of the list if we are in one, and
        # otherwise a sign we have wandered out of the negator's reach.
        if not crossed_list_break:
            return False
    return False


def _sentence_has_joiner(haystack: str, start: int) -> bool:
    """Is the match inside a sentence that actually coordinates a list?"""
    left = haystack.rfind(CLAUSE_SENTINEL, 0, start)
    right = haystack.find(CLAUSE_SENTINEL, start)
    sentence = haystack[left + 1 if left >= 0 else 0 : right if right >= 0 else None]
    return any(tok in _LIST_JOINERS for tok in sentence.split())


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
    flat = flatten(clauses)

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
