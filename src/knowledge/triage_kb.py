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

Aliases cover English, romanized Indic and native script. The native-script
entries matter more than they look: `IntakeAgent.record_complaint` asks the
model for the complaint in English, but that is a prompt instruction, not a
guarantee. When it was the only defence, a Hindi complaint matched nothing, and
a miss used to leave `allowed_otc_classes` unset - which means *unrestricted*,
not *none*. An antacid was recommended for a urinary infection that way. A miss
now fails closed and says so in the log.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from config import settings
from safety.redflags import (
    CLAUSE_SENTINEL,
    LIST_SENTINEL,
    _is_negated,
    _normalize_clauses,
    normalize,
)

logger = logging.getLogger("medlink.kb")

# `\w` does not match Indic combining vowel signs (category Mn/Mc), so splitting
# on `[^\w]+` alone tore every Indic word apart at its matras: "मुझे बुखार है"
# tokenised to ['म','झ','ब','ख','र','ह'] and matched nothing. U+0900-U+0D7F
# covers Devanagari through Malayalam, so keeping that range as word characters
# holds each word together - except U+0964/U+0965, the danda full stops, which
# sit inside that range and would otherwise glue themselves to the last word of
# every Hindi sentence ("दें।" never equals "दें").
_TOKEN_RE = re.compile(r"[^\w\u0900-\u0963\u0966-\u0D7F]+", re.UNICODE)
# BM25 fallback needs a clear signal before it overrides "no match".
_BM25_MIN_SCORE = 3.0


# Function words carry no symptom meaning, but in a small corpus the few entries
# that happen to contain them score them as rare and important. Once acidity's
# aliases included "burning in my chest", "my hair is greying" matched acidity.
_BM25_STOP = frozenset(
    "a an the i me my we our you your he she it its they them their is am are "  # noqa: SIM905
    "was were be been being have has had do does did and or but so if of to in "
    "on at by for with from after before since this that these those there here "
    "not no very too also just feel feeling felt get got".split()
)


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.split(normalize(text)) if t]


def _content(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in _BM25_STOP]


# Negation that comes AFTER the word it denies, as in Hindi and the Dravidian
# languages: "बुखार नहीं है", "காய்ச்சல் இல்லை", "bukhar nahi hai". English
# negation comes before and is handled by `_is_negated`. "ना" is left out on
# purpose: it is also the Hindi tag "है ना?", and the red-flag module dropped it
# for suppressing real emergencies.
_POSTPOSED_NEGATORS = frozenset(
    {
        "नहीं", "नही", "nahi", "nahin", "nahee",
        "இல்லை", "இல்ல", "illai", "illa",
        "లేదు", "ledu",
        "ಇಲ್ಲ", "ಇಲ್ಲಾ",
        "ഇല്ല", "ഇല്ലാ",
    }
)
# "Headache, but no fever": a contrast ends the reach of a later negation.
_CONTRAST = frozenset(
    {"but", "par", "lekin", "पर", "लेकिन", "मगर", "ஆனால்", "కానీ", "ಆದರೆ", "പക്ഷേ"}
)
# How far after a symptom its postposed negation can sit: "बुखार या उल्टी जैसी
# कोई दिक्कत नहीं" puts "नहीं" six words after "बुखार".
_POSTPOSED_REACH = 6


def _denied(clauses: str, start: int, end: int) -> bool:
    """Did the caller say they do NOT have the symptom matched at start:end?

    The forward scan stops at a comma as well as a full stop: in "सिर में दर्द
    है, बुखार नहीं" the headache is present and only the fever is denied.
    """
    if _is_negated(clauses, start):
        return True
    for token in clauses[end : end + 200].split()[:_POSTPOSED_REACH]:
        if token in (CLAUSE_SENTINEL, LIST_SENTINEL) or token in _CONTRAST:
            return False
        if token in _POSTPOSED_NEGATORS:
            return True
    return False


# Between the words of a multi-word alias, allow up to two others: "सिर में
# हल्का दर्द" (a mild headache) is still "सिर में दर्द", and "pain in my chest"
# is still "pain in chest". A gap word can never be a clause marker, so a phrase
# cannot be assembled across a comma or full stop. Patterns run on the clause
# text, where those markers are still present.
_GAP = (
    "(?:[ ]+[^\\s"
    + re.escape(CLAUSE_SENTINEL)
    + re.escape(LIST_SENTINEL)
    + "]+){0,2}[ ]+"
)


def _alias_pattern(alias: str) -> re.Pattern[str] | None:
    """Pattern for an alias, or None to use a plain substring search.

    A plain substring search matched "burn" inside "burning" and "burns", so a
    caller with burning in the chest after meals was routed to the burn-injury
    entry and asked whether the burnt area was bigger than their palm. Latin
    aliases now match whole words only.

    Indic-script words take suffixes directly and the aliases are written to be
    found inside them, so a single Indic word stays a substring search. A
    multi-word Indic alias gets the same gap allowance as a Latin one, without
    the word boundaries.
    """
    words = alias.split()
    joined = _GAP.join(re.escape(w) for w in words)
    if alias.isascii():
        return re.compile(r"\b" + joined + r"\b")
    if len(words) == 1:
        return None
    return re.compile(joined)


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
    # "emergency" presentations win over anything else mentioned alongside them.
    priority: str = "routine"
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
        # (normalized alias, entry, pattern) sorted longest-first so "chest pain"
        # beats "pain".
        self._aliases: list[tuple[str, TriageEntry, re.Pattern[str] | None]] = sorted(
            (
                (norm, entry, _alias_pattern(norm))
                for entry in entries
                for alias in entry.all_aliases()
                if (norm := normalize(alias))
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        docs = [_content(_tokenize(e.search_text())) for e in entries]
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
        # Matching runs on the clause text, which keeps the comma and full-stop
        # markers, so a match position can also be checked for negation.
        clauses = _normalize_clauses(complaint)

        # 1. Alias match. Callers lead with their main concern, so the alias
        #    appearing EARLIEST wins ("fever and body pain" -> fever, not
        #    body_ache); ties break toward the longer, more specific alias.
        #
        #    Except for emergencies. "Earliest wins" sent "acidity and chest
        #    pain", "fever and chest pain" and "I have gas and cannot breathe" to
        #    the harmless entry - which permits medicine and asks the wrong
        #    questions. A chest-pain or breathing complaint anywhere in the
        #    sentence decides the presentation.
        #
        #    And a symptom the caller denied decides nothing. "Headache for two
        #    days, but no fever" in Hindi matched fever, so a headache caller
        #    was asked about her temperature and given fever advice.
        best: tuple[int, int, int, TriageEntry] | None = None
        saw_denied = False
        for alias, entry, pattern in self._aliases:
            if pattern is not None:
                found = pattern.search(clauses)
                start, end = (found.start(), found.end()) if found else (-1, -1)
            else:
                start = clauses.find(alias)
                end = start + len(alias)
            if start == -1:
                continue
            if _denied(clauses, start, end):
                saw_denied = True
                continue
            routine = 0 if entry.priority == "emergency" else 1
            candidate = (routine, start, -len(alias), entry)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
        if best is not None:
            return best[3]
        if saw_denied:
            # Everything recognised was something they do NOT have. Guessing a
            # presentation from that is worse than admitting there isn't one.
            return None

        # 2. BM25 fallback for phrasings the aliases missed.
        if self._bm25 is not None:
            tokens = _content(_tokenize(complaint))
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
    # Only the full-stop marker becomes a space. The comma marker is left in on
    # purpose: it is neither a word nor whitespace, so a modifier phrase cannot
    # match across it. Flattening it too would let "no_urine" match "no
    # vomiting, urine is fine" through the filler-word allowance in _affirmed.
    flat = clauses.replace(CLAUSE_SENTINEL, " ")

    total = 0
    for key, points in entry.severity_modifiers.items():
        if _modifier_matches(key, ud, flat, clauses):
            total += points
    return total


# How many unrelated words may sit between the words of a modifier phrase.
# Enough for "blood in THE stool", nowhere near enough to join words from
# opposite ends of a sentence.
_MODIFIER_GAP_WORDS = 2


def _affirmed(words: list[str], flat: str, clauses: str, *, gap: int = 0) -> bool:
    """True if the words occur close together, at least once, without a negator.

    Reuses the red-flag module's negation scope (clause breaks, punctuation,
    3-token lookback), which was tested against real negated phrasings.

    ``gap`` allows filler words between them. It must stay small. The fallback
    this replaced asked only whether each word appeared *somewhere* in the whole
    transcript, which let "no_urine" fire on a child who was passing urine
    normally: it took the "no" from "no blood" and the "urine" from "still
    passing urine", scored +6, and escalated a mild case toward an ambulance.
    """
    filler = rf"(?:\s+\w+){{0,{gap}}}\s+" if gap else r"\s+"
    pattern = r"\b" + filler.join(re.escape(w) for w in words) + r"\b"
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
    # Same phrase, allowing a couple of filler words inside it. NOT "each word
    # somewhere in the transcript" - see _affirmed.
    return _affirmed(words, flat, clauses, gap=_MODIFIER_GAP_WORDS)


def apply_to_session(ud, complaint: str | None = None) -> TriageEntry | None:
    """Match the KB against the complaint and write its guidance onto the session.

    Sets the candidate follow-up questions, the OTC categories the medicine
    filter is allowed to consider, the self-care / referral text, and the
    presentation-specific severity bonus. Safe to call repeatedly - the bonus is
    recomputed, never accumulated.
    """
    kb = get_triage_kb()
    entry = kb.match(complaint or ud.chief_complaint or "")
    if entry is None and (heard := getattr(ud, "heard", None)):
        # Callers often describe the problem over several turns. One opened with
        # "it started a week ago and gets worse with spicy food" and only said
        # "burning in my chest after meals" two answers later, so the complaint
        # alone matched nothing and a plain acidity call got no advice. Their
        # own words are tried next, one clause per turn so a denial in one turn
        # cannot reach into another.
        entry = kb.match(" . ".join(heard))

    if entry is None:
        # Fail closed. Leaving allowed_otc_classes as None means "no class
        # restriction", so every medicine in the formulary becomes eligible for
        # a complaint nothing was understood about. An empty set means "none",
        # and the medicine filter then declines and advises a doctor.
        ud.kb_severity_bonus = 0
        ud.allowed_otc_classes = set()
        ud.self_care_advice = None
        ud.refer_when = None
        logger.warning(
            "complaint matched no KB entry - no medicine will be offered",
            extra={
                "call_id": getattr(ud, "call_id", None),
                "complaint": (complaint or ud.chief_complaint or "")[:120],
            },
        )
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
