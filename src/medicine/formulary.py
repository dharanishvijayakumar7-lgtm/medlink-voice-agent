"""Load, validate and search the curated OTC formulary.

Fixes the prototype's bugs: heterogeneous ``composition`` (string vs list) is
normalized on load, and search applies a BM25 score threshold instead of always
returning a fixed number of results.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from rank_bm25 import BM25Okapi

from config import settings

_TOKEN_RE = re.compile(r"[^\w]+", re.UNICODE)

# Filler words that must not, on their own, make a symptom query "match" a drug.
_MATCH_STOP = {
    "and",
    "the",
    "for",
    "with",
    "not",
    "any",
    "your",
    "you",
    "has",
    "have",
    "been",
    "lot",
    "lately",
    "since",
    "from",
    "this",
    "that",
    "very",
    "too",
    "day",
    "days",
    "also",
    "some",
    "mild",
    "bad",
    "was",
    "are",
    "get",
    "got",
    "such",
    "when",
    "after",
    "before",
    "into",
    "out",
    "off",
    "per",
    "use",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.split(text.casefold()) if t]


def _content_tokens(text: str) -> set[str]:
    """Tokens usable as a relevance signal: length > 2 and not a filler word."""
    return {t for t in _tokenize(text) if len(t) > 2 and t not in _MATCH_STOP}


class ActiveIngredient(BaseModel):
    name: str
    mg: float = 0.0


class FormularyEntry(BaseModel):
    id: str
    generic_name: str
    brand_names: list[str] = Field(default_factory=list)
    form: str = ""
    strength: str = ""
    active_ingredients: list[ActiveIngredient] = Field(default_factory=list)
    otc_status: str = "OTC"  # OTC | Rx | BTC
    india_schedule: str = "OTC"  # OTC | H | H1 | X
    therapeutic_class: str = ""
    indications: list[str] = Field(default_factory=list)
    lay_terms: dict[str, list[str]] = Field(default_factory=dict)
    adult_dose: str = ""
    max_daily_dose: str = ""
    paediatric_dose: str = ""
    min_age_years: int = 0
    pregnancy: str = "caution"  # ok | caution | avoid | not_applicable
    lactation: str = "caution"
    contraindications: list[str] = Field(default_factory=list)
    interactions: list[str] = Field(default_factory=list)
    major_side_effects: list[str] = Field(default_factory=list)
    duration_limit_days: int = 0
    overdose_note: str = ""
    availability_india: bool = True
    source: str = ""
    last_reviewed: str = ""

    @field_validator("id")
    @classmethod
    def _clean_id(cls, v: str) -> str:
        if v != v.strip() or " " in v:
            raise ValueError(
                f"formulary id must have no surrounding/inner spaces: {v!r}"
            )
        return v

    @property
    def is_truly_otc(self) -> bool:
        return (
            self.otc_status.upper() == "OTC"
            and self.india_schedule.upper() == "OTC"
            and self.availability_india
        )

    def all_names(self) -> list[str]:
        return [self.generic_name, *self.brand_names]

    def search_document(self) -> str:
        lay = " ".join(w for terms in self.lay_terms.values() for w in terms)
        # Indications weighted x3, lay terms x2 - these are what callers actually say.
        return " ".join(
            [
                self.generic_name,
                " ".join(self.brand_names),
                " ".join(self.indications) * 3,
                lay * 2,
                self.therapeutic_class.replace("_", " "),
            ]
        )


class Formulary:
    def __init__(self, entries: list[FormularyEntry]):
        self.entries = entries
        self.by_id: dict[str, FormularyEntry] = {e.id: e for e in entries}
        self._docs = [_tokenize(e.search_document()) for e in entries]
        self._bm25 = BM25Okapi(self._docs) if self._docs else None
        # Tokens that represent what the medicine is actually *for* - used as a
        # relevance gate so unrelated queries ("hair falling") match nothing even
        # if BM25 finds incidental token overlap in the wider document.
        self._match_tokens: list[set[str]] = []
        for e in entries:
            parts = [
                e.generic_name,
                " ".join(e.brand_names),
                " ".join(e.indications),
                " ".join(w for terms in e.lay_terms.values() for w in terms),
                e.therapeutic_class.replace("_", " "),
            ]
            self._match_tokens.append(_content_tokens(" ".join(parts)))

    @classmethod
    def from_json(cls, path: str | Path) -> Formulary:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        medicines = raw["medicines"] if isinstance(raw, dict) else raw
        entries = [FormularyEntry.model_validate(m) for m in medicines]
        ids = [e.id for e in entries]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate formulary ids: {sorted(dupes)}")
        return cls(entries)

    def known_names(self) -> set[str]:
        """Every generic/brand name, lowercased - for the downstream allow-list gate."""
        out: set[str] = set()
        for e in self.entries:
            for name in e.all_names():
                cleaned = re.sub(r"\(.*?\)", "", name).strip().casefold()
                if cleaned:
                    out.add(cleaned)
        return out

    def search(
        self,
        query: str,
        *,
        allowed_classes: set[str] | None = None,
        limit: int = 8,
        min_score: float | None = None,
    ) -> list[tuple[FormularyEntry, float]]:
        if not self._bm25 or not query.strip():
            return []
        threshold = settings.medicine_min_score if min_score is None else min_score
        query_tokens = _tokenize(query)
        query_set = _content_tokens(query)
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[tuple[FormularyEntry, float]] = []
        for i in ranked:
            if scores[i] < threshold:
                break
            # Relevance gate: the query must overlap what the drug is actually for.
            if not (query_set & self._match_tokens[i]):
                continue
            entry = self.entries[i]
            if (
                allowed_classes is not None
                and entry.therapeutic_class not in allowed_classes
            ):
                continue
            out.append((entry, float(scores[i])))
            if len(out) >= limit:
                break
        return out


@lru_cache(maxsize=2)
def get_formulary(path: str | None = None) -> Formulary:
    return Formulary.from_json(path or settings.formulary_path)
