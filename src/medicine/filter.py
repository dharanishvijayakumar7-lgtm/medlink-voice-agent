"""Deterministic OTC medicine safety pipeline.

Flow (see plan Part 6):

    symptom + allowed therapeutic classes (from triage KB)
        -> BM25 candidates from the curated formulary
        -> HARD FILTERS: OTC/schedule, India availability, age, pregnancy,
           lactation, contraindications vs disclosed conditions,
           interactions vs current medicines, symptom-duration limit
        -> rank survivors, take top N
        -> build spoken text FROM STRUCTURED FIELDS ONLY
        -> if nothing survives: safe fallback, never an unfiltered suggestion

The LLM only *reads out* what this returns; it never chooses a drug.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from config import settings
from medicine.formulary import Formulary, FormularyEntry, get_formulary

_STOPWORDS = {
    "known",
    "severe",
    "history",
    "disease",
    "problem",
    "problems",
    "current",
    "recent",
    "with",
    "the",
    "and",
    "for",
    "from",
    "any",
    "your",
    "you",
    "have",
    "under",
    "over",
    "age",
    "years",
    "old",
    "very",
    "high",
    "low",
    "or",
    "of",
    "a",
    "an",
    "in",
    "on",
    "to",
    "that",
    "this",
    "not",
    "no",
}
_WORD_RE = re.compile(r"[a-z]+")


def _keywords(text: str) -> set[str]:
    return {
        w
        for w in _WORD_RE.findall(text.casefold())
        if len(w) > 3 and w not in _STOPWORDS
    }


def _phrase_matches(rule_text: str, patient_items: list[str]) -> str | None:
    """Return the offending patient item if it plausibly matches a rule phrase.

    Conservative on the side of rejecting a medicine: a single shared meaningful
    keyword (e.g. "kidney", "ulcer", "asthma", "warfarin") is enough.
    """
    rule_kw = _keywords(rule_text)
    if not rule_kw:
        return None
    for item in patient_items:
        item = item.strip()
        if not item:
            continue
        if _keywords(item) & rule_kw:
            return item
    return None


@dataclass
class PatientContext:
    age_years: int | None = None
    is_pregnant: bool = False
    is_breastfeeding: bool = False
    current_medications: list[str] = field(default_factory=list)
    known_conditions: list[str] = field(default_factory=list)
    symptom_duration_days: int | None = None
    is_for_child: bool = False  # caller is asking on behalf of a child
    # What the caller described during triage (chief complaint + answers).
    # Checked against contraindications too: "blood in the stool" must block
    # loperamide even though it is a symptom rather than a known condition.
    reported_symptoms: list[str] = field(default_factory=list)

    def disclosed_context(self) -> list[str]:
        return [*self.known_conditions, *self.reported_symptoms]


@dataclass
class MedicineRecommendation:
    entry_id: str
    generic_name: str
    spoken_text: str
    cautions: list[str]
    structured: dict


@dataclass
class RecommendationResult:
    symptom: str
    recommendations: list[MedicineRecommendation]
    rejected: list[tuple[str, str]]  # (entry_id, human-readable reason)
    no_medicine_reason: str | None
    escalate: bool
    disclaimer: str

    @property
    def has_medicine(self) -> bool:
        return bool(self.recommendations)

    def to_spoken(self) -> str:
        """A single block the LLM can read out (it may rephrase, not alter facts)."""
        if not self.recommendations:
            return (
                f"{self.no_medicine_reason} {self.disclaimer}"
                if self.no_medicine_reason
                else self.disclaimer
            )
        parts = ["Here is what may safely help:"]
        for i, rec in enumerate(self.recommendations, 1):
            parts.append(f"\n{i}. {rec.spoken_text}")
        if self.escalate:
            parts.append(
                "\nBecause of how long this has lasted, also please see a doctor soon."
            )
        parts.append(f"\n{self.disclaimer}")
        return " ".join(parts)


def _dose_line(entry: FormularyEntry, patient: PatientContext) -> str:
    child = patient.is_for_child or (
        patient.age_years is not None and patient.age_years < 12
    )
    if child and entry.paediatric_dose:
        return f"For a child: {entry.paediatric_dose}"
    return f"Adult dose: {entry.adult_dose}"


def _build_recommendation(
    entry: FormularyEntry, patient: PatientContext, cautions: list[str]
) -> MedicineRecommendation:
    bits: list[str] = []
    name = entry.generic_name
    if entry.brand_names:
        name = f"{entry.generic_name} (sold as {entry.brand_names[0]})"
    bits.append(f"{name}.")
    if entry.indications:
        bits.append(f"It is used for {entry.indications[0]}.")
    bits.append(_dose_line(entry, patient))
    if entry.max_daily_dose:
        bits.append(f"Do not exceed {entry.max_daily_dose}")
    if entry.duration_limit_days:
        bits.append(
            f"Use it for at most {entry.duration_limit_days} day(s) for self-care."
        )
    key_contra = list(entry.contraindications[:2])
    if key_contra:
        bits.append("Do not use it if: " + "; ".join(key_contra) + ".")
    for caution in cautions:
        bits.append(caution)
    bits.append("It is available at any pharmacy or medical store.")
    if entry.overdose_note:
        bits.append(entry.overdose_note)
    bits.append("See a doctor if you are not better soon or if things get worse.")
    return MedicineRecommendation(
        entry_id=entry.id,
        generic_name=entry.generic_name,
        spoken_text=" ".join(bits),
        cautions=cautions,
        structured={
            "id": entry.id,
            "generic_name": entry.generic_name,
            "brand_names": entry.brand_names,
            "adult_dose": entry.adult_dose,
            "paediatric_dose": entry.paediatric_dose,
            "max_daily_dose": entry.max_daily_dose,
            "duration_limit_days": entry.duration_limit_days,
            "contraindications": entry.contraindications,
            "source": entry.source,
        },
    )


def recommend(
    symptom: str,
    patient: PatientContext | None = None,
    *,
    allowed_classes: set[str] | None = None,
    formulary: Formulary | None = None,
) -> RecommendationResult:
    patient = patient or PatientContext()
    fm = formulary or get_formulary()
    disclaimer = settings.disclaimer

    # A red-flag check used to sit here so an emergency presentation could never
    # be answered with an over-the-counter medicine. Removed on request along
    # with the rest of the red-flag layer.

    # Triage KB explicitly says no OTC medicine is appropriate for this presentation.
    if allowed_classes is not None and len(allowed_classes) == 0:
        return RecommendationResult(
            symptom=symptom,
            recommendations=[],
            rejected=[],
            no_medicine_reason=(
                "There is no over-the-counter medicine that is safe to recommend for "
                "this. Please see a doctor."
            ),
            escalate=True,
            disclaimer=disclaimer,
        )

    candidates = fm.search(symptom, allowed_classes=allowed_classes, limit=8)
    if not candidates:
        return RecommendationResult(
            symptom=symptom,
            recommendations=[],
            rejected=[],
            no_medicine_reason=(
                "I could not confidently match a safe over-the-counter medicine to "
                "what you described. Please rest, drink fluids, and see a doctor or "
                "pharmacist if it does not settle."
            ),
            escalate=False,
            disclaimer=disclaimer,
        )

    survivors: list[tuple[FormularyEntry, float, list[str]]] = []
    rejected: list[tuple[str, str]] = []
    escalate = False

    for entry, score in candidates:
        cautions: list[str] = []

        if not entry.is_truly_otc:
            rejected.append((entry.id, "not a non-prescription medicine in India"))
            continue

        if patient.age_years is not None and patient.age_years < entry.min_age_years:
            rejected.append(
                (
                    entry.id,
                    f"not suitable below age {entry.min_age_years}",
                )
            )
            continue

        if patient.is_pregnant:
            if entry.pregnancy == "avoid":
                rejected.append((entry.id, "should be avoided in pregnancy"))
                continue
            if entry.pregnancy == "caution":
                cautions.append(
                    "Since you are pregnant, check with a doctor or ANM before taking this."
                )

        if patient.is_breastfeeding and entry.lactation == "avoid":
            rejected.append((entry.id, "should be avoided while breastfeeding"))
            continue

        contra_hit = None
        disclosed = patient.disclosed_context()
        for rule in entry.contraindications:
            hit = _phrase_matches(rule, disclosed)
            if hit:
                contra_hit = (rule, hit)
                break
        if contra_hit:
            rejected.append(
                (
                    entry.id,
                    f"contraindicated: '{contra_hit[1]}' matches '{contra_hit[0]}'",
                )
            )
            continue

        interaction_hit = None
        for rule in entry.interactions:
            hit = _phrase_matches(rule, patient.current_medications)
            if hit:
                interaction_hit = (rule, hit)
                break
        if interaction_hit:
            rejected.append(
                (
                    entry.id,
                    f"possible interaction with '{interaction_hit[1]}'",
                )
            )
            continue

        if (
            patient.symptom_duration_days is not None
            and entry.duration_limit_days
            and patient.symptom_duration_days > entry.duration_limit_days
        ):
            escalate = True
            rejected.append(
                (
                    entry.id,
                    f"symptom lasting {patient.symptom_duration_days}d exceeds the "
                    f"{entry.duration_limit_days}d self-care limit",
                )
            )
            continue

        survivors.append((entry, score, cautions))

    survivors.sort(key=lambda t: t[1], reverse=True)

    # Never suggest two products sharing an active ingredient. Paracetamol is
    # the classic accidental-overdose route, and the formulary warns against
    # combining products that contain it.
    top: list[tuple[FormularyEntry, float, list[str]]] = []
    chosen_ingredients: set[str] = set()
    for entry, score, cautions in survivors:
        ingredients = {i.name.casefold() for i in entry.active_ingredients}
        if ingredients & chosen_ingredients:
            rejected.append(
                (entry.id, "duplicate active ingredient already recommended")
            )
            continue
        chosen_ingredients |= ingredients
        top.append((entry, score, cautions))
        if len(top) >= settings.medicine_max_results:
            break

    if not top:
        reason = (
            "The usual over-the-counter options are not safe given what you told me. "
            "Please see a doctor or pharmacist."
        )
        return RecommendationResult(
            symptom=symptom,
            recommendations=[],
            rejected=rejected,
            no_medicine_reason=reason,
            escalate=escalate,
            disclaimer=disclaimer,
        )

    recs = [_build_recommendation(e, patient, c) for e, _, c in top]
    return RecommendationResult(
        symptom=symptom,
        recommendations=recs,
        rejected=rejected,
        no_medicine_reason=None,
        escalate=escalate,
        disclaimer=disclaimer,
    )
