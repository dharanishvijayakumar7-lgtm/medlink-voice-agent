"""MedLink controlled OTC medicine layer.

The LLM never selects a drug from raw data. ``Formulary`` retrieves candidates
from a curated, structured list; ``recommend`` applies deterministic hard safety
filters and builds the spoken recommendation only from structured fields.
"""

from medicine.filter import (
    MedicineRecommendation,
    PatientContext,
    RecommendationResult,
    recommend,
)
from medicine.formulary import Formulary, FormularyEntry, get_formulary

__all__ = [
    "Formulary",
    "FormularyEntry",
    "MedicineRecommendation",
    "PatientContext",
    "RecommendationResult",
    "get_formulary",
    "recommend",
]
