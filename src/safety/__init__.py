"""MedLink safety layer: deterministic guards that do not depend on the LLM."""

from safety.redflags import RedFlagHit, detect_redflag, format_redflag_advice

__all__ = ["RedFlagHit", "detect_redflag", "format_redflag_advice"]
