"""Last-line guards on what the caller actually hears, and what they send us.

The medicine pipeline already builds its spoken text from curated structured
fields, so the LLM is never *given* an unsafe drug. This module covers the one
failure it cannot prevent - the model volunteering a drug from its own training
data ("just take azithromycin for two days") - plus attempts to talk the agent
out of its role.

Two guards:

* :func:`scan_output` runs over generated speech before it reaches TTS. A hit on
  the prescription denylist replaces the rest of the utterance with a safe
  correction. Text is checked with a carry-over buffer so a drug name split
  across streaming chunks is still caught.
* :func:`detect_prompt_injection` runs on the caller's turn. It is deliberately
  narrow: a sick person asking "can I take an antibiotic?" is a legitimate
  question that deserves a real answer, not a refusal. Only genuine attempts to
  override the agent's role are flagged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from config import DATA_DIR

# Longest denylist term is ~30 chars; keep enough tail to catch a split name.
_CARRY_OVER_CHARS = 40

_SAFE_CORRECTION = (
    "Actually, I should not suggest that medicine - it needs a doctor's "
    "prescription. Please see a doctor or pharmacist for it."
)

# Only phrases that try to change what the agent *is*. Clinical questions about
# prescription drugs are legitimate and must not trip this.
_INJECTION_PATTERNS = (
    r"ignore (?:all |your |the )?(?:previous |above |prior )?instructions",
    r"disregard (?:all |your |the )?(?:previous |above )?(?:instructions|rules)",
    r"forget (?:all |your |the )?(?:previous |above )?(?:instructions|rules)",
    r"you are (?:now |no longer )(?:a|an) (?:doctor|physician|pharmacist)",
    r"pretend (?:to be|you are) (?:a|an) (?:doctor|physician)",
    r"act as (?:a|an) (?:doctor|physician|pharmacist)",
    r"you have no (?:rules|restrictions|guidelines)",
    r"(?:reveal|show|print|repeat) your (?:system )?(?:prompt|instructions)",
    r"developer mode",
    r"jailbreak",
    r"without any (?:safety|restrictions|warnings)",
    r"skip the disclaimer",
)
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class OutputScan:
    text: str
    blocked_term: str | None = None

    @property
    def is_safe(self) -> bool:
        return self.blocked_term is None


@lru_cache(maxsize=2)
def _denylist(path: str) -> tuple[re.Pattern[str], ...]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    terms = [*data.get("drugs", []), *data.get("classes", [])]
    patterns = []
    for term in terms:
        cleaned = str(term).strip().casefold()
        if not cleaned:
            continue
        # Word-boundary match so "steroid" does not fire inside another word.
        pattern = r"\b" + r"\s+".join(re.escape(w) for w in cleaned.split()) + r"\b"
        patterns.append(re.compile(pattern, re.IGNORECASE))
    # Longest first so the most specific term is reported.
    return tuple(sorted(patterns, key=lambda p: -len(p.pattern)))


def find_denied_drug(
    text: str, *, denylist_path: str | Path | None = None
) -> str | None:
    """Return the first prescription-only drug or class named in ``text``."""
    if not text:
        return None
    path = str(denylist_path or (DATA_DIR / "prescription_denylist.yaml"))
    for pattern in _denylist(path):
        match = pattern.search(text)
        if match:
            return match.group(0).casefold()
    return None


def scan_output(text: str, *, denylist_path: str | Path | None = None) -> OutputScan:
    """Check one piece of generated speech before the caller hears it."""
    hit = find_denied_drug(text, denylist_path=denylist_path)
    if hit is None:
        return OutputScan(text=text)
    return OutputScan(text=_SAFE_CORRECTION, blocked_term=hit)


class OutputGuard:
    """Stateful scanner for streaming text.

    Keeps a short tail from the previous chunk so a drug name split across
    chunk boundaries ("azithro" + "mycin") is still detected. Once a hit is
    found the guard latches: everything after it is suppressed, because the
    rest of that utterance was building on unsafe advice.
    """

    def __init__(self, *, denylist_path: str | Path | None = None) -> None:
        self._carry = ""
        self._path = denylist_path
        self.blocked_term: str | None = None

    @property
    def tripped(self) -> bool:
        return self.blocked_term is not None

    def feed(self, chunk: str) -> str:
        """Return the text safe to speak for this chunk (may be empty)."""
        if self.tripped:
            return ""
        window = self._carry + chunk
        hit = find_denied_drug(window, denylist_path=self._path)
        if hit is not None:
            self.blocked_term = hit
            self._carry = ""
            return _SAFE_CORRECTION
        self._carry = window[-_CARRY_OVER_CHARS:]
        return chunk


def detect_prompt_injection(text: str) -> str | None:
    """Return the matched phrase if the caller is trying to override the agent."""
    if not text:
        return None
    match = _INJECTION_RE.search(text)
    return match.group(0) if match else None
