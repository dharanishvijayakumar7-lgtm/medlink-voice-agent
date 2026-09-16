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
* The same scan also catches medicines that are sold over the counter but are
  not in our formulary, and so have had none of the age, pregnancy and
  interaction checks run against them. `Formulary.known_names()` is the
  allow-list: any name we do stock is filtered out of that list at load time, so
  Crocin, Digene, Brufen and Combiflam stay speakable. That gate existed and was
  never called - the list was built, tested, and wired to nothing.
* :func:`detect_prompt_injection` runs on the caller's turn. It is deliberately
  narrow: a sick person asking "can I take an antibiotic?" is a legitimate
  question that deserves a real answer, not a refusal. Only genuine attempts to
  override the agent's role are flagged.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

from config import DATA_DIR

logger = logging.getLogger("medlink.safety")

# Longest denylist term is ~30 chars; keep enough tail to catch a split name.
_CARRY_OVER_CHARS = 40

_SAFE_CORRECTION = (
    "Actually, I should not suggest that medicine - it needs a doctor's "
    "prescription. Please see a doctor or pharmacist for it."
)
# Not a prescription problem: these are buyable, we just have not checked them
# against this caller's age, pregnancy, other medicines and conditions.
_UNVETTED_CORRECTION = (
    "Actually, I should not advise that particular medicine - I can only guide "
    "you on the ones I have checked properly. Please ask a doctor or pharmacist "
    "about it."
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
    reason: str | None = None  # "prescription" | "unvetted"

    @property
    def is_safe(self) -> bool:
        return self.blocked_term is None


def _compile(terms: list[str]) -> tuple[re.Pattern[str], ...]:
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


def _load(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


@lru_cache(maxsize=2)
def _denylist(path: str) -> tuple[re.Pattern[str], ...]:
    data = _load(path)
    return _compile([*data.get("drugs", []), *data.get("classes", [])])


@lru_cache(maxsize=2)
def _unvetted(path: str) -> tuple[re.Pattern[str], ...]:
    """OTC medicines we do not stock, minus anything the formulary does stock.

    The subtraction is the point: it makes a mistake in the data list harmless,
    because a medicine the agent is allowed to recommend can never be silenced.
    """
    from medicine.formulary import get_formulary

    try:
        allowed = get_formulary().known_names()
    except Exception:  # a broken formulary must not disable the guard
        logger.exception("could not load the formulary allow-list")
        allowed = set()
    terms = [
        t
        for t in _load(path).get("unvetted_otc", [])
        if str(t).strip().casefold() not in allowed
    ]
    return _compile(terms)


def _first_match(patterns, text: str) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(0).casefold()
    return None


def find_denied_drug(
    text: str, *, denylist_path: str | Path | None = None
) -> str | None:
    """Return the first prescription-only drug or class named in ``text``."""
    if not text:
        return None
    return _first_match(_denylist(_path(denylist_path)), text)


def find_unvetted_medicine(
    text: str, *, denylist_path: str | Path | None = None
) -> str | None:
    """Return the first buyable-but-unchecked medicine named in ``text``."""
    if not text:
        return None
    return _first_match(_unvetted(_path(denylist_path)), text)


def _path(given: str | Path | None) -> str:
    return str(given or (DATA_DIR / "prescription_denylist.yaml"))


def scan_output(text: str, *, denylist_path: str | Path | None = None) -> OutputScan:
    """Check one piece of generated speech before the caller hears it."""
    hit = find_denied_drug(text, denylist_path=denylist_path)
    if hit is not None:
        return OutputScan(
            text=_SAFE_CORRECTION, blocked_term=hit, reason="prescription"
        )
    hit = find_unvetted_medicine(text, denylist_path=denylist_path)
    if hit is not None:
        return OutputScan(
            text=_UNVETTED_CORRECTION, blocked_term=hit, reason="unvetted"
        )
    return OutputScan(text=text)


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
        self.reason: str | None = None

    @property
    def tripped(self) -> bool:
        return self.blocked_term is not None

    def feed(self, chunk: str) -> str:
        """Return the text safe to speak for this chunk (may be empty)."""
        if self.tripped:
            return ""
        window = self._carry + chunk
        scan = scan_output(window, denylist_path=self._path)
        if not scan.is_safe:
            self.blocked_term = scan.blocked_term
            self.reason = scan.reason
            self._carry = ""
            return scan.text
        self._carry = window[-_CARRY_OVER_CHARS:]
        return chunk


def detect_prompt_injection(text: str) -> str | None:
    """Return the matched phrase if the caller is trying to override the agent."""
    if not text:
        return None
    match = _INJECTION_RE.search(text)
    return match.group(0) if match else None
