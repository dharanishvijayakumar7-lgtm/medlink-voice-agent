"""Behavior lock for the deterministic emergency red-flag guard.

Per AGENTS.md: safety-critical behavior is defined by tests first.
"""

import pytest

from safety.redflags import detect_redflag, format_redflag_advice, normalize


@pytest.mark.parametrize(
    ("utterance", "expected_category"),
    [
        # English / code-switched
        ("I have been having chest pain since morning", "cardiac"),
        ("my chest is paining and it spreads to my left arm", "cardiac"),
        ("sir I cannot breathe properly", "breathing"),
        ("there is a lot of breathing problem since last night", "breathing"),
        ("suddenly my face is drooping and speech is slurred", "stroke"),
        ("the wound is bleeding a lot and blood is not stopping", "severe_bleeding"),
        ("he is unconscious and not responding", "unconscious"),
        ("my son is having a fit right now", "seizure"),
        ("my throat is closing after eating peanuts", "anaphylaxis"),
        ("I don't want to live anymore", "self_harm"),
        ("a snake bite on my leg while in the field", "poisoning_bite"),
        ("she is pregnant and bleeding heavily", "obstetric"),
        ("the baby is not drinking milk and is very drowsy", "infant_danger"),
        # Hindi (Devanagari + romanized)
        ("मुझे सीने में दर्द हो रहा है", "cardiac"),
        ("seene mein dard ho raha hai", "cardiac"),
        ("साँस नहीं आ रही है", "breathing"),
        ("saap ne kaata hai", "poisoning_bite"),
        # Dravidian scripts
        ("எனக்கு நெஞ்சு வலி இருக்கு", "cardiac"),  # Tamil
        (" എനിക്ക് ശ്വാസം കിട്ടുന്നില്ല", "breathing"),  # Malayalam
        ("ನನಗೆ ಎದೆ ನೋವು ಇದೆ", "cardiac"),  # Kannada
    ],
)
def test_emergency_phrases_are_detected(utterance, expected_category):
    hit = detect_redflag(utterance)
    assert hit is not None, f"missed red flag in: {utterance!r}"
    assert hit.category_id == expected_category


@pytest.mark.parametrize(
    "utterance",
    [
        "I have a mild headache and a runny nose since yesterday",
        "some acidity after eating spicy food",
        "my knee hurts a little when I walk",
        "I feel tired and have a low fever",
        "choti si khaansi hai",
    ],
)
def test_non_emergencies_do_not_trigger(utterance):
    assert detect_redflag(utterance) is None


@pytest.mark.parametrize(
    "utterance",
    [
        "no chest pain, just a cough",
        "I have not had any chest pain",
        "there is no bleeding now",
    ],
)
def test_clear_negations_are_skipped(utterance):
    assert detect_redflag(utterance) is None


def test_priority_ordering_prefers_time_critical_category():
    # Both breathing and dehydration cues present; cardiac/breathing win.
    hit = detect_redflag("I cannot breathe and I have not passed urine all day")
    assert hit is not None
    assert hit.priority == "emergency"
    assert hit.category_id == "breathing"


def test_empty_and_whitespace_input():
    assert detect_redflag("") is None
    assert detect_redflag("   \n  ") is None


def test_advice_is_formatted_with_local_numbers():
    hit = detect_redflag("crushing chest pain and sweating")
    assert hit is not None
    advice = format_redflag_advice(hit, emergency_number="112", ambulance_number="108")
    assert "108" in advice
    assert "{" not in advice  # no unfilled placeholders


def test_normalize_preserves_indic_marks():
    # casefold + punctuation strip, but Devanagari matra intact
    assert normalize("सीने, में  दर्द!") == "सीने में दर्द"


# --------------------------------------------------------- negation scope ---
# Regression locks for an audit that found the negation guard suppressing real
# emergencies. Each case below was a live false negative.


@pytest.mark.parametrize(
    "text",
    [
        # Punctuation must end a clause: the "no" belongs to the fever, not the
        # chest pain. Before the fix, normalize() destroyed the comma first.
        "no fever, chest pain since morning",
        "not vomiting, severe chest pain now",
        "there is no one here, I have chest pain",
        # A negator further back belongs to an earlier clause.
        "I have no doubt this is chest pain",
        # "na" is a Hindi discourse filler, "ondu" is Kannada for "one".
        # Both were in the negator list and silently killed the match.
        "mujhe na chest pain ho raha hai",
        "ondu chest pain ide",
    ],
)
def test_negation_does_not_leak_across_clauses(text):
    """A negator must not suppress an emergency it does not belong to."""
    hit = detect_redflag(text)
    assert hit is not None and hit.is_emergency, f"emergency missed in {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        "I have no chest pain",
        "no difficulty breathing at all",
        "I have not had any chest pain",
    ],
)
def test_genuine_negation_still_suppresses(text):
    """The guard must still do its job for a real denial."""
    assert detect_redflag(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "he is not breathing",
        "she stopped breathing",
        "the baby is not breathing",
    ],
)
def test_not_breathing_is_an_emergency(text):
    """The most literal phrasing of the most time-critical emergency.

    The breathing lexicon had "cannot breathe" and "not able to breathe" but no
    plain "not breathing", so this matched no term at all.
    """
    hit = detect_redflag(text)
    assert hit is not None and hit.is_emergency
    assert hit.category_id == "breathing"


def test_term_still_matches_across_punctuation():
    """Clause splitting must not stop a term matching across a comma."""
    assert detect_redflag("chest, pain") is not None


def test_indic_terms_survive_clause_splitting():
    """Unicode terms are substring-matched; the sentinel must not break them."""
    assert detect_redflag("मुझे साँस नहीं आ रही, बहुत तकलीफ है") is not None


def test_emergency_categories_precede_urgent_ones():
    """detect_redflag returns the FIRST match in file order, not the highest
    priority, despite what its docstring says. That is only safe while every
    `emergency` category is listed before every `urgent` one - otherwise an
    urgent match would mask a genuine emergency in the same sentence.
    """
    import yaml

    from config import DATA_DIR

    cats = yaml.safe_load((DATA_DIR / "redflags.yaml").read_text(encoding="utf-8"))
    priorities = [c.get("priority", "emergency") for c in cats["categories"]]
    first_urgent = next(
        (i for i, p in enumerate(priorities) if p != "emergency"), len(priorities)
    )
    assert all(p == "emergency" for p in priorities[:first_urgent])
    assert all(p != "emergency" for p in priorities[first_urgent:]), (
        "an emergency category is listed after an urgent one; file-order matching "
        "would let the urgent flag mask it"
    )


@pytest.mark.parametrize(
    "text,label",
    [
        ("எனக்கு கொஞ்சம் லைட்டா தலை வலிக்குது அப்புறம் செஸ்ட் பெயின் இருக்கு", "tamil live call"),
        ("எனக்கு செஸ்ட் பெயின் இருக்கு", "tamil loanword"),
        ("मुझे चेस्ट पेन हो रहा है", "hindi loanword"),
        ("ఛెస్ట్ పెయిన్ ఉంది", "telugu loanword"),
    ],
)
def test_english_loanwords_in_indic_script_fire(text, label):
    """Callers mix English medical words into their own language, and Sarvam STT
    transcribes them phonetically in the native script - so neither the plain
    English terms nor the native-language terms matched. Found by a live Tamil
    test call that said "செஸ்ட் பெயின்" and raised no red flag at all.
    """
    hit = detect_redflag(text)
    assert hit is not None and hit.is_emergency, f"emergency missed: {label}"


def test_indic_non_emergency_still_quiet():
    """The loanword terms must not make every Indic utterance an emergency."""
    assert detect_redflag("எனக்கு தலை வலி") is None


# ------------------------------------------ negated lists (audit finding) ---
# A comma was treated as a hard clause end, so in "no vomiting blood, black
# stools, or chest pain" the denial applied only to the first item and every
# later one scored as PRESENT. A simulated call of ordinary week-old acidity was
# assessed "urgent (score 7)" purely because the caller said they had NOT had
# black stools.


def _negated(text: str, phrase: str) -> bool:
    import re

    from safety.redflags import _is_negated, _normalize_clauses, flatten

    clauses = _normalize_clauses(text)
    pattern = r"\b" + r"\s+".join(re.escape(w) for w in phrase.split()) + r"\b"
    match = re.search(pattern, flatten(clauses))
    assert match, f"{phrase!r} not found in {text!r}"
    return _is_negated(clauses, match.start())


@pytest.mark.parametrize(
    "text,phrase",
    [
        ("no vomiting blood, black stools, or chest pain", "black stools"),
        ("no vomiting blood, black stools, or chest pain", "chest pain"),
        ("no fever, cough, or rash", "rash"),
        ("no cough, no rash, and no neck stiffness", "neck stiffness"),
    ],
)
def test_a_denial_carries_across_every_item_of_the_list(text, phrase):
    assert _negated(text, phrase)


@pytest.mark.parametrize(
    "text,phrase",
    [
        # No joiner, so these two phrases are a contrast, not a list. Reading it
        # as a denial would swallow a real emergency.
        ("no fever, chest pain", "chest pain"),
        ("no fever, chest pain since morning", "chest pain"),
        # A fresh assertion ends the denial.
        ("no fever, I have chest pain", "chest pain"),
        # So does a full stop, a contrastive conjunction, and a new sentence
        # after a genuine list.
        ("no fever. chest pain since morning", "chest pain"),
        ("no cough, but chest pain is there", "chest pain"),
        ("no fever, cough, or rash. chest pain started today", "chest pain"),
    ],
)
def test_a_denial_does_not_reach_a_symptom_the_caller_actually_reports(text, phrase):
    assert not _negated(text, phrase)


def test_ordinary_acidity_is_not_scored_urgent():
    """The whole chain, as the simulated call ran it."""
    from knowledge.triage_kb import apply_to_session
    from session_state import MedLinkUserData
    from workflows import routing

    ud = MedLinkUserData(call_id="t", caller_phone=None, channel="web")
    ud.chief_complaint = "acidity and burning in my chest after meals"
    apply_to_session(ud, ud.chief_complaint)
    ud.record_answer("duration", "About a week, still about the same.")
    ud.record_answer(
        "severity",
        "Worse after spicy food, but no vomiting blood, black stools, "
        "or chest pain on walking.",
    )
    ud.record_answer("associated", "Burning in chest after meals.")
    apply_to_session(ud, ud.chief_complaint)

    severity, urgency = routing.assess(ud)
    assert urgency == "self_care", f"scored {severity}"


@pytest.mark.parametrize(
    "text",
    ["I don't have chest pain", "I haven't had chest pain", "there isn't any chest pain"],
)
def test_contracted_negations_are_recognised(text):
    """"don't" used to normalise to "don t" and match no negator."""
    assert _negated(text, "chest pain")


def test_cannot_is_not_a_denial():
    """"I can't breathe" is the symptom itself."""
    from safety.redflags import normalize

    assert normalize("I can't breathe") == "i cant breathe"
