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
