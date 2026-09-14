"""What the intake and triage tools save from a live call.

Found on a real phone call: the caller gave their name first, the agent said
"Thank you, Dharanish" and then lost it, and a replay showed the model sometimes
passes the literal string "null" for an unknown detail.
"""

import pytest

from workflows.intake import _NO_NAME
from workflows.triage import MAX_POSSIBLE_CAUSES, clean_possible_causes


def test_possible_causes_keep_order_and_drop_duplicates():
    assert clean_possible_causes(
        ["Tension headache", "dehydration", "tension  headache"]
    ) == ["Tension headache", "dehydration"]


@pytest.mark.parametrize("junk", ["null", "None", "unknown", "", "   "])
def test_placeholder_causes_are_dropped(junk):
    assert clean_possible_causes([junk, "migraine"]) == ["migraine"]


def test_possible_causes_are_capped():
    causes = [f"cause {i}" for i in range(10)]
    assert len(clean_possible_causes(causes)) == MAX_POSSIBLE_CAUSES


def test_missing_or_malformed_causes_are_an_empty_list():
    assert clean_possible_causes(None) == []
    assert clean_possible_causes([None, 42]) == []  # type: ignore[list-item]


@pytest.mark.parametrize("placeholder", ["null", "none", "unknown", "n/a"])
def test_name_placeholders_are_recognised(placeholder):
    assert placeholder in _NO_NAME


# ------------------------------------------------------------ open questions ---
# From a simulated call: the agent re-asked "how long" after being told, because
# the KB's questions were never removed once answered.

HEADACHE_KB = [
    "How long has the headache been there?",
    "Did it start suddenly and very badly, or build up slowly?",
    "Is there fever, neck stiffness, or vomiting with it?",
    "Any weakness, difficulty speaking, or trouble seeing?",
]


def test_answered_duration_is_not_asked_again():
    from session_state import MedLinkUserData
    from workflows.triage import open_questions

    ud = MedLinkUserData()
    ud.candidate_questions = list(HEADACHE_KB)
    ud.record_answer("duration", "since this morning")
    ud.record_answer("severity", "moderate")
    assert all("how long" not in q.casefold() for q in open_questions(ud))


def test_kb_warning_questions_come_before_generic_ones():
    from session_state import SLOT_QUESTIONS, MedLinkUserData
    from workflows.triage import open_questions

    ud = MedLinkUserData()
    ud.candidate_questions = list(HEADACHE_KB)
    ud.record_answer("duration", "since this morning")
    ud.record_answer("severity", "moderate")
    questions = open_questions(ud)
    assert questions[0] == "Did it start suddenly and very badly, or build up slowly?"
    assert SLOT_QUESTIONS["associated"] not in questions


def test_without_a_kb_match_the_essentials_are_asked():
    from session_state import SLOT_QUESTIONS, MedLinkUserData
    from workflows.triage import open_questions

    questions = open_questions(MedLinkUserData())
    assert questions == [
        SLOT_QUESTIONS["duration"],
        SLOT_QUESTIONS["severity"],
        SLOT_QUESTIONS["associated"],
    ]


# -------------------------------------------------------------- noise turns ---
# From a real phone call: 1- and 3-character "utterances" at 0.46-0.47
# confidence kept restarting the agent so the caller could never talk. Real
# speech on the same call came through at 0.98-0.99.


@pytest.mark.parametrize(
    "text,confidence",
    [("a", 0.46), ("hmm", 0.47), ("", 0.99), ("   ", None), ("ok yes", 0.3)],
)
def test_echo_and_line_noise_are_ignored(text, confidence):
    from workflows.base import is_noise_turn

    assert is_noise_turn(text, confidence)


@pytest.mark.parametrize(
    "text,confidence",
    [
        ("no", 0.97),  # a clearly heard one-word answer is a real answer
        ("haan", 0.9),
        ("I have a headache since morning", 0.45),  # long: let the LLM judge
        ("yes", None),  # typed/console input carries no confidence
    ],
)
def test_real_answers_are_kept(text, confidence):
    from workflows.base import is_noise_turn

    assert not is_noise_turn(text, confidence)
