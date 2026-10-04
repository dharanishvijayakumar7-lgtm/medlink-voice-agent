"""Demo mode: three questions, then straight to the advice.

For a demo the call has to be short - greeting, the problem, at most three
questions, then the assessment. Normal behaviour is unchanged and stays the
default; `MEDLINK_DEMO_MODE=true` is the only thing that turns this on.

What demo mode must NOT touch is pinned here too: the severity scoring, the
urgency thresholds, and the escalation path.
"""

import pytest

from config import DEMO_MAX_FOLLOWUPS, settings
from session_state import MedLinkUserData
from workflows import routing


@pytest.fixture
def demo(monkeypatch):
    monkeypatch.setattr(settings, "demo_mode", True)
    return settings


def _ud(**kwargs) -> MedLinkUserData:
    ud = MedLinkUserData(call_id="t", caller_phone=None, channel="web")
    for key, value in kwargs.items():
        setattr(ud, key, value)
    return ud


# ------------------------------------------------------------- the cap ---


def test_the_cap_is_lower_in_demo_mode(demo):
    assert settings.followup_limit == DEMO_MAX_FOLLOWUPS == 3


def test_the_normal_cap_is_untouched_by_default():
    assert settings.demo_mode is False
    assert settings.followup_limit == settings.max_followup_questions == 5


def test_demo_mode_never_raises_a_lower_configured_cap(demo, monkeypatch):
    """MEDLINK_MAX_FOLLOWUPS keeps working; demo mode only ever lowers it."""
    monkeypatch.setattr(settings, "max_followup_questions", 2)
    assert settings.followup_limit == 2


# ------------------------------------------- the gate lets the call move on ---


def test_three_questions_end_the_questioning_even_with_a_slot_empty(demo):
    """The whole point: at the cap the agent proceeds with what it has."""
    ud = _ud(chief_complaint="fever")
    ud.record_answer("duration", "since yesterday")
    ud.record_answer("severity", "quite high")
    ud.triage_turns = DEMO_MAX_FOLLOWUPS  # three questions answered
    assert "associated" not in ud.answers
    assert routing.has_enough_information(ud)


def test_two_questions_are_not_enough_yet(demo):
    ud = _ud(chief_complaint="fever")
    ud.record_answer("duration", "since yesterday")
    ud.triage_turns = 2
    assert not routing.has_enough_information(ud)


def test_questions_asked_still_counts_when_the_model_records_slots(demo):
    """Either signal reaching the cap is enough - whichever comes first."""
    ud = _ud(chief_complaint="fever", questions_asked=DEMO_MAX_FOLLOWUPS)
    assert routing.has_enough_information(ud)


def test_normal_mode_still_wants_the_warning_sign_answer():
    ud = _ud(chief_complaint="fever")
    ud.record_answer("duration", "since yesterday")
    ud.record_answer("severity", "quite high")
    ud.triage_turns = 3
    assert not routing.has_enough_information(ud)


async def test_finish_questions_is_accepted_at_the_cap(demo, monkeypatch):
    """No refusal once three questions have been asked, slot empty or not."""
    from types import SimpleNamespace

    from knowledge.triage_kb import apply_to_session
    from workflows.recommend import RecommendAgent
    from workflows.triage import TriageAgent

    monkeypatch.setattr(settings, "enable_db", False)
    ud = _ud(chief_complaint="fever")
    apply_to_session(ud, "fever")
    ud.triage_turns = DEMO_MAX_FOLLOWUPS

    result = await TriageAgent.finish_questions(
        TriageAgent(), SimpleNamespace(userdata=ud)
    )
    assert isinstance(result, RecommendAgent)


# ------------------------------------------------ question order and wording ---


def test_duration_then_severity_then_warning_signs(demo):
    """The three most informative questions, in that order."""
    from knowledge.triage_kb import apply_to_session
    from workflows.triage import open_questions

    ud = _ud(chief_complaint="fever")
    apply_to_session(ud, "fever")
    asked = open_questions(ud)
    assert len(asked) <= DEMO_MAX_FOLLOWUPS
    assert "how long" in asked[0].casefold()
    assert "how bad" in asked[1].casefold()


def test_a_question_the_caller_already_answered_is_skipped(demo):
    from knowledge.triage_kb import apply_to_session
    from workflows.triage import open_questions

    ud = _ud(chief_complaint="fever")
    ud.heard.append("I have had a fever for two days and it is quite bad.")
    apply_to_session(ud, "fever")
    asked = " ".join(open_questions(ud)).casefold()
    assert "how long" not in asked
    assert "how bad" not in asked


def test_the_demo_prompts_ask_for_a_short_answer_and_no_open_ended_loop():
    from workflows.recommend import DEMO_INSTRUCTIONS, INSTRUCTIONS

    # The normal prompt keeps the call open until the caller runs out of
    # questions; the demo one closes straight after the answer.
    assert "once they have no more questions" in INSTRUCTIONS
    assert "once they have no more questions" not in DEMO_INSTRUCTIONS
    assert "end_call" in DEMO_INSTRUCTIONS
    assert "60-80 words" in DEMO_INSTRUCTIONS
    # The four parts the caller must hear, in order.
    for part in ("what it looks like", "what to do", "what not to do", "doctor"):
        assert part in DEMO_INSTRUCTIONS.casefold()


def test_the_agent_never_says_the_word_diagnosis():
    from workflows.recommend import DEMO_INSTRUCTIONS

    assert "diagnosis" not in DEMO_INSTRUCTIONS.casefold()


def test_the_disclaimer_is_still_required_in_demo_mode():
    from workflows.recommend import DEMO_INSTRUCTIONS

    assert settings.disclaimer[:40] in DEMO_INSTRUCTIONS


# --------------------------------------------- what demo mode must not change ---


def test_severity_scoring_is_identical_in_demo_mode(monkeypatch):
    ud = _ud(chief_complaint="chest pain")
    ud.record_answer("severity", "very severe")
    ud.patient.age_years = 70
    normal = routing.compute_severity(ud)
    monkeypatch.setattr(settings, "demo_mode", True)
    assert routing.compute_severity(ud) == normal


@pytest.mark.parametrize("score", [0, 2, 4, 5, 7, 8, 10])
def test_urgency_thresholds_are_identical_in_demo_mode(score, monkeypatch):
    normal = routing.decide_urgency(score)
    monkeypatch.setattr(settings, "demo_mode", True)
    assert routing.decide_urgency(score) == normal


async def test_an_emergency_still_goes_to_escalation_in_demo_mode(demo, monkeypatch):
    """Demo mode must never shorten or skip the escalation path."""
    from types import SimpleNamespace

    from knowledge.triage_kb import apply_to_session
    from workflows.escalate import EscalateAgent
    from workflows.triage import TriageAgent

    monkeypatch.setattr(settings, "enable_db", False)
    ud = _ud(chief_complaint="chest pain")
    apply_to_session(ud, "chest pain")
    ud.record_answer("duration", "since this morning")
    ud.record_answer("severity", "very severe, spreading to my left arm")
    ud.record_answer("associated", "sweating and short of breath")
    ud.triage_turns = DEMO_MAX_FOLLOWUPS

    result = await TriageAgent.finish_questions(
        TriageAgent(), SimpleNamespace(userdata=ud)
    )
    assert isinstance(result, EscalateAgent)
    assert routing.should_escalate(ud)


def test_the_escalation_prompt_is_the_same_in_demo_mode(monkeypatch):
    from workflows.escalate import EscalateAgent

    normal = EscalateAgent().instructions
    monkeypatch.setattr(settings, "demo_mode", True)
    assert EscalateAgent().instructions == normal
    assert settings.ambulance_number in normal


async def test_the_call_closes_once_and_does_not_keep_saying_goodbye(demo):
    """A demo call ended with three goodbyes in a row."""
    from types import SimpleNamespace

    from workflows.recommend import RecommendAgent

    ud = _ud(chief_complaint="fever")
    context = SimpleNamespace(userdata=ud)

    first = await RecommendAgent.end_call(RecommendAgent(), context)
    assert "goodbye" in first.casefold()
    second = await RecommendAgent.end_call(RecommendAgent(), context)
    # Whatever a repeat returns must be safe to say out loud: the model reads
    # tool results verbatim sometimes.
    assert second == "Take care."
    assert ud.disposition == "self_care"


def test_intake_does_not_spend_a_turn_on_age_in_demo_mode(demo):
    """That question runs before the questioning stage, so the cap never sees
    it - a demo call ended up with four questions instead of three."""
    from workflows.intake import instructions

    assert "how old are you" not in instructions().casefold()
    assert "do not ask who it is for" in instructions().casefold()


def test_intake_still_asks_who_and_how_old_normally():
    from workflows.intake import instructions

    assert "how old are you" in instructions().casefold()


async def test_a_repeated_close_is_safe_to_say_aloud(demo):
    """The model read a tool note out: "the call has already been closed"."""
    from types import SimpleNamespace

    from workflows.recommend import RecommendAgent

    context = SimpleNamespace(userdata=_ud(chief_complaint="fever"))
    await RecommendAgent.end_call(RecommendAgent(), context)
    repeat = await RecommendAgent.end_call(RecommendAgent(), context)
    assert "note" not in repeat.casefold() and "closed" not in repeat.casefold()
    assert len(repeat) < 40
