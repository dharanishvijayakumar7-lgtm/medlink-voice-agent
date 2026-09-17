"""The agent must not mistake its own voice for the caller.

On speakerphone the agent's voice comes back up the line, Sarvam transcribes it,
and LiveKit - which runs its interruption check on every interim transcript in
this session - paused the agent mid-word. The caller heard half a word, a gap,
then the rest, over and over. These tests pin the guard that drops that echo and
the cases where it must NOT drop what the caller says.
"""

import pytest
from livekit.agents import stt

from echo_guard import ECHO_TAIL_SECONDS, AgentSpeech, is_echo, longest_shared_run

REPLY = (
    "From what you have told me this sounds like a viral fever. Give ORS after "
    "every loose motion, and do you have a headache or any neck stiffness?"
)


def agent_said(text: str, *, speaking: bool = True) -> AgentSpeech:
    """An AgentSpeech fed in awkward pieces, the way LLM text arrives."""
    speech = AgentSpeech()
    for i in range(0, len(text), 7):
        speech.feed(text[i : i + 7])
    speech.flush()
    if speaking:
        speech.started()
    else:
        speech.stopped(now=100.0)
    return speech


# ------------------------------------------------------ while it is talking ---


@pytest.mark.parametrize(
    "heard",
    [
        "give ors",  # the two-word interim LiveKit would already pause on
        "give ORS after every loose motion",
        "give ors after every loose notion",  # a little misheard
    ],
)
def test_its_own_words_coming_back_are_echo(heard):
    assert agent_said(REPLY).is_echo(heard)


@pytest.mark.parametrize(
    "heard",
    [
        "wait it is for my son",
        "no no my son has the fever",  # shares a word, not a sequence
        "stop please",
    ],
)
def test_a_caller_cutting_in_is_not_echo(heard):
    assert not agent_said(REPLY).is_echo(heard)


def test_a_single_word_is_never_judged():
    assert not agent_said(REPLY).is_echo("fever")
    assert not agent_said(REPLY).is_echo("")


# -------------------------------------------------- just after it stopped ---


@pytest.mark.parametrize(
    "answer",
    [
        # Answers echo the question's wording. They must survive.
        "yes I have a headache",
        "no neck stiffness",
        "no headache",
    ],
)
def test_an_answer_that_repeats_the_question_is_kept(answer):
    speech = agent_said(REPLY, speaking=False)
    assert not speech.is_echo(answer, now=100.5)


def test_the_late_tail_of_its_own_sentence_is_still_echo():
    """The STT delivers the end of the agent's sentence after it has stopped."""
    speech = agent_said(REPLY, speaking=False)
    assert speech.is_echo("do you have a headache or any neck", now=100.8)


def test_the_guard_switches_off_once_the_echo_tail_has_passed():
    speech = agent_said(REPLY, speaking=False)
    late = 100.0 + ECHO_TAIL_SECONDS + 0.5
    assert not speech.is_echo("do you have a headache or any neck", now=late)


def test_nothing_is_echo_before_the_agent_has_said_anything():
    speech = AgentSpeech()
    speech.started()
    assert not speech.is_echo("give ors after every loose motion")


# ------------------------------------------------------------- languages ---


def test_tamil_echo_is_caught_and_a_tamil_caller_is_not():
    speech = agent_said("ஒவ்வொரு பேதிக்குப் பிறகும் ஓ ஆர் எஸ் கொடுங்கள்")
    assert speech.is_echo("பேதிக்குப் பிறகும் ஓ ஆர் எஸ்")
    assert not speech.is_echo("என் மகனுக்கு காய்ச்சல்")


def test_hindi_echo_is_caught_and_a_hindi_caller_is_not():
    speech = agent_said("हर दस्त के बाद ओ आर एस दें")
    assert speech.is_echo("दस्त के बाद ओ आर एस")
    assert not speech.is_echo("रुकिए मेरे बेटे को बुखार है")


# ------------------------------------------------------------- mechanics ---


def test_a_word_split_across_chunks_is_stored_whole():
    speech = AgentSpeech()
    for piece in ("Take parac", "etamol every ", "six hours"):
        speech.feed(piece)
    speech.flush()
    assert "paracetamol" in speech.recent
    assert "parac" not in speech.recent


def test_order_is_what_separates_echo_from_an_answer():
    said = ["do", "you", "have", "a", "headache"]
    assert longest_shared_run(["have", "a", "headache"], said) == 3
    assert longest_shared_run(["headache", "have"], said) == 1
    assert is_echo("you have a", said)
    assert not is_echo("headache you", said)


# ----------------------------------------------- the stt_node wrapper itself ---


def _event(kind, text=""):
    alternatives = [stt.SpeechData(language="en", text=text)] if text else []
    return stt.SpeechEvent(type=kind, alternatives=alternatives)


async def _run_stt_node(monkeypatch, speech, events):
    from livekit.agents import Agent

    from session_state import MedLinkUserData
    from workflows.base import MedLinkAgent

    userdata = MedLinkUserData(call_id="t", caller_phone=None, channel="pstn")
    userdata.agent_speech = speech

    async def fake_default(agent, audio, model_settings):
        for event in events:
            yield event

    monkeypatch.setattr(Agent.default, "stt_node", fake_default)
    monkeypatch.setattr(MedLinkAgent, "data", property(lambda self: userdata))

    agent = MedLinkAgent(instructions="test")
    return [event async for event in agent.stt_node(None, None)]


async def test_stt_node_drops_echo_and_keeps_the_caller(monkeypatch):
    kind = stt.SpeechEventType
    events = [
        _event(kind.START_OF_SPEECH),
        _event(kind.INTERIM_TRANSCRIPT, "give ors"),
        _event(kind.INTERIM_TRANSCRIPT, "give ors after every loose motion"),
        _event(kind.FINAL_TRANSCRIPT, "give ors after every loose motion"),
        _event(kind.INTERIM_TRANSCRIPT, "wait it is for my son"),
        _event(kind.FINAL_TRANSCRIPT, "wait it is for my son"),
        _event(kind.END_OF_SPEECH),
    ]
    passed = await _run_stt_node(monkeypatch, agent_said(REPLY), events)

    texts = [e.alternatives[0].text for e in passed if e.alternatives]
    assert texts == ["wait it is for my son", "wait it is for my son"]
    # Speech boundaries are left alone - only transcripts can pause the agent.
    kinds = [e.type for e in passed]
    assert kind.START_OF_SPEECH in kinds and kind.END_OF_SPEECH in kinds


async def test_stt_node_passes_everything_when_the_agent_is_quiet(monkeypatch):
    kind = stt.SpeechEventType
    speech = agent_said(REPLY, speaking=False)
    speech.stopped_at = -1e9  # long ago
    events = [_event(kind.FINAL_TRANSCRIPT, "give ors after every loose motion")]
    passed = await _run_stt_node(monkeypatch, speech, events)
    assert len(passed) == 1


# ------------------------------------------------------ interruption config ---


def test_barge_in_can_be_switched_off_without_a_code_change(monkeypatch):
    import agent
    from config import settings

    monkeypatch.setattr(settings, "allow_barge_in", False)
    assert agent._interruption_options() == {"enabled": False}


def test_a_false_pause_resumes_quickly(monkeypatch):
    """LiveKit's 2.0 s default left a hole in the middle of a word."""
    import agent
    from config import settings

    monkeypatch.setattr(settings, "allow_barge_in", True)
    options = agent._interruption_options()
    assert options["false_interruption_timeout"] < 2.0
    assert options["min_words"] >= 2


def test_an_acronym_the_stt_writes_closed_up_still_counts_as_echo():
    """The agent spells ORS out; the STT writes it back as one word. This is the
    Hindi echo that got past the guard in the speakerphone reproduction."""
    speech = agent_said("हर दस्त के बाद ओ आर एस दें, और चौदह दिन तक जिंक दें।")
    assert speech.is_echo("हर दस के बाद ओआरएस से और चौदह दिन तक ब्रिंक से।")
    assert not speech.is_echo("रुकिए मेरे बेटे को बुखार है")


def test_the_hindi_full_stop_is_not_part_of_a_word():
    from echo_guard import words

    assert words("जिंक दें।") == ["जिंक", "दें"]
