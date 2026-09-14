"""Wiring checks for the agent graph.

Conversational behaviour is covered by the simulations in scenarios.yaml, which
run full conversations on LiveKit Cloud. These are the structural guarantees
worth pinning without a live session or an LLM - above all, which tools each
agent is allowed to reach.
"""

import pytest

from config import settings
from workflows import EscalateAgent, IntakeAgent, RecommendAgent, TriageAgent

ALL_AGENTS = (IntakeAgent, TriageAgent, RecommendAgent, EscalateAgent)


def _tool_names(agent) -> set[str]:
    return {getattr(t, "name", getattr(t, "__name__", "?")) for t in agent.tools}


@pytest.mark.parametrize("agent_cls", ALL_AGENTS)
def test_every_agent_constructs_with_instructions(agent_cls):
    agent = agent_cls()
    assert agent.instructions.strip()


@pytest.mark.parametrize("agent_cls", ALL_AGENTS)
def test_prompts_stay_small(agent_cls):
    """The whole point of the workflow split - no more one 4k mega-prompt."""
    assert len(agent_cls().instructions) < 2500


def test_escalate_agent_cannot_reach_the_medicine_tool():
    """Structural guarantee: during an emergency there is no drug tool to call."""
    assert "get_medicine_guidance" not in _tool_names(EscalateAgent())


def test_only_the_recommend_agent_has_the_medicine_tool():
    holders = [
        cls.__name__
        for cls in ALL_AGENTS
        if "get_medicine_guidance" in _tool_names(cls())
    ]
    assert holders == ["RecommendAgent"]


def test_intake_only_records_the_complaint():
    assert _tool_names(IntakeAgent()) == {"record_complaint"}


def test_triage_can_record_answers_and_route():
    tools = _tool_names(TriageAgent())
    assert {"record_answer", "record_patient_context", "finish_questions"} <= tools
    # Triage must not be able to hand out medicine before it has assessed.
    assert "get_medicine_guidance" not in tools


def test_escalate_can_capture_consent():
    assert "record_consent" in _tool_names(EscalateAgent())


@pytest.mark.parametrize("agent_cls", ALL_AGENTS)
def test_every_agent_carries_the_shared_safety_hook(agent_cls):
    from workflows.base import MedLinkAgent

    assert isinstance(agent_cls(), MedLinkAgent)
    assert hasattr(agent_cls(), "on_user_turn_completed")
    assert hasattr(agent_cls(), "tts_node")


def test_escalate_prompt_forbids_medicine_explicitly():
    instructions = EscalateAgent().instructions.lower()
    assert "never suggest any medicine" in instructions
    assert settings.ambulance_number in EscalateAgent().instructions


def test_recommend_prompt_forbids_prescription_drugs():
    instructions = RecommendAgent().instructions.lower()
    assert "antibiotic" in instructions
    assert "prescription" in instructions


# ----------------------------------------------------- multilingual output ---


def test_every_supported_language_has_a_greeting():
    """The caller hears the greeting before any LLM runs, so a missing language
    silently falls back to English. Only en-IN and hi-IN existed, which is part
    of why the agent appeared to speak just Hindi and English.
    """
    from config import SUPPORTED_LANGUAGES
    from workflows.intake import GREETINGS

    missing = sorted(set(SUPPORTED_LANGUAGES.values()) - set(GREETINGS))
    assert not missing, f"no greeting for {missing}"


def test_greetings_are_in_native_script():
    """Sarvam's Bulbul expects native script; romanised text is mispronounced."""
    from workflows.intake import GREETINGS

    for code, text in GREETINGS.items():
        if code == "en-IN":
            continue
        assert any(ord(ch) > 0x0900 for ch in text), f"{code} greeting is not native script"


def test_sarvam_tts_can_retarget_every_supported_language():
    """The runtime language switch depends on this; all six must be accepted."""
    from livekit.plugins.sarvam.tts import SarvamTTSLanguages

    from config import SUPPORTED_LANGUAGES

    allowed = set(SarvamTTSLanguages.__args__)
    missing = sorted(set(SUPPORTED_LANGUAGES.values()) - allowed)
    assert not missing, f"Sarvam TTS cannot speak {missing}"


# ------------------------------------------------------------ caller identity ---
# Real phone calls were being filed under MEDLINK_DEV_CALLER_PHONE: the caller's
# number was read from the room before the agent had connected, when there are
# no remote participants yet. These pin the fixed lookup.


class _Participant:
    def __init__(self, kind, attributes):
        self.kind = kind
        self.attributes = attributes


class _Room:
    def __init__(self, name):
        self.name = name


class _Ctx:
    def __init__(self, room_name="medlink-call_x", participant=None, delay=0.0):
        self.room = _Room(room_name)
        self._participant = participant
        self._delay = delay
        self.waited = False

    async def wait_for_participant(self, **_):
        import asyncio

        self.waited = True
        await asyncio.sleep(self._delay)
        return self._participant


async def test_sip_caller_number_is_read_from_the_joined_participant(monkeypatch):
    from livekit import rtc

    import agent

    monkeypatch.setattr(settings, "dev_caller_phone", "+919000000000")
    sip = _Participant(
        rtc.ParticipantKind.PARTICIPANT_KIND_SIP, {"sip.phoneNumber": "+916361754795"}
    )
    assert await agent._caller_phone(_Ctx(participant=sip)) == "+916361754795"


async def test_non_sip_participant_falls_back_to_the_dev_number(monkeypatch):
    from livekit import rtc

    import agent

    monkeypatch.setattr(settings, "dev_caller_phone", "+919000000000")
    web = _Participant(rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD, {})
    assert await agent._caller_phone(_Ctx(participant=web)) == "+919000000000"


async def test_console_session_does_not_wait_for_a_participant(monkeypatch):
    import agent

    monkeypatch.setattr(settings, "dev_caller_phone", "+919000000000")
    ctx = _Ctx(room_name="console")
    assert await agent._caller_phone(ctx) == "+919000000000"
    assert ctx.waited is False


async def test_caller_that_never_joins_times_out_to_the_fallback(monkeypatch):
    import agent

    monkeypatch.setattr(settings, "dev_caller_phone", "")
    monkeypatch.setattr(agent, "CALLER_WAIT_TIMEOUT", 0.01)
    assert await agent._caller_phone(_Ctx(delay=1.0)) is None


async def test_mocked_non_string_phone_attribute_is_ignored(monkeypatch):
    """A mocked room returns a truthy non-string that broke normalise_phone()."""
    from livekit import rtc

    import agent

    monkeypatch.setattr(settings, "dev_caller_phone", "")
    odd = _Participant(rtc.ParticipantKind.PARTICIPANT_KIND_SIP, {"sip.phoneNumber": object()})
    assert await agent._caller_phone(_Ctx(participant=odd)) is None
