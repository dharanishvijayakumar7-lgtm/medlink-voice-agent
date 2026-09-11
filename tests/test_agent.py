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
