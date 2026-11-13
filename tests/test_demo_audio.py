"""Audio wiring that keeps noisy demo calls from cancelling replies."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from livekit import rtc

import agent
from config import settings


def test_protected_replies_discard_overlapping_input(monkeypatch):
    monkeypatch.setattr(settings, "allow_barge_in", False)
    assert agent._interruption_options() == {
        "enabled": False,
        "discard_audio_if_uninterruptible": True,
    }


def test_optional_barge_in_resumes_false_interruptions(monkeypatch):
    monkeypatch.setattr(settings, "allow_barge_in", True)
    options = agent._interruption_options()
    assert options["mode"] == "adaptive"
    assert options["resume_false_interruption"] is True


@pytest.mark.parametrize("sip", [True, False])
def test_filter_selected_by_participant_kind(monkeypatch, sip):
    monkeypatch.setattr(agent.krisp, "voice_isolation_telephony", lambda: "phone")
    monkeypatch.setattr(agent.krisp, "voice_isolation", lambda: "web")
    participant = SimpleNamespace(
        kind=rtc.ParticipantKind.PARTICIPANT_KIND_SIP
        if sip
        else rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
    )
    assert agent._noise_cancellation(SimpleNamespace(participant=participant)) == (
        "phone" if sip else "web"
    )


def test_worker_preloads_llm_resources():
    import sys

    agent._prewarm(None)
    assert "openai.resources.models" in sys.modules
    assert "openai.resources.chat.completions.completions" in sys.modules
    assert agent.server.setup_fnc is agent._prewarm


@pytest.mark.parametrize(
    "agent_cls",
    [
        agent.IntakeAgent,
        pytest.param("TriageAgent"),
        pytest.param("RecommendAgent"),
        pytest.param("EscalateAgent"),
    ],
)
async def test_protected_audio_is_silent_to_stt_then_listening_resumes(
    monkeypatch, agent_cls
):
    """Exercise SDK audio routing with each workflow's effective speech policy."""
    from livekit.agents import AgentSession, TurnHandlingOptions
    from livekit.agents.voice.agent_activity import AgentActivity

    import workflows

    monkeypatch.setattr(settings, "allow_barge_in", False)
    workflow = (
        getattr(workflows, agent_cls)() if isinstance(agent_cls, str) else agent_cls()
    )
    session = AgentSession(
        turn_handling=TurnHandlingOptions(
            interruption=agent._interruption_options(),
            turn_detection="manual",
        ),
        vad=None,
    )
    activity = AgentActivity(workflow, session)
    activity._started = True
    recognition = Mock()
    activity._audio_recognition = recognition
    activity._current_speech = SimpleNamespace(
        done=lambda: False,
        interrupted=False,
        allow_interruptions=activity.allow_interruptions,
    )
    assert not activity._current_speech.allow_interruptions
    frame = rtc.AudioFrame(
        data=b"\x01\x01" * 160,
        sample_rate=8000,
        num_channels=1,
        samples_per_channel=160,
    )
    activity.push_audio(frame)
    muted = recognition._push_audio.call_args.kwargs["stt_frame"]
    assert muted is not None and not any(muted.data)
    activity._current_speech = None
    activity.push_audio(frame)
    assert recognition._push_audio.call_args.args[0] is frame
    assert recognition._push_audio.call_args.kwargs["stt_frame"] is None
    await session.aclose()
