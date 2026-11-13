from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from config import SUPPORTED_LANGUAGES
from session_state import MedLinkUserData
from workflows.conversation import ConversationAgent, ConversationPolicy, Decision


async def test_greeting_explains_role_before_inviting_the_caller(monkeypatch):
    say = AsyncMock()
    monkeypatch.setattr(
        ConversationAgent, "session", property(lambda self: SimpleNamespace(say=say))
    )
    await ConversationAgent().on_enter()
    text = say.await_args.args[0]
    assert "voice assistant" in text and "not a doctor" in text
    assert text.index("not a doctor") < text.index("bothering")


def test_closing_speaks_explanation_and_actions_without_five_part_repetition():
    policy = ConversationPolicy(questions=[(str(i), str(i)) for i in range(4)])
    decision = Decision(
        action="assess",
        language="en-IN",
        likely="Possible inherited hair thinning.",
        why="This may run in your family, since your father has similar thinning.",
        do="Handle your hair gently and let it dry naturally.",
        avoid="Avoid tight hairstyles and harsh chemicals.",
        seek_care="Book a visit to check the ongoing thinning. I am a health assistant, not a doctor.",
    )
    text = policy.accept(decision)
    assert text == (
        decision.why
        + " "
        + decision.do
        + " Book a visit to check the ongoing thinning."
    )
    assert decision.likely and decision.avoid  # internal fields stay available
    assert "not a doctor" not in text


@pytest.mark.parametrize("language", list(SUPPORTED_LANGUAGES.values()))
def test_offline_hair_closing_is_short_and_keeps_care_instructions(
    monkeypatch, language
):
    from knowledge.triage_kb import get_triage_kb
    from workflows.conversation_text import IDENTITY_NOTICES

    data = MedLinkUserData(language=language)
    monkeypatch.setattr(
        ConversationAgent,
        "session",
        property(lambda self: SimpleNamespace(userdata=data, tts=None)),
    )
    agent = ConversationAgent()
    agent.entry = get_triage_kb().by_id["hair_loss"]
    agent.policy.questions = [(str(i), str(i)) for i in range(4)]
    decision = agent._recovery_assessment(language)
    text = agent._commit(decision, fallback=True)
    assert text == " ".join((decision.why, decision.do, decision.seek_care))
    assert IDENTITY_NOTICES[language] not in text
    assert data.refer_when == decision.seek_care
    assert data.possible_causes and data.self_care_advice
    from workflows.base import _script_language

    assert "???" not in text
    if language != "en-IN":
        assert all(
            _script_language(part) == language
            for part in (decision.why, decision.do, decision.seek_care)
        )
    if language == "en-IN":
        assert len(text.split()) <= 70
        assert "painful" in text and "sudden" in text
