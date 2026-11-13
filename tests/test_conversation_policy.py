import pytest

from workflows.conversation import ConversationPolicy, Decision


def assessment(**kwargs):
    return Decision(
        action="assess",
        language="en-IN",
        likely="This may be reflux.",
        why="The burning follows meals.",
        do="Eat smaller meals.",
        avoid="Avoid lying down after meals.",
        seek_care="Get help if it worsens.",
        **kwargs,
    )


def test_no_seventh_question():
    policy = ConversationPolicy()
    for i in range(6):
        policy.accept(
            Decision(
                action="ask", language="en-IN", topic=str(i), question=f"Question {i}?"
            )
        )
    with pytest.raises(ValueError):
        policy.accept(
            Decision(action="ask", language="en-IN", topic="extra", question="More?")
        )
    assert policy.accept(assessment())


def test_repeated_topic_does_not_use_budget():
    policy = ConversationPolicy()
    question = Decision(
        action="ask", language="en-IN", topic="duration", question="How long?"
    )
    policy.accept(question)
    with pytest.raises(ValueError):
        policy.accept(question)
    assert len(policy.questions) == 1


def test_doses_and_medicines_rejected():
    for text in ("Take paracetamol.", "Take 500 mg.", "Take Crocin."):
        decision = assessment()
        decision.do = text
        with pytest.raises(ValueError):
            ConversationPolicy().accept(decision)


def test_incomplete_assessment_rejected():
    decision = assessment()
    decision.why = ""
    with pytest.raises(ValueError):
        ConversationPolicy().accept(decision)


@pytest.mark.parametrize(
    "language", ["en-IN", "hi-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN"]
)
def test_localized_fallback_and_emergency(language):
    from workflows.conversation import EMERGENCY, FALLBACK

    assert len(FALLBACK[language]) > 40
    assert "112" in EMERGENCY[language]


async def test_runtime_retries_seventh_question_then_falls_back(monkeypatch):
    from types import SimpleNamespace

    from livekit.agents import ChatContext

    from session_state import MedLinkUserData
    from workflows.conversation import FALLBACK, ConversationAgent

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def __aiter__(self):
            async def chunks():
                yield SimpleNamespace(
                    delta=SimpleNamespace(
                        content=Decision(
                            action="ask",
                            language="en-IN",
                            topic="extra",
                            question="More?",
                        ).model_dump_json()
                    )
                )

            return chunks()

    class Model:
        def chat(self, **kwargs):
            assert kwargs["extra_kwargs"]["response_format"]["type"] == "json_object"
            return Stream()

    data = MedLinkUserData()
    monkeypatch.setattr(
        ConversationAgent,
        "session",
        property(lambda self: SimpleNamespace(userdata=data, llm=Model(), tts=None)),
    )
    agent = ConversationAgent()
    agent.policy.questions = [(str(i), str(i)) for i in range(6)]
    ctx = ChatContext()
    ctx.add_message(role="user", content="It is still mild.")
    replies = [part async for part in agent.llm_node(ctx, [], None)]
    assert replies == [FALLBACK["en-IN"]]
    assert len(agent.policy.questions) == 6


async def test_emergency_spoken_before_any_model_request(monkeypatch):
    from types import SimpleNamespace

    from livekit.agents import ChatContext

    from session_state import MedLinkUserData
    from workflows.conversation import EMERGENCY, ConversationAgent

    data = MedLinkUserData()
    monkeypatch.setattr(
        ConversationAgent,
        "session",
        property(lambda self: SimpleNamespace(userdata=data, tts=None)),
    )
    agent = ConversationAgent()
    ctx = ChatContext()
    ctx.add_message(
        role="user", content="I have crushing chest pain and trouble breathing."
    )
    replies = [part async for part in agent.llm_node(ctx, [], None)]
    assert replies == [EMERGENCY["en-IN"]]
    assert data.escalated
    assert not agent.policy.questions
