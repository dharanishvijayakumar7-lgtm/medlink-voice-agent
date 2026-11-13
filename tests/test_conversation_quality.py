"""Regressions from the two-year hair-loss phone call. No provider calls."""

from types import SimpleNamespace

import pytest
from livekit.agents import ChatContext

from knowledge.triage_kb import get_triage_kb
from session_state import MedLinkUserData
from workflows.conversation import ConversationAgent, Decision


class Model:
    def __init__(self, response):
        self.response = response
        self.contexts = []

    def chat(self, **kwargs):
        self.contexts.append(kwargs["chat_ctx"])
        return Stream(self.response)


class Stream:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    def __aiter__(self):
        async def chunks():
            if isinstance(self.response, Exception):
                raise self.response
            yield SimpleNamespace(delta=SimpleNamespace(content=self.response))

        return chunks()


def setup_agent(monkeypatch, response, language="en-IN"):
    model = Model(response)
    data = MedLinkUserData(language=language)
    tts = SimpleNamespace(update_options=lambda **kwargs: None)
    monkeypatch.setattr(
        ConversationAgent,
        "session",
        property(lambda self: SimpleNamespace(userdata=data, llm=model, tts=tts)),
    )
    agent = ConversationAgent()
    return agent, data, model


def context(*callers):
    ctx = ChatContext()
    for text in callers:
        ctx.add_message(role="user", content=text)
    return ctx


@pytest.mark.parametrize(
    "complaint",
    [
        "hair fall",
        "hair loss",
        "My hair has thinned gradually at the front hairline for two years.",
        "बाल झड़ रहे हैं",
        "முடி உதிர்கிறது",
        "జుట్టు రాలుతోంది",
        "ಕೂದಲು ಉದುರುತ್ತಿದೆ",
        "മുടി കൊഴിയുന്നു",
    ],
)
def test_hair_loss_has_a_relevant_reference(complaint):
    entry = get_triage_kb().match(complaint)
    assert entry is not None and entry.id == "hair_loss"
    assert len(entry.followup_questions) >= 6
    assert entry.self_care and entry.avoid and entry.explanation


async def test_reference_is_supplied_before_first_question(monkeypatch):
    decision = Decision(
        action="ask",
        language="en-IN",
        topic="duration",
        question="How long have you noticed hair loss?",
    )
    agent, data, model = setup_agent(monkeypatch, decision.model_dump_json())
    replies = [p async for p in agent.llm_node(context("I have hair fall."), [], None)]
    assert replies
    prompt = " ".join(m.text_content or "" for m in model.contexts[0].items)
    assert "hair_pattern" in prompt and "family_history" in prompt
    assert "scalp" in prompt.lower()
    assert data.triage_entry_id == "hair_loss"


async def test_two_year_answer_and_invalid_output_recover_with_hair_question(
    monkeypatch,
):
    agent, data, model = setup_agent(monkeypatch, "invalid JSON")
    agent.policy.questions = [("duration", "How long have you had hair fall?")]
    replies = [
        p
        async for p in agent.llm_node(
            context("I have hair fall.", "About two years."), [], None
        )
    ]
    assert len(model.contexts) == 2
    assert len(agent.policy.questions) == 2 and not agent.policy.assessed
    assert agent.actions[-1].topic == "hair_pattern"
    assert "patch" in replies[0].lower() or "hairline" in replies[0].lower()
    assert "severe" not in replies[0].lower()
    assert data.questions_asked == 2


@pytest.mark.parametrize(
    "language,complaint",
    [
        ("en-IN", "I have hair fall."),
        ("hi-IN", "मेरे बाल झड़ रहे हैं।"),
        ("ta-IN", "எனக்கு முடி உதிர்கிறது."),
        ("te-IN", "నాకు జుట్టు రాలుతోంది."),
        ("kn-IN", "ನನ್ನ ಕೂದಲು ಉದುರುತ್ತಿದೆ."),
        ("ml-IN", "എനിക്ക് മുടി കൊഴിയുന്നു."),
    ],
)
async def test_invalid_assessment_has_complete_localized_hair_guidance(
    monkeypatch, language, complaint
):
    agent, data, _ = setup_agent(monkeypatch, "invalid JSON", language)
    agent.policy.questions = [
        (t, t) for t in ("duration", "hair_pattern", "scalp_changes", "family_history")
    ]
    replies = [p async for p in agent.llm_node(context(complaint), [], None)]
    assert len(replies) == 1
    assessment = agent.actions[-1]
    assert assessment.action == "assess" and assessment.language == language
    assert all(
        (
            assessment.likely,
            assessment.why,
            assessment.do,
            assessment.avoid,
            assessment.seek_care,
        )
    )
    assert agent.policy.assessed and data.self_care_advice
    if language == "en-IN":
        assert "gentle" in assessment.do.lower()
        assert "tight" in assessment.avoid.lower()
        assert "rest and look after yourself" not in replies[0].lower()


async def test_unknown_complaint_keeps_a_valid_uncertain_assessment(monkeypatch):
    decision = Decision(
        action="assess",
        language="en-IN",
        confirmed_symptoms="persistent hiccups",
        likely="I cannot identify the cause of these hiccups on a call.",
        why="Several causes are possible, and these details do not distinguish them.",
        do="Keep a record of when they occur and any triggers.",
        avoid="Avoid trying unverified remedies.",
        seek_care="Arrange medical review because they are persistent.",
    )
    agent, _data, _ = setup_agent(monkeypatch, decision.model_dump_json())
    agent.policy.questions = [(str(i), str(i)) for i in range(4)]
    replies = [
        p async for p in agent.llm_node(context("I have persistent hiccups."), [], None)
    ]
    assert decision.do in replies[0]
    assert agent.actions[-1].action == "assess"


async def test_provider_timeout_does_not_pretend_to_be_a_diagnosis(monkeypatch):
    agent, _, _ = setup_agent(monkeypatch, TimeoutError())
    replies = [p async for p in agent.llm_node(context("I have hair loss."), [], None)]
    assert replies and not agent.policy.assessed
    assert not agent.policy.questions
    assert "connection" in replies[0].lower() or "service" in replies[0].lower()


async def test_repeat_for_language_switch_does_not_spend_question_budget(monkeypatch):
    decision = {
        "action": "repeat",
        "language": "hi-IN",
        "topic": "hair_pattern",
        "question": "बाल कहाँ से झड़ रहे हैं?",
    }
    import json

    agent, data, _ = setup_agent(monkeypatch, json.dumps(decision))
    agent.policy.questions = [
        ("duration", "How long?"),
        ("hair_pattern", "Where is the thinning?"),
    ]
    replies = [
        p
        async for p in agent.llm_node(
            context("I have hair loss.", "Please speak Hindi."), [], None
        )
    ]
    assert replies == [decision["question"]]
    assert len(agent.policy.questions) == 2 and data.language == "hi-IN"


async def test_duration_cannot_be_misclassified_as_a_request_to_stop(monkeypatch):
    decision = Decision(
        action="assess",
        language="en-IN",
        stop_quote="About two years.",
        likely="The cause is uncertain.",
        why="Duration alone is not enough.",
        do="Handle hair gently.",
        avoid="Avoid tight hairstyles.",
        seek_care="Arrange a review.",
    )
    agent, _, _ = setup_agent(monkeypatch, decision.model_dump_json())
    agent.policy.questions = [("duration", "How long?")]
    replies = [
        p
        async for p in agent.llm_node(
            context("I have hair loss.", "About two years."), [], None
        )
    ]
    assert replies and not agent.policy.assessed
    assert agent.actions[-1].action == "ask"


async def test_unrelated_hair_severity_question_is_replaced(monkeypatch):
    decision = Decision(
        action="ask", language="en-IN", topic="severity", question="How severe is it?"
    )
    agent, _, _ = setup_agent(monkeypatch, decision.model_dump_json())
    agent.policy.questions = [("duration", "How long?")]
    replies = [
        p
        async for p in agent.llm_node(
            context("I have hair loss.", "Two years."), [], None
        )
    ]
    assert "severe" not in replies[0]
    assert agent.actions[-1].topic == "hair_pattern"


async def test_uncited_model_fact_cannot_change_the_assessment(monkeypatch):
    from workflows.conversation import ReportedFact

    decision = Decision(
        action="ask",
        language="en-IN",
        topic="duration",
        question="How long?",
        facts=[
            ReportedFact(
                topic="family_history",
                value="Father has hair loss",
                quote="My father has hair loss",
            )
        ],
    )
    agent, _, _ = setup_agent(monkeypatch, decision.model_dump_json())
    replies = [p async for p in agent.llm_node(context("I have hair loss."), [], None)]
    assert replies and "family_history" not in agent.facts


def test_duration_alias_cannot_spend_another_question():
    from workflows.conversation import ConversationPolicy

    policy = ConversationPolicy()
    policy.accept(
        Decision(
            action="ask",
            language="en-IN",
            topic="duration_of_hiccups",
            question="How long have you had hiccups?",
        )
    )
    with pytest.raises(ValueError, match="already"):
        policy.accept(
            Decision(
                action="ask",
                language="en-IN",
                topic="duration",
                question="How long has this been happening?",
            )
        )


async def test_legacy_reference_recovery_does_not_reask_duration(monkeypatch):
    agent, _, _ = setup_agent(monkeypatch, "invalid JSON")
    agent.policy.questions = [("duration", "When did your fever start?")]
    replies = [
        p
        async for p in agent.llm_node(context("I have fever.", "Yesterday."), [], None)
    ]
    assert replies
    assert "how many days" not in replies[0].lower()
    assert agent.actions[-1].topic != "duration"
