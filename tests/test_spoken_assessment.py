"""Wrong-language, unsupported and unfinished assessments must not reach speech."""

from types import SimpleNamespace

import pytest

from config import SUPPORTED_LANGUAGES
from session_state import MedLinkUserData
from workflows.conversation import ConversationAgent, ConversationPolicy, Decision
from workflows.conversation_text import GENERAL_ASSESSMENT


def assessment(language):
    return Decision(action="assess", language=language, **GENERAL_ASSESSMENT[language])


@pytest.mark.parametrize("language", list(SUPPORTED_LANGUAGES.values()))
@pytest.mark.parametrize("field", ["why", "do", "seek_care"])
def test_every_spoken_assessment_field_must_match_language(language, field):
    decision = assessment(language)
    other = "te-IN" if language == "en-IN" else "en-IN"
    setattr(decision, field, GENERAL_ASSESSMENT[other][field])
    policy = ConversationPolicy(questions=[(str(i), str(i)) for i in range(4)])
    with pytest.raises(ValueError, match="spoken_language"):
        policy.accept(decision)
    assert not policy.assessed


def test_telugu_sentence_with_english_diagnosis_is_rejected():
    decision = assessment("te-IN")
    decision.why = "Androgenetic alopecia వల్ల మీ జుట్టు రాలుతూ ఉండవచ్చు."
    with pytest.raises(ValueError, match="spoken_language"):
        ConversationPolicy(questions=[(str(i), str(i)) for i in range(4)]).accept(
            decision
        )


@pytest.mark.parametrize("field", ["why", "do", "seek_care"])
@pytest.mark.parametrize("ending", ["", "...", "…", "\ufffd."])
def test_unfinished_or_corrupt_closing_is_rejected(field, ending):
    decision = assessment("te-IN")
    setattr(decision, field, "మీ సమస్యకు కారణం" + ending)
    with pytest.raises(ValueError, match="spoken_text"):
        ConversationPolicy(questions=[(str(i), str(i)) for i in range(4)]).accept(
            decision
        )


@pytest.mark.parametrize("language", list(SUPPORTED_LANGUAGES.values()))
def test_general_assessments_remain_speakable(language):
    policy = ConversationPolicy(questions=[(str(i), str(i)) for i in range(4)])
    assert policy.accept(assessment(language))


@pytest.mark.parametrize("language", list(SUPPORTED_LANGUAGES.values()))
def test_wrong_script_followup_cannot_escape_assessment_checks(language):
    policy = ConversationPolicy(assessed=True)
    other = "te-IN" if language == "en-IN" else "en-IN"
    with pytest.raises(ValueError, match="spoken_language"):
        policy.accept(
            Decision(
                action="answer",
                language=language,
                answer=GENERAL_ASSESSMENT[other]["why"],
            )
        )


@pytest.mark.parametrize("language", list(SUPPORTED_LANGUAGES.values()))
def test_known_reference_without_translation_has_honest_localized_recovery(
    monkeypatch, language
):
    from knowledge.triage_kb import get_triage_kb

    data = MedLinkUserData(language=language)
    monkeypatch.setattr(
        ConversationAgent,
        "session",
        property(lambda self: SimpleNamespace(userdata=data, tts=None)),
    )
    agent = ConversationAgent()
    agent.entry = get_triage_kb().by_id["headache"]
    agent.policy.questions = [(str(i), str(i)) for i in range(6)]
    decision = agent._recovery_assessment(language)
    assert decision is not None
    assert agent._commit(decision, fallback=True)
    if language != "en-IN":
        assert decision.why == GENERAL_ASSESSMENT[language]["why"]


def test_model_assessment_schema_asks_for_exact_explanation_evidence():
    schema = Decision.model_json_schema()
    assert "explanation_quote" in schema["properties"]


def test_missing_internal_labels_do_not_discard_valid_native_explanation(monkeypatch):
    from knowledge.triage_kb import get_triage_kb

    data = MedLinkUserData(language="te-IN")
    monkeypatch.setattr(
        ConversationAgent,
        "session",
        property(lambda self: SimpleNamespace(userdata=data, tts=None)),
    )
    agent = ConversationAgent()
    agent.entry = get_triage_kb().by_id["hair_loss"]
    agent.policy.questions = [(str(i), str(i)) for i in range(4)]
    decision = Decision(
        action="assess",
        language="te-IN",
        why="మీ నాన్నకు కూడా ఇలాగే ఉండటం వల్ల వంశపారంపర్య కారణం ఒక అవకాశం కావచ్చు.",
        do="జుట్టును సున్నితంగా చూసుకోండి.",
        seek_care="కారణాన్ని తెలుసుకోవడానికి డాక్టర్ పరీక్ష అవసరం.",
    )
    assert decision.why in agent._commit(decision)
    assert decision.likely == decision.why
    assert decision.avoid


async def test_runtime_policy_is_in_one_leading_system_message(monkeypatch):
    from unittest.mock import AsyncMock

    from livekit.agents import ChatContext

    data = MedLinkUserData()
    monkeypatch.setattr(
        ConversationAgent,
        "session",
        property(lambda self: SimpleNamespace(userdata=data, tts=None)),
    )
    agent = ConversationAgent()
    agent.policy.questions = [(str(i), str(i)) for i in range(6)]
    propose = AsyncMock(return_value=assessment("en-IN"))
    agent._propose = propose
    ctx = ChatContext()
    ctx.add_message(role="system", content=agent.instructions)
    ctx.add_message(role="user", content="I have hiccups.")
    assert [p async for p in agent.llm_node(ctx, [], None)]
    sent = propose.await_args.args[0]
    instructions = [m for m in sent.items if m.role in ("system", "developer")]
    assert len(instructions) == 1
    assert sent.items[0] is instructions[0]
    assert "Question count: 6" in instructions[0].text_content


async def test_confirmed_primary_complaint_replaces_wrong_native_reference(monkeypatch):
    from unittest.mock import AsyncMock

    from livekit.agents import ChatContext

    from knowledge.triage_kb import get_triage_kb

    data = MedLinkUserData(language="te-IN")
    monkeypatch.setattr(
        ConversationAgent,
        "session",
        property(lambda self: SimpleNamespace(userdata=data, tts=None)),
    )
    agent = ConversationAgent()
    agent.entry = get_triage_kb().by_id["breathlessness"]
    agent.policy.questions = [(str(i), str(i)) for i in range(6)]
    decision = assessment("te-IN")
    decision.confirmed_symptoms = "intermittent hiccups"
    decision.explanation_quote = "ఎక్కిళ్లు"
    agent._propose = AsyncMock(return_value=decision)
    ctx = ChatContext()
    ctx.add_message(
        role="user", content="రెండు రోజులుగా ఎక్కిళ్లు వస్తున్నాయి. తినడానికి ఇబ్బందిగా ఉంది."
    )
    assert [p async for p in agent.llm_node(ctx, [], None)]
    assert data.triage_entry_id is None
    assert agent.entry is None


async def test_speech_normalization_keeps_native_words_and_complete_ending(monkeypatch):
    from livekit.agents import Agent

    data = MedLinkUserData(language="te-IN")
    monkeypatch.setattr(
        ConversationAgent,
        "session",
        property(lambda self: SimpleNamespace(userdata=data, tts=None)),
    )
    received = []

    async def fake_tts(self, text, model_settings):
        received.extend([chunk async for chunk in text])
        if False:
            yield

    monkeypatch.setattr(Agent.default, "tts_node", fake_tts)

    async def source():
        yield "మెల్లగా నీళ్లు తాగండి; "
        yield "అవసరమైతే వైద్యుడిని కలవండి।"

    frames = await ConversationAgent().tts_node(source(), None)
    assert [frame async for frame in frames] == []
    assert "".join(received) == "మెల్లగా నీళ్లు తాగండి. అవసరమైతే వైద్యుడిని కలవండి."


def test_tts_uses_larger_context_and_controlled_delivery(monkeypatch):
    from unittest.mock import Mock

    from livekit.plugins import sarvam

    from config import settings
    from speech.providers import build_tts

    monkeypatch.setattr(settings, "sarvam_api_key", "test-key-not-a-secret")
    factory = Mock()
    monkeypatch.setattr(sarvam, "TTS", factory)
    build_tts()
    kwargs = factory.call_args.kwargs
    assert kwargs["pace"] == 0.95
    assert kwargs["temperature"] == 0.4
    assert kwargs["min_buffer_size"] == 100
    assert kwargs["max_chunk_length"] == 500


@pytest.mark.parametrize(
    "name", ["पैरासिटामोल", "பாராசிட்டமால்", "పారాసిటమాల్", "ಪ್ಯಾರಾಸಿಟಮಾಲ್", "പാരസെറ്റമോൾ", "ఓఆర్ఎస్"]
)
def test_transliterated_medicine_names_do_not_bypass_medicine_free_policy(name):
    from workflows.conversation import medicine_free

    assert not medicine_free(name)


def test_configuration_repr_does_not_print_credentials():
    from config import Settings

    config = Settings.model_construct(
        sarvam_api_key="private-value-one",
        livekit_api_secret="private-value-two",
        phone_hash_key="private-value-three",
        field_encryption_key="private-value-four",
        database_url="private-value-five",
    )
    assert "private-value" not in repr(config)


@pytest.mark.parametrize("language", list(SUPPORTED_LANGUAGES.values()))
def test_language_switch_retargets_voice_and_language_together(monkeypatch, language):
    from unittest.mock import Mock

    from config import settings

    data = MedLinkUserData()
    tts = Mock()
    monkeypatch.setattr(settings, "tts_language_speakers", {"te-IN": "neha"})
    monkeypatch.setattr(
        ConversationAgent,
        "session",
        property(lambda self: SimpleNamespace(userdata=data, tts=tts)),
    )
    ConversationAgent()._language(language)
    tts.update_options.assert_called_once_with(
        target_language_code=language,
        speaker="neha" if language == "te-IN" else settings.sarvam_tts_speaker,
    )
