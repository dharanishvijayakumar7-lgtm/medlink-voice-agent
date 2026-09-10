"""Behavior lock for the Bhashini speech wrapper.

The ULCA/Dhruva HTTP layer is mocked, so the request shapes, response parsing,
language mapping and audio conversion are all verified without credentials and
without a network call. The wire contract itself still needs one live smoke test
once a BHASHINI_API_KEY is available.
"""

import base64

import httpx
import pytest
from livekit.agents import APIStatusError, inference, stt

from config import settings
from speech import bhashini as bh
from speech.providers import (
    PAID_PROVIDERS,
    PaidProviderBlockedError,
    build_stt,
    build_tts,
)

SAMPLE_RATE = 16000
CONFIG_RESPONSE = {
    "pipelineResponseConfig": [{"config": [{"serviceId": "svc-asr-hi"}]}],
    "pipelineInferenceAPIEndPoint": {
        "callbackUrl": "https://dhruva.example/services/inference/pipeline",
        "inferenceApiKey": {"name": "Authorization", "value": "secret-token"},
    },
}


def _pcm(seconds: float = 0.2) -> bytes:
    return b"\x00\x01" * int(SAMPLE_RATE * seconds)


class Recorder:
    """Captures the requests our client makes and replays canned responses."""

    def __init__(self, *, asr_text="namaste", tts_audio=None, fail_status=None):
        self.requests: list[httpx.Request] = []
        self.payloads: list[dict] = []
        self.asr_text = asr_text
        self.tts_audio = tts_audio or bh.pcm_to_wav(_pcm(), 8000)
        self.fail_status = fail_status

    def handler(self, request: httpx.Request) -> httpx.Response:
        import json

        self.requests.append(request)
        payload = json.loads(request.content or b"{}")
        self.payloads.append(payload)

        if self.fail_status:
            return httpx.Response(self.fail_status, text="upstream exploded")

        if str(request.url) == bh.ULCA_CONFIG_URL:
            return httpx.Response(200, json=CONFIG_RESPONSE)

        task = payload["pipelineTasks"][0]["taskType"]
        if task == "asr":
            return httpx.Response(
                200,
                json={"pipelineResponse": [{"output": [{"source": self.asr_text}]}]},
            )
        return httpx.Response(
            200,
            json={
                "pipelineResponse": [
                    {
                        "audio": [
                            {"audioContent": base64.b64encode(self.tts_audio).decode()}
                        ]
                    }
                ]
            },
        )


def _client(recorder: Recorder) -> bh.BhashiniClient:
    client = bh.BhashiniClient(
        api_key="test-key", user_id="test-user", pipeline_id="pipe-1"
    )
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(recorder.handler))
    return client


# ------------------------------------------------------------- conversions ---


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("en-IN", "en"),
        ("hi-IN", "hi"),
        ("ta-IN", "ta"),
        ("te-IN", "te"),
        ("kn-IN", "kn"),
        ("ml-IN", "ml"),
        ("hi", "hi"),
        ("MR-IN", "mr"),
    ],
)
def test_language_codes_map_to_bhashini(code, expected):
    assert bh.to_bhashini_language(code) == expected


def test_all_supported_languages_are_mappable():
    from config import SUPPORTED_LANGUAGES

    for code in SUPPORTED_LANGUAGES.values():
        assert code in bh.BCP47_TO_BHASHINI


def test_wav_roundtrip_preserves_pcm():
    pcm = _pcm()
    wav = bh.pcm_to_wav(pcm, SAMPLE_RATE)
    assert wav[:4] == b"RIFF"
    out, rate, channels = bh.wav_to_pcm(wav)
    assert out == pcm
    assert rate == SAMPLE_RATE
    assert channels == 1


# --------------------------------------------------------------------- ASR ---


async def test_transcribe_sends_the_documented_payload():
    rec = Recorder(asr_text="mujhe bukhar hai")
    client = _client(rec)

    text = await client.transcribe(
        bh.pcm_to_wav(_pcm(), SAMPLE_RATE), "hi-IN", SAMPLE_RATE
    )
    assert text == "mujhe bukhar hai"

    config_req, compute_req = rec.requests
    assert str(config_req.url) == bh.ULCA_CONFIG_URL
    assert config_req.headers["userID"] == "test-user"
    assert config_req.headers["ulcaApiKey"] == "test-key"

    assert (
        str(compute_req.url)
        == CONFIG_RESPONSE["pipelineInferenceAPIEndPoint"]["callbackUrl"]
    )
    assert compute_req.headers["Authorization"] == "secret-token"

    task = rec.payloads[1]["pipelineTasks"][0]
    assert task["taskType"] == "asr"
    assert task["config"]["language"]["sourceLanguage"] == "hi"
    assert task["config"]["serviceId"] == "svc-asr-hi"
    assert task["config"]["samplingRate"] == SAMPLE_RATE
    assert rec.payloads[1]["inputData"]["audio"][0]["audioContent"]

    await client.aclose()


async def test_pipeline_config_is_cached_across_calls():
    rec = Recorder()
    client = _client(rec)
    wav = bh.pcm_to_wav(_pcm(), SAMPLE_RATE)

    await client.transcribe(wav, "hi-IN", SAMPLE_RATE)
    await client.transcribe(wav, "hi-IN", SAMPLE_RATE)

    config_calls = [r for r in rec.requests if str(r.url) == bh.ULCA_CONFIG_URL]
    assert len(config_calls) == 1, "pipeline config should be fetched once per language"
    await client.aclose()


async def test_each_language_gets_its_own_config():
    rec = Recorder()
    client = _client(rec)
    wav = bh.pcm_to_wav(_pcm(), SAMPLE_RATE)

    await client.transcribe(wav, "hi-IN", SAMPLE_RATE)
    await client.transcribe(wav, "ta-IN", SAMPLE_RATE)

    config_calls = [r for r in rec.requests if str(r.url) == bh.ULCA_CONFIG_URL]
    assert len(config_calls) == 2
    await client.aclose()


async def test_stt_returns_a_final_transcript_event():
    from livekit import rtc

    rec = Recorder(asr_text="enakku kaaichal irukku")
    engine = bh.BhashiniSTT(language="ta-IN", client=_client(rec))
    frame = rtc.AudioFrame(
        data=_pcm(),
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=SAMPLE_RATE // 5,
    )

    event = await engine._recognize_impl(frame)
    assert event.type == stt.SpeechEventType.FINAL_TRANSCRIPT
    assert event.alternatives[0].text == "enakku kaaichal irukku"
    assert event.alternatives[0].language == "ta-IN"
    await engine.aclose()


async def test_stt_language_can_change_mid_conversation():
    rec = Recorder()
    engine = bh.BhashiniSTT(language="hi-IN", client=_client(rec))
    engine.update_language("ml-IN")

    from livekit import rtc

    frame = rtc.AudioFrame(
        data=_pcm(),
        sample_rate=SAMPLE_RATE,
        num_channels=1,
        samples_per_channel=SAMPLE_RATE // 5,
    )
    await engine._recognize_impl(frame)
    assert (
        rec.payloads[1]["pipelineTasks"][0]["config"]["language"]["sourceLanguage"]
        == "ml"
    )
    await engine.aclose()


# --------------------------------------------------------------------- TTS ---


async def test_synthesize_sends_the_documented_payload():
    rec = Recorder()
    client = _client(rec)

    audio = await client.synthesize("aapko bukhar hai", "hi-IN", sample_rate=8000)
    assert audio == rec.tts_audio

    task = rec.payloads[1]["pipelineTasks"][0]
    assert task["taskType"] == "tts"
    assert task["config"]["language"]["sourceLanguage"] == "hi"
    assert task["config"]["gender"] == "female"
    assert task["config"]["samplingRate"] == 8000
    assert rec.payloads[1]["inputData"]["input"][0]["source"] == "aapko bukhar hai"
    await client.aclose()


# ------------------------------------------------------------------ errors ---


async def test_upstream_failure_raises_api_status_error():
    rec = Recorder(fail_status=503)
    client = _client(rec)
    with pytest.raises(APIStatusError):
        await client.transcribe(
            bh.pcm_to_wav(_pcm(), SAMPLE_RATE), "hi-IN", SAMPLE_RATE
        )
    await client.aclose()


async def test_unexpected_response_shape_is_reported_clearly():
    def handler(request):
        if str(request.url) == bh.ULCA_CONFIG_URL:
            return httpx.Response(200, json=CONFIG_RESPONSE)
        return httpx.Response(200, json={"unexpected": "shape"})

    client = bh.BhashiniClient(api_key="k", user_id="u", pipeline_id="p")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(Exception, match="unexpected Bhashini ASR response"):
        await client.transcribe(
            bh.pcm_to_wav(_pcm(), SAMPLE_RATE), "hi-IN", SAMPLE_RATE
        )
    await client.aclose()


def test_missing_credentials_fail_loudly(monkeypatch):
    monkeypatch.setattr(settings, "bhashini_api_key", "")
    with pytest.raises(bh.BhashiniNotConfiguredError, match="BHASHINI_API_KEY"):
        bh.BhashiniClient()


# ------------------------------------------------------- provider selection ---


def test_missing_key_degrades_to_livekit_inference(monkeypatch):
    monkeypatch.setattr(settings, "speech_provider", "bhashini")
    monkeypatch.setattr(settings, "bhashini_api_key", "")
    # Must not raise - the agent has to start regardless.
    assert build_stt() is not None
    assert build_tts() is not None


@pytest.mark.parametrize("provider", ["google", "azure", "sarvam"])
def test_paid_providers_are_refused_under_free_tier_only(monkeypatch, provider):
    monkeypatch.setattr(settings, "speech_provider", provider)
    monkeypatch.setattr(settings, "free_tier_only", True)
    with pytest.raises(PaidProviderBlockedError, match="bills per use"):
        build_stt()
    with pytest.raises(PaidProviderBlockedError):
        build_tts()


def test_bhashini_is_never_treated_as_a_paid_provider():
    """The free provider must never be caught by the spend guard."""
    assert "bhashini" not in PAID_PROVIDERS


def test_livekit_is_never_treated_as_paid():
    """The default free provider must never be caught by the spend guard."""
    assert "livekit" not in PAID_PROVIDERS


def test_livekit_is_the_explicit_free_default(monkeypatch):
    """speech_provider=livekit builds LiveKit Inference STT/TTS, no spend guard."""
    monkeypatch.setattr(settings, "speech_provider", "livekit")
    monkeypatch.setattr(settings, "free_tier_only", True)
    assert isinstance(build_stt(), inference.STT)
    assert isinstance(build_tts(), inference.TTS)
