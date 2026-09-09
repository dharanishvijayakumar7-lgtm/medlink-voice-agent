"""Bhashini (government ULCA / Dhruva) speech provider - free, all 22 languages.

There is no LiveKit plugin for Bhashini, so this wraps its REST API in the
LiveKit ``stt.STT`` / ``tts.TTS`` interfaces.

Two-step API, as documented by ULCA:

1. **Pipeline config** - ``POST`` to the ULCA auth host with ``userID`` and
   ``ulcaApiKey`` headers. Returns the inference ``Authorization`` value, the
   compute ``callbackUrl``, and a ``serviceId`` per task. Cached per
   (task, language) because it rarely changes and costs a round trip.
2. **Compute** - ``POST`` to that callback URL with the Authorization header,
   base64 audio in, base64 audio or text out.

Bhashini's REST inference is request/response, not streaming, so ``BhashiniSTT``
declares ``streaming=False``. Wrap it in ``stt.StreamAdapter`` with a VAD (see
``speech.providers``) and LiveKit segments utterances for it. Bhashini also
publishes a newer streaming WebSocket ASR; when credentials for it are in hand,
add a streaming path here and flip the capability.

Nothing in this module costs money.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import wave
from dataclasses import dataclass

import httpx
from livekit import rtc
from livekit.agents import APIConnectionError, APIStatusError, stt, tts, utils
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings

logger = logging.getLogger("medlink.speech.bhashini")

ULCA_CONFIG_URL = (
    "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
)
# Used only if the config response omits a callback URL.
DEFAULT_COMPUTE_URL = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"

# Bhashini speaks ISO-639-1; MedLink carries BCP-47 everywhere else.
BCP47_TO_BHASHINI: dict[str, str] = {
    "en-IN": "en",
    "hi-IN": "hi",
    "ta-IN": "ta",
    "te-IN": "te",
    "kn-IN": "kn",
    "ml-IN": "ml",
}
DEFAULT_TTS_SAMPLE_RATE = 8000  # telephony-friendly


def to_bhashini_language(code: str) -> str:
    """BCP-47 -> ISO-639-1, tolerating a bare 2-letter code."""
    if code in BCP47_TO_BHASHINI:
        return BCP47_TO_BHASHINI[code]
    return code.split("-", 1)[0].lower()


def pcm_to_wav(pcm: bytes, sample_rate: int, num_channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(num_channels)
        handle.setsampwidth(2)  # linear16
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def wav_to_pcm(data: bytes) -> tuple[bytes, int, int]:
    """Strip a WAV container back to raw PCM. Returns (pcm, rate, channels)."""
    with wave.open(io.BytesIO(data), "rb") as handle:
        return (
            handle.readframes(handle.getnframes()),
            handle.getframerate(),
            handle.getnchannels(),
        )


class BhashiniNotConfiguredError(RuntimeError):
    """Raised when Bhashini credentials are missing."""


@dataclass(frozen=True)
class _PipelineConfig:
    compute_url: str
    auth_header: str
    auth_value: str
    service_id: str


class BhashiniClient:
    """Shared ULCA auth + Dhruva compute client."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        user_id: str | None = None,
        pipeline_id: str | None = None,
        timeout: float = 15.0,
    ) -> None:
        self._api_key = api_key or settings.bhashini_api_key
        self._user_id = user_id or settings.bhashini_user_id
        self._pipeline_id = pipeline_id or settings.bhashini_pipeline_id
        if not self._api_key:
            raise BhashiniNotConfiguredError(
                "BHASHINI_API_KEY is not set. Register at https://bhashini.gov.in "
                "and set BHASHINI_API_KEY, BHASHINI_USER_ID and BHASHINI_PIPELINE_ID."
            )
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._configs: dict[tuple[str, str], _PipelineConfig] = {}
        self._lock = asyncio.Lock()

    async def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        retry=retry_if_exception_type(httpx.TransportError),
        reraise=True,
    )
    async def _fetch_config(self, task: str, language: str) -> _PipelineConfig:
        payload: dict = {
            "pipelineTasks": [
                {"taskType": task, "config": {"language": {"sourceLanguage": language}}}
            ]
        }
        if self._pipeline_id:
            payload["pipelineRequestConfig"] = {"pipelineId": self._pipeline_id}

        client = await self._http()
        response = await client.post(
            ULCA_CONFIG_URL,
            json=payload,
            headers={"userID": self._user_id, "ulcaApiKey": self._api_key},
        )
        if response.status_code != 200:
            raise APIStatusError(
                f"Bhashini pipeline config failed: {response.text[:300]}",
                status_code=response.status_code,
            )
        body = response.json()
        endpoint = body.get("pipelineInferenceAPIEndPoint", {})
        key = endpoint.get("inferenceApiKey", {})
        service_id = ""
        for entry in body.get("pipelineResponseConfig", []):
            configs = entry.get("config", [])
            if configs:
                service_id = configs[0].get("serviceId", "")
                break
        return _PipelineConfig(
            compute_url=endpoint.get("callbackUrl") or DEFAULT_COMPUTE_URL,
            auth_header=key.get("name") or "Authorization",
            auth_value=key.get("value", ""),
            service_id=service_id,
        )

    async def config_for(self, task: str, language: str) -> _PipelineConfig:
        cache_key = (task, language)
        if cache_key in self._configs:
            return self._configs[cache_key]
        async with self._lock:
            if cache_key not in self._configs:
                self._configs[cache_key] = await self._fetch_config(task, language)
        return self._configs[cache_key]

    async def _compute(self, config: _PipelineConfig, payload: dict) -> dict:
        client = await self._http()
        response = await client.post(
            config.compute_url,
            json=payload,
            headers={config.auth_header: config.auth_value},
        )
        if response.status_code != 200:
            raise APIStatusError(
                f"Bhashini compute failed: {response.text[:300]}",
                status_code=response.status_code,
            )
        return response.json()

    # ------------------------------------------------------------------ ASR ---

    async def transcribe(
        self, wav_bytes: bytes, language: str, sample_rate: int
    ) -> str:
        lang = to_bhashini_language(language)
        config = await self.config_for("asr", lang)
        task_config: dict = {
            "language": {"sourceLanguage": lang},
            "audioFormat": "wav",
            "samplingRate": sample_rate,
        }
        if config.service_id:
            task_config["serviceId"] = config.service_id

        body = await self._compute(
            config,
            {
                "pipelineTasks": [{"taskType": "asr", "config": task_config}],
                "inputData": {
                    "audio": [{"audioContent": base64.b64encode(wav_bytes).decode()}]
                },
            },
        )
        try:
            return body["pipelineResponse"][0]["output"][0]["source"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise APIConnectionError(
                f"unexpected Bhashini ASR response shape: {str(body)[:200]}"
            ) from exc

    # ------------------------------------------------------------------ TTS ---

    async def synthesize(
        self,
        text: str,
        language: str,
        *,
        gender: str = "female",
        sample_rate: int = DEFAULT_TTS_SAMPLE_RATE,
    ) -> bytes:
        lang = to_bhashini_language(language)
        config = await self.config_for("tts", lang)
        task_config: dict = {
            "language": {"sourceLanguage": lang},
            "gender": gender,
            "samplingRate": sample_rate,
        }
        if config.service_id:
            task_config["serviceId"] = config.service_id

        body = await self._compute(
            config,
            {
                "pipelineTasks": [{"taskType": "tts", "config": task_config}],
                "inputData": {"input": [{"source": text}]},
            },
        )
        try:
            encoded = body["pipelineResponse"][0]["audio"][0]["audioContent"]
        except (KeyError, IndexError, TypeError) as exc:
            raise APIConnectionError(
                f"unexpected Bhashini TTS response shape: {str(body)[:200]}"
            ) from exc
        return base64.b64decode(encoded)


# --------------------------------------------------------------------- STT ---


class BhashiniSTT(stt.STT):
    """Non-streaming ASR. Wrap in ``stt.StreamAdapter`` with a VAD."""

    def __init__(
        self,
        *,
        language: str = "hi-IN",
        client: BhashiniClient | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._language = language
        self._client = client or BhashiniClient()

    def update_language(self, language: str) -> None:
        """Follow the caller when they switch language mid-conversation."""
        self._language = language

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language=None,
        conn_options=None,
    ) -> stt.SpeechEvent:
        frame = rtc.combine_audio_frames(buffer)
        lang = language if isinstance(language, str) and language else self._language
        wav_bytes = pcm_to_wav(bytes(frame.data), frame.sample_rate, frame.num_channels)
        text = await self._client.transcribe(wav_bytes, lang, frame.sample_rate)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(language=lang, text=text)],
        )

    async def aclose(self) -> None:
        await self._client.aclose()


# --------------------------------------------------------------------- TTS ---


class BhashiniTTS(tts.TTS):
    """Non-streaming synthesis; LiveKit buffers sentences for us."""

    def __init__(
        self,
        *,
        language: str = "hi-IN",
        gender: str = "female",
        sample_rate: int = DEFAULT_TTS_SAMPLE_RATE,
        client: BhashiniClient | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._language = language
        self._gender = gender
        self._client = client or BhashiniClient()

    def update_language(self, language: str) -> None:
        self._language = language

    def synthesize(self, text: str, *, conn_options=None) -> tts.ChunkedStream:
        kwargs = {"conn_options": conn_options} if conn_options is not None else {}
        return _BhashiniChunkedStream(tts=self, input_text=text, **kwargs)

    async def aclose(self) -> None:
        await self._client.aclose()


class _BhashiniChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        engine: BhashiniTTS = self._tts  # type: ignore[assignment]
        audio = await engine._client.synthesize(
            self.input_text,
            engine._language,
            gender=engine._gender,
            sample_rate=engine.sample_rate,
        )
        # Bhashini returns a WAV container; emit raw PCM at the declared rate.
        try:
            pcm, rate, channels = wav_to_pcm(audio)
        except wave.Error:
            pcm, rate, channels = audio, engine.sample_rate, 1

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=rate,
            num_channels=channels,
            mime_type="audio/pcm",
        )
        output_emitter.push(pcm)
        output_emitter.flush()
