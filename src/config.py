"""Centralized configuration for the MedLink voice agent.

Every tunable (model IDs, API keys, thresholds, file paths, feature flags) is
read from the environment here so nothing is hardcoded across the codebase.
Loads `.env.local` for local development; in production the process environment
is authoritative.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = parent of src/. Data files and .env.local live here.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env.local")


# The six languages MedLink targets, as BCP-47 codes (shared across providers).
SUPPORTED_LANGUAGES: dict[str, str] = {
    "english": "en-IN",
    "hindi": "hi-IN",
    "tamil": "ta-IN",
    "telugu": "te-IN",
    "kannada": "kn-IN",
    "malayalam": "ml-IN",
}
DEFAULT_LANGUAGE_CODE = "en-IN"


# --- Sarvam AI model IDs ------------------------------------------------------
# STT, TTS and the LLM all run on Sarvam under one SARVAM_API_KEY. Gemini, Groq
# and Cerebras were evaluated as LLM fallbacks and removed once Sarvam won on
# both latency and Indic quality; there is no provider switching left.
# Swap these while testing credit burn; each is also overridable from the env
# via the MEDLINK_*_MODEL variables on Settings below.
#
# STT: the realtime WebSocket model. `sarvam.STTRealtime` pins `saaras:v3-realtime`
# internally and takes no model argument, so this constant records what we run
# for logging/diagnostics. `saaras:v4` is the batch-ish `sarvam.STT` class.
SARVAM_STT_MODEL = "saaras:v3-realtime"
# "fast" | "balanced" | "simulated". `fast` trims first-token latency at some
# accuracy cost; `balanced` is the Sarvam default and what a triage call wants.
SARVAM_STT_STREAM_TYPE = "balanced"
# TTS: bulbul:v3 covers all six MedLink languages and 39 voices.
SARVAM_TTS_MODEL = "bulbul:v3"
# Female, warm, customer-care tuned - the closest fit for a health helpline.
SARVAM_TTS_SPEAKER = "ritu"
# LLM: the `-conversations` variant is tuned for multi-turn voice and is the only
# model on Sarvam's /v1 endpoint. Alternatives: sarvam-105b (reasoning, /v2),
# glm5.2, gemma4 (vision). Tool calling works on all of them.
SARVAM_LLM_MODEL = "sarvam-105b-conversations"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LiveKit (transport) ---
    livekit_url: str = Field(default="", alias="LIVEKIT_URL")
    livekit_api_key: str = Field(default="", alias="LIVEKIT_API_KEY")
    livekit_api_secret: str = Field(default="", alias="LIVEKIT_API_SECRET")
    agent_name: str = Field(default="medlink-agent", alias="LIVEKIT_AGENT_NAME")

    # --- Speech / LLM ---
    # All three stages run on Sarvam under a single SARVAM_API_KEY, billed
    # against prepaid credits. Chosen for Indic quality and latency: LiveKit
    # Inference's Cartesia voice was the recurring complaint, Bulbul covers all
    # six MedLink languages natively, and Sarvam's LLM measured fastest to first
    # token of everything tried.
    # BCP-47 codes MedLink recognizes; passed to the STT as a multi-language config.
    stt_language_codes: list[str] = Field(
        default=["en-IN", "hi-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN"]
    )
    # 8000 Hz mono for SIP/telephony; 22050 for web frontends. Sarvam's realtime
    # STT accepts only 8000 or 16000; its TTS is resampled by LiveKit either way.
    audio_sample_rate: int = Field(default=8000, alias="MEDLINK_AUDIO_SAMPLE_RATE")

    # --- Turn taking ---
    # How long to wait after the caller stops speaking before replying. These
    # were cut to 0.3 / 1.8s (and Sarvam's silence to 600ms) for latency, and a
    # real phone call showed the cost: the agent answered half a sentence after
    # a natural pause and talked over the caller. Back to LiveKit's defaults -
    # the end-of-turn model only waits toward max_delay when the caller sounds
    # mid-thought. Rural callers pause; lower these only with a real call to test.
    endpointing_min_delay: float = Field(
        default=0.5, alias="MEDLINK_ENDPOINTING_MIN_DELAY"
    )
    endpointing_max_delay: float = Field(
        default=3.0, alias="MEDLINK_ENDPOINTING_MAX_DELAY"
    )
    # Silence Sarvam's own VAD waits for before closing an utterance (its default).
    stt_min_silence_ms: int = Field(default=1000, alias="MEDLINK_STT_MIN_SILENCE_MS")
    # A transcript below this confidence AND at most MAX_NOISE_WORDS long is
    # treated as line noise or echo and ignored. Real speech on a test call came
    # through at 0.98-0.99; the fragments that kept restarting the agent were
    # 0.46-0.47. A clearly heard "no" is high-confidence, so it is still kept.
    min_turn_confidence: float = Field(
        default=0.6, alias="MEDLINK_MIN_TURN_CONFIDENCE"
    )
    llm_model: str = Field(default=SARVAM_LLM_MODEL, alias="MEDLINK_LLM_MODEL")

    # --- Sarvam: the whole pipeline, one key ---
    sarvam_api_key: str = Field(default="", alias="SARVAM_API_KEY")
    sarvam_stt_model: str = Field(
        default=SARVAM_STT_MODEL, alias="MEDLINK_SARVAM_STT_MODEL"
    )
    sarvam_stt_stream_type: str = Field(
        default=SARVAM_STT_STREAM_TYPE, alias="MEDLINK_SARVAM_STT_STREAM_TYPE"
    )
    sarvam_tts_model: str = Field(
        default=SARVAM_TTS_MODEL, alias="MEDLINK_SARVAM_TTS_MODEL"
    )
    sarvam_tts_speaker: str = Field(
        default=SARVAM_TTS_SPEAKER, alias="MEDLINK_SARVAM_TTS_SPEAKER"
    )

    # --- Persistence ---
    database_url: str = Field(default="", alias="DATABASE_URL")
    # HMAC key for phone-number lookup hashing; Fernet key for at-rest PII fields.
    phone_hash_key: str = Field(default="", alias="MEDLINK_PHONE_HASH_KEY")
    field_encryption_key: str = Field(default="", alias="MEDLINK_FIELD_ENCRYPTION_KEY")
    history_retention_days: int = Field(default=90, alias="MEDLINK_RETENTION_DAYS")
    # Seconds to wait for a database connection before giving up on the write.
    db_connect_timeout: float = Field(default=3.0, alias="MEDLINK_DB_CONNECT_TIMEOUT")

    # --- Firestore export (call summaries for the mobile app) ---
    # After each call, a structured summary is written to Firestore under the
    # caller's phone number. This is the store the separate mobile app reads;
    # Postgres above stays the agent's own history.
    enable_firestore_export: bool = Field(
        default=False, alias="MEDLINK_ENABLE_FIRESTORE"
    )
    # Service-account JSON. Relative paths resolve against the project root.
    # Gitignored - the key has full admin access to the Firebase project.
    firebase_credentials_path: Path = Field(
        default=PROJECT_ROOT / "MEDLINK_FIREBASE_CREDENTIALS.json",
        alias="MEDLINK_FIREBASE_CREDENTIALS",
    )
    # The caller has already hung up by the time this runs, so a few seconds is
    # fine - but it must never hold up worker shutdown indefinitely.
    firestore_timeout: float = Field(default=10.0, alias="MEDLINK_FIRESTORE_TIMEOUT")

    # --- Telephony ---
    # Console/web sessions carry no caller ID, so there is no phone number to
    # identify a patient by and no `users` row gets created. Set this locally to
    # pretend a console session came from a given number, which makes the
    # returning-caller flow testable without telephony. Ignored whenever a real
    # SIP caller ID is present, and must stay unset in production.
    dev_caller_phone: str = Field(default="", alias="MEDLINK_DEV_CALLER_PHONE")
    sms_provider: str = Field(default="", alias="MEDLINK_SMS_PROVIDER")  # plivo|exotel
    emergency_number: str = Field(default="112", alias="MEDLINK_EMERGENCY_NUMBER")
    ambulance_number: str = Field(default="108", alias="MEDLINK_AMBULANCE_NUMBER")

    # --- Safety / conversation thresholds ---
    # BM25 score below this => "no confident medicine match" (avoids the current
    # always-return-3 behavior).
    medicine_min_score: float = Field(default=2.5, alias="MEDLINK_MEDICINE_MIN_SCORE")
    medicine_max_results: int = Field(default=2, alias="MEDLINK_MEDICINE_MAX_RESULTS")
    max_followup_questions: int = Field(default=5, alias="MEDLINK_MAX_FOLLOWUPS")
    # Severity score (0-10 scale from triage KB modifiers) routing thresholds.
    severity_urgent: int = Field(default=5, alias="MEDLINK_SEVERITY_URGENT")
    severity_emergency: int = Field(default=8, alias="MEDLINK_SEVERITY_EMERGENCY")

    # --- Feature flags (let the demo run without every service wired) ---
    enable_db: bool = Field(default=False, alias="MEDLINK_ENABLE_DB")
    enable_telephony: bool = Field(default=False, alias="MEDLINK_ENABLE_TELEPHONY")
    # Clinical content (transcript, complaint, answers, summary) is stored only
    # with the caller's spoken consent. Set MEDLINK_REQUIRE_CONSENT=false in
    # LOCAL DEVELOPMENT ONLY, so there is data to inspect before the consent
    # flow is wired. Must stay True in production.
    require_consent: bool = Field(default=True, alias="MEDLINK_REQUIRE_CONSENT")
    # Speculative LLM calls before the caller's turn is confirmed. Off: on a real
    # call it contributed to the agent starting to answer mid-sentence.
    preemptive_generation: bool = Field(
        default=False, alias="MEDLINK_PREEMPTIVE_GENERATION"
    )

    # --- Data files ---
    formulary_path: Path = Field(default=DATA_DIR / "formulary.json")
    triage_kb_path: Path = Field(default=DATA_DIR / "triage_kb.yaml")
    providers_path: Path = Field(default=DATA_DIR / "providers.json")

    @property
    def disclaimer(self) -> str:
        return (
            "I'm a health assistant, not a doctor. This is general guidance only. "
            "Please see a qualified doctor if you are worried, if things get worse, "
            "or if you do not feel better soon."
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
