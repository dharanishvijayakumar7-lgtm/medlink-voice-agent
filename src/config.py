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


# --- Sarvam AI model IDs (SPEECH ONLY) ---------------------------------------
# STT and TTS run on Sarvam under one SARVAM_API_KEY. The reasoning LLM does NOT
# - it has its own provider chain further down, so the stage that fires once per
# caller turn is not competing with the audio path for the same rate limit.
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


# --- Reasoning LLM: provider fallback chain ---------------------------------
# Speech stays on Sarvam, but the LLM is the chattiest stage by far (one call per
# caller turn, plus speculative ones), so it runs on its own providers to keep
# Sarvam's rate limit for STT/TTS. Tried in this order; each falls through to the
# next on a rate limit, an API error, or a slow first token. All three have a
# free tier, so the chain costs nothing until it does not.
# Gemini: Flash-Lite, pinned not aliased. Measured median time-to-first-token
# over 3 warm samples each: 3.1-flash-lite 2.07s, 3.6-flash 3.87s,
# 3.5-flash-lite 7.36s, flash-lite-latest 11.22s. Re-measure before changing.
GEMINI_LLM_MODEL = "gemini-3.1-flash-lite"
# Groq: fast inference, reliable tool calling - the usual second choice.
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"
# Cerebras: last resort, also OpenAI-compatible.
CEREBRAS_LLM_MODEL = "gpt-oss-120b"


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
    # SPEECH: Sarvam STT + TTS under a single SARVAM_API_KEY, billed against
    # prepaid credits (not a free tier - see `free_tier_only` below). Sarvam was
    # chosen for Indic quality: LiveKit Inference's Cartesia voice was the
    # recurring complaint, and Sarvam's Bulbul covers all six MedLink languages
    # natively.
    # LLM: a separate Gemini -> Groq -> Cerebras chain (see below and
    # llm_factory) - one vendor for everything meant one rate limit for
    # everything, and the LLM is the stage that fires most.
    # Bhashini (free government ULCA/ONDC APIs) remains available as a
    # zero-cost fallback: set this to "bhashini" and fill the BHASHINI_* keys.
    # One of: sarvam | bhashini | google | azure (the last two cost money and
    # have no wiring here - they are refused by the spend guard).
    speech_provider: str = Field(default="sarvam", alias="MEDLINK_SPEECH_PROVIDER")
    # BCP-47 codes MedLink recognizes; passed to the STT as a multi-language config.
    stt_language_codes: list[str] = Field(
        default=["en-IN", "hi-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN"]
    )
    # Optional per-language STT provider override, e.g. {"ta-IN": "bhashini"}.
    stt_language_overrides: dict[str, str] = Field(default_factory=dict)
    # 8000 Hz mono for SIP/telephony; 22050 for web frontends. Sarvam's realtime
    # STT accepts only 8000 or 16000; its TTS is resampled by LiveKit either way.
    audio_sample_rate: int = Field(default=8000, alias="MEDLINK_AUDIO_SAMPLE_RATE")
    # LLM: language-agnostic reasoning (English prompts, multilingual I/O).
    # Providers are tried in order; a missing key simply drops that link out of
    # the chain, so a single key is enough to run.
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default=GEMINI_LLM_MODEL, alias="MEDLINK_GEMINI_MODEL")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default=GROQ_LLM_MODEL, alias="MEDLINK_GROQ_MODEL")
    cerebras_api_key: str = Field(default="", alias="CEREBRAS_API_KEY")
    cerebras_model: str = Field(
        default=CEREBRAS_LLM_MODEL, alias="MEDLINK_CEREBRAS_MODEL"
    )
    # Seconds to wait for a provider's first token before giving up on it and
    # moving down the chain. A voice call cannot absorb much more than this.
    llm_attempt_timeout: float = Field(
        default=5.0, alias="MEDLINK_LLM_ATTEMPT_TIMEOUT"
    )
    # Sarvam's own LLM, kept for the `sarvam` entry in the chain below.
    llm_model: str = Field(default=SARVAM_LLM_MODEL, alias="MEDLINK_LLM_MODEL")

    # --- Sarvam (primary provider: STT + LLM + TTS, one key) ---
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

    # --- Paid providers with no wiring: refused by the spend guard ---
    azure_speech_key: str = Field(default="", alias="AZURE_SPEECH_KEY")
    azure_speech_region: str = Field(default="", alias="AZURE_SPEECH_REGION")

    # --- Bhashini (DEFAULT speech provider - free; custom wrapper, no LiveKit plugin) ---
    bhashini_api_key: str = Field(default="", alias="BHASHINI_API_KEY")
    bhashini_user_id: str = Field(default="", alias="BHASHINI_USER_ID")
    bhashini_pipeline_id: str = Field(default="", alias="BHASHINI_PIPELINE_ID")
    bhashini_auth_token: str = Field(default="", alias="BHASHINI_AUTH_TOKEN")

    # --- Persistence ---
    database_url: str = Field(default="", alias="DATABASE_URL")
    # HMAC key for phone-number lookup hashing; Fernet key for at-rest PII fields.
    phone_hash_key: str = Field(default="", alias="MEDLINK_PHONE_HASH_KEY")
    field_encryption_key: str = Field(default="", alias="MEDLINK_FIELD_ENCRYPTION_KEY")
    history_retention_days: int = Field(default=90, alias="MEDLINK_RETENTION_DAYS")
    # Seconds to wait for a database connection before giving up on the write.
    db_connect_timeout: float = Field(default=3.0, alias="MEDLINK_DB_CONNECT_TIMEOUT")

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
    # Speculative LLM calls before the caller's turn is confirmed: lower latency,
    # but discarded calls still count against the provider's rate limit. Off by
    # default - see the note in agent.py.
    preemptive_generation: bool = Field(
        default=False, alias="MEDLINK_PREEMPTIVE_GENERATION"
    )
    # Spend guard: refuse to construct a provider we have not deliberately
    # funded. Sarvam is exempt - it runs on prepaid credits we chose to buy, and
    # is the whole pipeline now. This still blocks google/azure, which have no
    # wiring and no budget. Keep it True.
    free_tier_only: bool = Field(default=True, alias="MEDLINK_FREE_TIER_ONLY")

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
