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
    # ZERO-COST STACK. Speech defaults to LiveKit Inference (Deepgram + Cartesia),
    # bundled with the LiveKit Cloud free tier - no extra key, no card. Bhashini
    # (free government ULCA/ONDC APIs, better Indic quality) is the intended
    # upgrade: fill the BHASHINI_* keys and set this to "bhashini".
    # The LLM is Gemini via a Google AI Studio key, which has a genuinely free
    # tier and is NOT billed Google Cloud. Nothing here should ever incur spend.
    # One of: livekit | bhashini | google | azure | sarvam (the last three cost
    # money and are opt-in only).
    speech_provider: str = Field(default="livekit", alias="MEDLINK_SPEECH_PROVIDER")
    # LiveKit Inference model IDs used when speech_provider is "livekit" (or as the
    # fallback for any provider that cannot start). Tunable without code edits.
    stt_model: str = Field(default="deepgram/nova-3:multi", alias="MEDLINK_STT_MODEL")
    tts_model: str = Field(default="cartesia/sonic-2", alias="MEDLINK_TTS_MODEL")
    # BCP-47 codes MedLink recognizes; passed to the STT as a multi-language config.
    stt_language_codes: list[str] = Field(
        default=["en-IN", "hi-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN"]
    )
    # Optional per-language STT provider override, e.g. {"ta-IN": "sarvam"}.
    stt_language_overrides: dict[str, str] = Field(default_factory=dict)
    # 8000 Hz mono for SIP/telephony; 22050 for web frontends.
    audio_sample_rate: int = Field(default=8000, alias="MEDLINK_AUDIO_SAMPLE_RATE")
    # LLM: language-agnostic reasoning (English prompts, multilingual I/O).
    # Gemini free tier via Google AI Studio (https://aistudio.google.com/apikey)
    # - no credit card, no Google Cloud billing account.
    llm_model: str = Field(default="gemini-3.6-flash", alias="MEDLINK_LLM_MODEL")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    # If no Gemini key is set, fall back to LiveKit Inference (bundled with the
    # LiveKit Cloud free tier) so `console` mode still runs.
    llm_fallback_model: str = Field(
        default="google/gemma-3-27b-it", alias="MEDLINK_LLM_FALLBACK_MODEL"
    )

    # --- Paid providers: opt-in only, never used unless speech_provider selects them ---
    azure_speech_key: str = Field(default="", alias="AZURE_SPEECH_KEY")
    azure_speech_region: str = Field(default="", alias="AZURE_SPEECH_REGION")
    sarvam_api_key: str = Field(default="", alias="SARVAM_API_KEY")
    sarvam_stt_model: str = Field(default="saaras:v4", alias="MEDLINK_SARVAM_STT_MODEL")

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
    # Hard guard: refuse to construct any provider that bills. Keep this True
    # unless you have deliberately decided to spend money.
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
