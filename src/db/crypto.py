"""PII protection for stored caller data.

Two separate keys, two separate jobs:

  * ``MEDLINK_PHONE_HASH_KEY`` -> HMAC-SHA256 lookup hash. Deterministic, so a
    returning caller can be recognised, but not reversible.
  * ``MEDLINK_FIELD_ENCRYPTION_KEY`` -> Fernet (AES-128-CBC + HMAC). Reversible,
    used only where we genuinely need the value back (calling someone back,
    handing a doctor a number).

A raw phone number is never written to the database.
"""

from __future__ import annotations

import hmac
import re
from functools import lru_cache
from hashlib import sha256

from cryptography.fernet import Fernet

from config import settings

_NON_DIGITS = re.compile(r"[^\d+]")


class CryptoNotConfiguredError(RuntimeError):
    """Raised when persistence is enabled without the required keys."""


def generate_field_key() -> str:
    """Helper for operators: prints a valid Fernet key for .env.local."""
    return Fernet.generate_key().decode()


def normalise_phone(phone: str) -> str:
    """Canonical form so the same caller always hashes identically.

    Indian numbers arrive as 9876543210, +919876543210, 0091..., 09876543210.
    """
    cleaned = _NON_DIGITS.sub("", phone or "").lstrip("+")
    cleaned = cleaned.removeprefix("00")
    if len(cleaned) > 10 and cleaned.startswith("91"):
        cleaned = cleaned[2:]
    cleaned = cleaned.lstrip("0")
    return cleaned


def hash_phone(phone: str) -> str:
    """Deterministic, non-reversible lookup key for a phone number."""
    key = settings.phone_hash_key
    if not key:
        raise CryptoNotConfiguredError(
            "MEDLINK_PHONE_HASH_KEY is required when MEDLINK_ENABLE_DB=true. "
            "Set it to any long random string."
        )
    return hmac.new(key.encode(), normalise_phone(phone).encode(), sha256).hexdigest()


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    key = settings.field_encryption_key
    if not key:
        raise CryptoNotConfiguredError(
            "MEDLINK_FIELD_ENCRYPTION_KEY is required when MEDLINK_ENABLE_DB=true. "
            "Generate one with: uv run python -c "
            "'from db.crypto import generate_field_key; print(generate_field_key())'"
        )
    try:
        return Fernet(key.encode() if isinstance(key, str) else key)
    except Exception as exc:
        raise CryptoNotConfiguredError(
            f"MEDLINK_FIELD_ENCRYPTION_KEY is not a valid Fernet key: {exc}"
        ) from exc


def encrypt(plaintext: str | None) -> bytes | None:
    if plaintext is None or plaintext == "":
        return None
    return _fernet().encrypt(plaintext.encode())


def decrypt(ciphertext: bytes | None) -> str | None:
    if not ciphertext:
        return None
    return _fernet().decrypt(ciphertext).decode()
