"""Application-level encryption for cached content at rest.

Uses Fernet (AES-128-CBC + HMAC-SHA256) from the `cryptography` package.
When CACHE_ENCRYPTION_KEY is not set, operates in passthrough mode (no encryption).
"""
from __future__ import annotations

import logging

from sriti.core.settings import settings

logger = logging.getLogger(__name__)

_fernet = None


def _get_fernet():
    global _fernet
    if _fernet is not None:
        return _fernet
    if not settings.cache_encryption_key:
        return None
    try:
        from cryptography.fernet import Fernet
        _fernet = Fernet(settings.cache_encryption_key.encode())
        return _fernet
    except Exception:
        logger.warning("Invalid CACHE_ENCRYPTION_KEY — cache encryption disabled")
        return None


def encrypt(plaintext: str) -> str:
    """Encrypt a string. Returns ciphertext if key is configured, else passthrough."""
    f = _get_fernet()
    if f is None:
        return plaintext
    return f.encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    """Decrypt a string. Returns plaintext if key is configured, else passthrough."""
    f = _get_fernet()
    if f is None:
        return ciphertext
    try:
        return f.decrypt(ciphertext.encode()).decode()
    except Exception:
        # If decryption fails (e.g., key rotated, or unencrypted legacy data), return raw
        logger.debug("Cache decryption failed — returning raw content (may be legacy unencrypted)")
        return ciphertext
