"""AES-256-GCM envelope for payloads in transit over the broker."""
from __future__ import annotations

import base64
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class CryptoError(ValueError):
    pass


class PayloadCipher:
    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise CryptoError("AES-256 requires a 32-byte key")
        self._aes = AESGCM(key)

    def encrypt(self, plaintext: bytes, aad: bytes = b"") -> str:
        nonce = os.urandom(12)
        return base64.urlsafe_b64encode(nonce + self._aes.encrypt(nonce, plaintext, aad)).decode()

    def decrypt(self, token: str, aad: bytes = b"") -> bytes:
        try:
            raw = base64.urlsafe_b64decode(token.encode())
            return self._aes.decrypt(raw[:12], raw[12:], aad)
        except (InvalidTag, ValueError) as exc:
            raise CryptoError("payload failed authentication") from exc
