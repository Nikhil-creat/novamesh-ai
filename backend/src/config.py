"""Centralised, validated, environment-driven configuration."""
from __future__ import annotations

import base64
import binascii
from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

STREAM_VISION = "novamesh:vision:events"
STREAM_PROPOSALS = "novamesh:consensus:proposals"
STREAM_VOTES = "novamesh:consensus:votes"
TELEMETRY_CHANNEL = "novamesh:telemetry"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "NovaMesh AI"
    node_id: str = "novamesh-core-1"
    log_level: str = "INFO"

    # Groq
    groq_api_key: SecretStr
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_primary_model: str = "llama-3.3-70b-versatile"
    groq_fallback_model: str = "llama-3.1-8b-instant"
    groq_timeout_s: float = 30.0
    groq_max_concurrency: int = Field(6, ge=1, le=64)

    # Infra
    redis_url: str = "redis://localhost:6379/0"
    postgres_dsn: str = "postgresql://novamesh:novamesh@localhost:5432/novamesh"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "novamesh_kb"

    # RAG
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    crag_threshold: float = Field(0.75, ge=0.0, le=1.0)

    # Security
    jwt_secret: SecretStr
    jwt_algorithm: str = "HS256"
    jwt_issuer: str = "novamesh"
    jwt_audience: str = "novamesh-api"
    jwt_ttl_minutes: int = Field(60, ge=1, le=1440)
    aes_key_b64: SecretStr
    bootstrap_admin_user: str = "admin"
    bootstrap_admin_password: SecretStr
    injection_threshold: float = Field(0.7, ge=0.1, le=1.0)
    max_json_body_bytes: int = 256 * 1024
    max_upload_bytes: int = 10 * 1024 * 1024

    # Vision
    vision_pretrained: bool = True
    vision_anomaly_threshold: float = Field(0.80, ge=0.0, le=1.0)
    vision_warmup_frames: int = Field(30, ge=2)
    vision_queue_size: int = Field(64, ge=1)
    vision_input_size: int = 320
    rtsp_allowed_hosts: str = ""

    # Consensus / mesh
    consensus_threshold: float = Field(0.66, ge=0.5, lt=1.0)
    consensus_timeout_s: float = 30.0
    hitl_timeout_s: float = 900.0
    validator_count: int = Field(5, ge=3, le=5)
    embedded_validators: bool = True
    incident_cooldown_s: float = 60.0
    incident_concurrency: int = 4

    @field_validator("jwt_secret")
    @classmethod
    def _jwt_len(cls, v: SecretStr) -> SecretStr:
        if len(v.get_secret_value()) < 32:
            raise ValueError("JWT_SECRET must be at least 32 characters")
        return v

    @field_validator("aes_key_b64")
    @classmethod
    def _aes_len(cls, v: SecretStr) -> SecretStr:
        try:
            raw = base64.b64decode(v.get_secret_value(), validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("AES_KEY_B64 must be valid base64") from exc
        if len(raw) != 32:
            raise ValueError("AES_KEY_B64 must decode to exactly 32 bytes (AES-256)")
        return v

    @field_validator("bootstrap_admin_password")
    @classmethod
    def _pw_len(cls, v: SecretStr) -> SecretStr:
        if len(v.get_secret_value()) < 12:
            raise ValueError("BOOTSTRAP_ADMIN_PASSWORD must be at least 12 characters")
        return v

    @property
    def aes_key(self) -> bytes:
        return base64.b64decode(self.aes_key_b64.get_secret_value())

    @property
    def rtsp_hosts(self) -> frozenset[str]:
        return frozenset(h.strip().lower() for h in self.rtsp_allowed_hosts.split(",") if h.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
