"""Strict Pydantic v2 domain schemas shared across the platform."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Severity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def rank(self) -> int:
        return _SEV_RANK[self]

    @classmethod
    def max_of(cls, *items: Severity) -> Severity:
        return max(items, key=lambda s: s.rank)


_SEV_RANK = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2, Severity.CRITICAL: 3}


class Role(StrEnum):
    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"

    @property
    def rank(self) -> int:
        return {"viewer": 0, "operator": 1, "admin": 2}[self.value]


# ----------------------------------------------------------------- vision
class BoundingBox(BaseModel):
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)
    score: float = Field(ge=0, le=1)


class VisionFrameEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: new_id("vf"))
    source_id: str
    captured_at: datetime = Field(default_factory=utcnow)
    anomaly_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    boxes: list[BoundingBox] = Field(default_factory=list)
    embedding: list[float] = Field(min_length=512, max_length=512)
    model_name: str
    latency_ms: float


class VisionAnalysis(BaseModel):
    event: VisionFrameEvent
    anomalous: bool
    published: bool


# ----------------------------------------------------------------- auth
class LoginRequest(Strict):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class MintRequest(Strict):
    subject: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:@-]+$")
    role: Role
    ttl_minutes: int = Field(60, ge=1, le=1440)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    role: Role


class TokenClaims(BaseModel):
    sub: str
    role: Role
    exp: int


# ----------------------------------------------------------------- assistant
class ChatRequest(Strict):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[A-Za-z0-9_.:-]+$")


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    tools_used: list[str]
    model: str
    prompt_tokens: int
    completion_tokens: int
    guardrail_flags: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------- mesh / consensus
class ActionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    args: dict[str, Any] = Field(default_factory=dict)


class ExecutionProposal(BaseModel):
    proposal_id: str = Field(default_factory=lambda: new_id("prop"))
    incident_id: str
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    severity: Severity
    rationale: str
    evidence: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def requires_hitl(self) -> bool:
        return self.severity is Severity.CRITICAL


class Vote(StrEnum):
    AGREE = "AGREE"
    DISAGREE = "DISAGREE"
    ABSTAIN = "ABSTAIN"


class VoteMessage(BaseModel):
    proposal_id: str
    validator_id: str
    vote: Vote
    rationale: str = Field(max_length=1000)
    cast_at: datetime = Field(default_factory=utcnow)


class ConsensusStatus(StrEnum):
    PENDING = "PENDING"
    REJECTED = "REJECTED"
    PASSED = "PASSED"
    AWAITING_HITL = "AWAITING_HITL"
    APPROVED = "APPROVED"
    HITL_DENIED = "HITL_DENIED"
    EXPIRED = "EXPIRED"
    EXECUTING = "EXECUTING"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"


class ConsensusResult(BaseModel):
    proposal_id: str
    agree_weight: float
    disagree_weight: float
    abstain_weight: float
    total_weight: float
    participation: float
    approval_ratio: float
    threshold: float
    quorum_met: bool
    passed: bool
    votes: list[VoteMessage]


class ProposalRecord(BaseModel):
    proposal: ExecutionProposal
    status: ConsensusStatus = ConsensusStatus.PENDING
    result: ConsensusResult | None = None
    hitl_deadline: datetime | None = None
    hitl_actor: str | None = None
    hitl_note: str | None = None
    execution: dict[str, Any] | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)


class HitlDecision(Strict):
    approve: bool
    note: str = Field("", max_length=500)


class IncidentRequest(Strict):
    query: str = Field(min_length=3, max_length=2000)
    hint_action: ActionRequest | None = None
