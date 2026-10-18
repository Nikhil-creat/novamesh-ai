"""Deterministic safety policies enforced by the Compliance-Agent and re-checked by every validator."""
from __future__ import annotations

from typing import Any

from src.core.actions import ActionError, ActionRegistry
from src.schemas import Severity

POLICIES = {
    "P-001": "Only registered, allow-listed actions may be proposed.",
    "P-002": "Action arguments must validate against the action schema.",
    "P-003": "quarantine_source requires at least two independent evidence items.",
    "P-004": "pause_vision_ingest requires an assessed threat level of HIGH or above.",
    "P-005": "resume_vision_ingest is forbidden while the incident threat is CRITICAL.",
}


def evaluate_action(
    registry: ActionRegistry, action: str, args: dict[str, Any], *, evidence_count: int, threat: Severity
) -> list[str]:
    violations: list[str] = []
    try:
        registry.validate(action, args)
    except ActionError as exc:
        code = "P-001" if "unknown action" in str(exc) else "P-002"
        return [f"{code}: {exc}"]
    if action == "quarantine_source" and evidence_count < 2:
        violations.append(f"P-003: {POLICIES['P-003']}")
    if action == "pause_vision_ingest" and threat.rank < Severity.HIGH.rank:
        violations.append(f"P-004: {POLICIES['P-004']}")
    if action == "resume_vision_ingest" and threat is Severity.CRITICAL:
        violations.append(f"P-005: {POLICIES['P-005']}")
    return violations
