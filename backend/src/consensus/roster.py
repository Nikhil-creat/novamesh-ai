"""Static validator roster shared by the coordinator and validator nodes (authoritative weights)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RosterEntry:
    validator_id: str
    persona: str
    weight: float


ROSTER: tuple[RosterEntry, ...] = (
    RosterEntry("val-sec", "security engineer focused on attack surface, containment and blast radius", 1.5),
    RosterEntry("val-sre", "site reliability engineer focused on availability and rollback safety", 1.25),
    RosterEntry("val-cmp", "compliance officer focused on policy, auditability and least privilege", 1.25),
    RosterEntry("val-ops", "operations lead focused on business impact and operator workload", 1.0),
    RosterEntry("val-vis", "computer-vision specialist focused on evidence quality and false positives", 1.0),
)
