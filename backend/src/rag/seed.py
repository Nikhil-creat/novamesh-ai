"""Seed enterprise knowledge so CRAG is useful on first boot."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class KnowledgeDoc:
    doc_id: str
    text: str
    source: str = "kb"
    metadata: dict[str, Any] = field(default_factory=dict)


SEED_DOCS: list[KnowledgeDoc] = [
    KnowledgeDoc("kb-e4012", "Error E-4012 REDIS_STREAM_BACKLOG: consumer group lag exceeds 10k entries. Cause: slow consumers or a crashed validator. Fix: check consumer liveness, inspect the :dlq stream, scale consumers, XTRIM only after confirming no pending entries.", "runbook"),
    KnowledgeDoc("kb-e5001", "Error E-5001 PG_POOL_EXHAUSTED: PostgreSQL connection pool exhausted in the audit ledger. Cause: long transactions or advisory lock contention. Fix: inspect pg_stat_activity, raise pool max_size, verify no stuck transaction holds pg_advisory_xact_lock(7771).", "runbook"),
    KnowledgeDoc("kb-e7203", "Error E-7203 CNN_LATENCY_SPIKE: vision inference latency above 500 ms. Cause: CPU throttling, queue saturation, or oversized frames. Fix: check container CPU limits, reduce VISION_INPUT_SIZE, pause the noisiest camera, or add GPU nodes.", "runbook"),
    KnowledgeDoc("kb-e9100", "Error E-9100 CONSENSUS_QUORUM_FAILURE: fewer than 50 percent of validator weight voted before timeout. Cause: validators offline or Groq rate limiting (HTTP 429). Fix: restart validator nodes, verify GROQ_API_KEY quota, confirm fallback model llama-3.1-8b-instant is reachable.", "runbook"),
    KnowledgeDoc("kb-vis-anom", "Runbook VISION-ANOMALY: when a camera reports an anomaly score above 0.8, review bounding boxes and embedding drift. A sustained anomaly from one source for more than 3 frames suggests tampering or scene change. Containment options: pause_vision_ingest (HIGH), quarantine_source (CRITICAL, requires human approval).", "runbook"),
    KnowledgeDoc("kb-pol-hitl", "Policy SEC-POL-07: any CRITICAL action requires consensus above 66 percent weighted approval AND explicit human-in-the-loop administrator approval before execution. Approval expires after the configured HITL timeout and the proposal is then rejected.", "policy"),
    KnowledgeDoc("kb-e3301", "Error E-3301 QDRANT_UNAVAILABLE: vector search backend unreachable. CRAG falls back to system log search. Fix: check the qdrant-db container healthcheck and the qdrant volume, then run the rebuild_search_index admin action.", "runbook"),
]
