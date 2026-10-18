"""Prometheus metric registry (single process)."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

LLM_REQUESTS = Counter("novamesh_llm_requests_total", "Groq requests", ["model", "outcome"])
LLM_TOKENS = Counter("novamesh_llm_tokens_total", "Groq tokens", ["model", "kind"])
LLM_LATENCY = Histogram("novamesh_llm_latency_seconds", "Groq latency", ["model"])
VISION_LATENCY = Histogram(
    "novamesh_vision_latency_ms", "CNN end-to-end latency (ms)",
    buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
)
VISION_FRAMES = Counter("novamesh_vision_frames_total", "Frames processed", ["outcome"])
VISION_SCORE = Histogram("novamesh_vision_anomaly_score", "Anomaly score", buckets=(0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 1.0))
VISION_QUEUE = Gauge("novamesh_vision_queue_depth", "Frame queue depth")
CRAG_REQUESTS = Counter("novamesh_crag_requests_total", "CRAG runs", ["verdict"])
CRAG_RELEVANCE = Histogram("novamesh_crag_relevance", "Top relevance", buckets=(0.1, 0.25, 0.5, 0.75, 0.9, 1.0))
CONSENSUS_ROUNDS = Counter("novamesh_consensus_rounds_total", "Consensus rounds", ["outcome"])
CONSENSUS_DURATION = Histogram("novamesh_consensus_duration_seconds", "Round duration")
HITL_PENDING = Gauge("novamesh_hitl_pending", "Proposals awaiting human approval")
GUARDRAIL_EVENTS = Counter("novamesh_guardrail_events_total", "Guardrail hits", ["kind"])
HTTP_REQUESTS = Counter("novamesh_http_requests_total", "HTTP requests", ["method", "path", "status"])
HTTP_LATENCY = Histogram("novamesh_http_latency_seconds", "HTTP latency", ["method"])
AGENT_NODE_DURATION = Histogram("novamesh_agent_node_seconds", "LangGraph node duration", ["node"])
DLQ_MESSAGES = Counter("novamesh_dlq_messages_total", "Messages sent to DLQ", ["stream"])
