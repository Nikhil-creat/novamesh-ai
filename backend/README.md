# NovaMesh AI — backend

> Designed and developed by **NIKHIL CHARY SRIRAMOJU**. v2 adds `/api/v1/about`, AI incident reports (`/mesh/incidents/{id}/report`) and an explainable posture score (`/insights/posture`).

Enterprise autonomous agent mesh + real-time CNN vision security ops + personal AI assistant.
Python 3.12 · FastAPI/uvloop · Groq (llama-3.3-70b-versatile → llama-3.1-8b-instant on 429) · PyTorch ResNet-50 · LangGraph · Qdrant hybrid CRAG · Redis Streams · PostgreSQL hash-chained ledger · Prometheus.

## Folder tree
```
novamesh-ai/
├── requirements.txt  .env.example  .dockerignore
├── docker/  Dockerfile  docker-compose.yml  prometheus.yml
└── src/
    ├── config.py  schemas.py  container.py
    ├── core/       llm_groq.py  redis_bus.py  ledger.py  metrics.py  actions.py  action_catalog.py
    ├── vision/     cnn_engine.py
    ├── rag/        crag.py  seed.py
    ├── assistant/  service.py
    ├── agents/     graph.py  policies.py
    ├── consensus/  engine.py  validator.py  roster.py  validator_node.py
    ├── security/   pii.py  guardrails.py  auth.py  crypto.py  middleware.py
    └── api/        app.py
```

## Run
```bash
cp .env.example .env      # fill GROQ_API_KEY, JWT_SECRET, AES_KEY_B64, passwords
docker compose --env-file .env -f docker/docker-compose.yml up --build -d
curl -s localhost:8000/readyz
TOKEN=$(curl -s localhost:8000/api/v1/auth/token -H 'content-type: application/json' \
  -d '{"username":"admin","password":"<BOOTSTRAP_ADMIN_PASSWORD>"}' | jq -r .access_token)
curl -s localhost:8000/api/v1/assistant/chat -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"message":"What does error E-7203 mean and is the system healthy?"}'
curl -s localhost:8000/api/v1/vision/ingest -H "authorization: Bearer $TOKEN" -F source_id=cam-1 -F file=@frame.jpg
curl -s localhost:8000/api/v1/mesh/incidents -H "authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"query":"Camera cam-1 shows sustained anomalies","hint_action":{"name":"quarantine_source","args":{"source_id":"cam-1"}}}'
curl -s localhost:8000/api/v1/consensus/proposal/<id>/hitl-approve -X POST -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"approve":true,"note":"verified on site"}'
```
WebSocket: `ws://localhost:8000/ws/v1/telemetry?token=$TOKEN` (events: `cnn_latency`, `graph_update`, `consensus_state`, plus a 500 ms `snapshot`).
Docs at `/docs`, metrics at `/metrics`, Prometheus on `127.0.0.1:9090`.

## Design notes
- **Consensus**: each validator (separate Redis consumer group) votes AGREE/DISAGREE/ABSTAIN with roster weights (coordinator-side authoritative). Quorum = ≥50% of total weight participates; pass = AGREE/(AGREE+DISAGREE) **> 0.66**. CRITICAL passes pause at a LangGraph `interrupt()` until `hitl-approve`; unanswered approvals expire (`HITL_TIMEOUT_S`).
- **Severity** = max(action intrinsic, log severity, vision threat). Only `quarantine_source` is intrinsically CRITICAL. The assistant can run non-CRITICAL admin commands (ADMIN role) but never CRITICAL ones.
- **CRAG**: dense (Qdrant) + BM25 → RRF → cross-encoder rerank; relevance = sigmoid(logit) vs 0.75. Below threshold → query rewrite → re-retrieve → fallback tools (audit-ledger logs, vision events).
- **Vision**: 2048-d ResNet-50 pooled features → fixed seeded random projection → L2-normalised 512-d. Anomaly score = online diagonal-Gaussian drift vs. a baseline learned from normal frames (first `VISION_WARMUP_FRAMES` score 0). Boxes are activation-saliency regions of interest, not a trained detector; swap in a detection head if you need class labels.
- **Security**: ASGI middleware redacts PII (regex+Luhn, optional spaCy NER) and blocks prompt injection on `/assistant` and `/mesh` JSON bodies; responses go through a canary/secret/PII/destructive-command evaluator; Redis stream payloads are AES-256-GCM encrypted (stream name as AAD); JWT RBAC viewer<operator<admin.
- **Ledger**: every state change is appended with SHA-256 chaining; `GET /api/v1/ledger/verify`.

## Known limits (be aware before production)
- Single worker by design: consensus records, LangGraph checkpoints (`MemorySaver`) and the vision baseline are in-process. A restart loses in-flight HITL proposals (the ledger keeps the audit trail). Swap in a Postgres checkpointer to persist.
- Incident dispatch from the vision stream is at-most-once after the ack. Telemetry Pub/Sub is not encrypted (it carries no embeddings).
- Not executed in my build sandbox (no network): I verified syntax and the PII logic only. Expect to fix small integration issues on first `docker compose up`; send me the traceback and I'll patch.
