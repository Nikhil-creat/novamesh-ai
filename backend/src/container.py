"""Dependency container: constructs, starts and stops every subsystem in order."""
from __future__ import annotations

import logging
from typing import Any

from src.agents.graph import MeshOrchestrator
from src.assistant.service import AssistantService
from src.config import STREAM_VISION, Settings
from src.consensus.engine import ConsensusEngine
from src.core.action_catalog import build_action_registry
from src.core.ledger import Ledger
from src.core.llm_groq import GroqGateway
from src.core.redis_bus import RedisBus
from src.rag.crag import CRAGEngine, HybridRetriever, RetrievedDoc
from src.security.auth import AuthService
from src.security.crypto import PayloadCipher
from src.security.guardrails import InjectionDetector, ResponseEvaluator
from src.security.pii import PIIRedactor
from src.vision.cnn_engine import VisionEngine

log = logging.getLogger(__name__)


class Container:
    def __init__(self, settings: Settings, redactor: PIIRedactor, detector: InjectionDetector) -> None:
        self.settings, self.redactor, self.detector = settings, redactor, detector
        self.auth = AuthService(settings)
        self.evaluator = ResponseEvaluator(redactor)
        self.bus = RedisBus(settings, PayloadCipher(settings.aes_key))
        self.ledger = Ledger(settings.postgres_dsn)
        self.llm = GroqGateway(settings)
        self.retriever = HybridRetriever(settings)
        self.vision = VisionEngine(settings, self.bus)
        self.actions = build_action_registry(self.vision, self.retriever)
        self.crag = CRAGEngine(settings, self.retriever, self.llm, [self._log_tool, self._vision_tool])
        self.consensus = ConsensusEngine(settings, self.bus, self.llm, self.ledger, self.actions)
        self.mesh = MeshOrchestrator(settings, self.bus, self.llm, self.crag, self.ledger, self.consensus, self.actions)
        self.assistant = AssistantService(settings, self.llm, self.crag, self.retriever, self.ledger, self.bus, self.vision, self.consensus, self.actions, self.evaluator, detector)

    async def _log_tool(self, query: str) -> list[RetrievedDoc]:
        rows = await self.ledger.search(query, 5)
        return [RetrievedDoc(f"ledger-{r['id']}", f"[{r['ts'].isoformat()}] {r['event_type']} by {r['actor']}: {r['payload'][:600]}", 0.5, "system_logs") for r in rows]

    async def _vision_tool(self, query: str) -> list[RetrievedDoc]:
        events = await self.bus.tail(STREAM_VISION, 5)
        return [RetrievedDoc(f"vision-{e['event_id']}", f"Vision anomaly source={e['source_id']} score={e['anomaly_score']} at {e['captured_at']}", 0.5, "vision_events") for e in events]

    async def start(self) -> None:
        await self.bus.connect()
        await self.ledger.connect()
        await self.retriever.start()
        await self.vision.start()
        await self.consensus.start()
        await self.mesh.start()
        await self.ledger.safe_append("SYSTEM_START", self.settings.node_id, {"ner": self.redactor.ner_enabled})
        log.info("NovaMesh core started")

    async def stop(self) -> None:
        for step in (self.mesh.stop, self.consensus.stop, self.vision.stop, self.retriever.close, self.llm.aclose, self.ledger.close, self.bus.close):
            try:
                await step()
            except Exception:  # noqa: BLE001
                log.exception("shutdown step failed")

    def snapshot(self) -> dict[str, Any]:
        return {"vision": self.vision.stats(), "consensus": self.consensus.summary(), "llm": self.llm.usage.snapshot()}
