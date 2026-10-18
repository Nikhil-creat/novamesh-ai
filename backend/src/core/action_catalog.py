"""Concrete allow-listed actions. Handlers are bound to live services when available."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from src.core.actions import ActionError, ActionRegistry, ActionSpec, NoArgs, SourceArgs
from src.schemas import Severity

if TYPE_CHECKING:
    from src.rag.crag import HybridRetriever
    from src.vision.cnn_engine import VisionEngine


def build_action_registry(vision: VisionEngine | None = None, retriever: HybridRetriever | None = None) -> ActionRegistry:
    reg = ActionRegistry()

    def need(obj: Any, what: str) -> Any:
        if obj is None:
            raise ActionError(f"{what} is not available on this node")
        return obj

    async def pause(_: BaseModel) -> dict[str, Any]:
        need(vision, "vision engine").pause()
        return {"vision_paused": True}

    async def resume(_: BaseModel) -> dict[str, Any]:
        need(vision, "vision engine").resume()
        return {"vision_paused": False}

    async def rebuild(_: BaseModel) -> dict[str, Any]:
        r = need(retriever, "retriever")
        r.invalidate()
        return {"documents_indexed": await r.corpus_size()}

    async def quarantine(a: BaseModel) -> dict[str, Any]:
        assert isinstance(a, SourceArgs)
        need(vision, "vision engine").quarantine(a.source_id)
        return {"quarantined": a.source_id}

    reg.register(ActionSpec("rebuild_search_index", "Refresh the BM25/vector search index from Qdrant.", Severity.LOW, NoArgs, rebuild))
    reg.register(ActionSpec("resume_vision_ingest", "Resume paused CNN vision ingestion.", Severity.MEDIUM, NoArgs, resume))
    reg.register(ActionSpec("pause_vision_ingest", "Pause all CNN vision ingestion (containment).", Severity.HIGH, NoArgs, pause))
    reg.register(ActionSpec("quarantine_source", "Block a camera/IoT source from ingestion.", Severity.CRITICAL, SourceArgs, quarantine))
    return reg
