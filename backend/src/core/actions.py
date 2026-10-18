"""Allow-listed administrative action registry (shared by assistant, agents and consensus executor)."""
from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.schemas import Severity

log = logging.getLogger(__name__)


class ActionError(Exception):
    pass


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_id: str = Field(pattern=r"^[A-Za-z0-9_.:-]{1,64}$")


@dataclass(frozen=True, slots=True)
class ActionSpec:
    name: str
    description: str
    severity: Severity
    args_model: type[BaseModel]
    handler: Callable[[BaseModel], Awaitable[dict[str, Any]]]


class ActionRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ActionSpec] = {}

    def register(self, spec: ActionSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> ActionSpec | None:
        return self._specs.get(name)

    def catalog(self) -> list[dict[str, Any]]:
        return [
            {"name": s.name, "description": s.description, "severity": s.severity.value, "args": s.args_model.model_json_schema().get("properties", {})}
            for s in self._specs.values()
        ]

    def validate(self, name: str, args: dict[str, Any]) -> tuple[ActionSpec, BaseModel]:
        spec = self._specs.get(name)
        if spec is None:
            raise ActionError(f"unknown action '{name}'")
        try:
            return spec, spec.args_model.model_validate(args)
        except ValidationError as exc:
            raise ActionError(f"invalid args for '{name}': {exc.errors()[0]['msg']}") from exc

    async def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        spec, parsed = self.validate(name, args)
        try:
            return await spec.handler(parsed)
        except ActionError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("action %s failed", name)
            raise ActionError(f"action '{name}' failed: {exc}") from exc
