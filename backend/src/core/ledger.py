"""PostgreSQL append-only, hash-chained audit ledger."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

import asyncpg

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_ledger (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL,
    event_type TEXT NOT NULL,
    actor TEXT NOT NULL,
    payload TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS idx_audit_event_ts ON audit_ledger (event_type, ts DESC);
"""
GENESIS = "GENESIS"


def _digest(prev: str, ts: datetime, event_type: str, actor: str, payload: str) -> str:
    return hashlib.sha256(f"{prev}|{ts.isoformat()}|{event_type}|{actor}|{payload}".encode()).hexdigest()


class Ledger:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=10, command_timeout=15)
        async with self.pool.acquire() as conn:
            await conn.execute(SCHEMA)

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    def _p(self) -> asyncpg.Pool:
        if self.pool is None:
            raise RuntimeError("ledger not connected")
        return self.pool

    async def ping(self) -> bool:
        async with self._p().acquire() as conn:
            return (await conn.fetchval("SELECT 1")) == 1

    async def append(self, event_type: str, actor: str, payload: dict[str, Any]) -> int:
        body = json.dumps(payload, sort_keys=True, default=str)
        async with self._p().acquire() as conn, conn.transaction():
            await conn.execute("SELECT pg_advisory_xact_lock(7771)")
            prev = await conn.fetchval("SELECT hash FROM audit_ledger ORDER BY id DESC LIMIT 1") or GENESIS
            ts = datetime.now(timezone.utc)
            return await conn.fetchval(
                "INSERT INTO audit_ledger (ts, event_type, actor, payload, prev_hash, hash) "
                "VALUES ($1,$2,$3,$4,$5,$6) RETURNING id",
                ts, event_type, actor, body, prev, _digest(prev, ts, event_type, actor, body),
            )

    async def safe_append(self, event_type: str, actor: str, payload: dict[str, Any]) -> None:
        try:
            await self.append(event_type, actor, payload)
        except Exception:  # noqa: BLE001 - audit failures are logged, never crash the caller
            log.exception("ledger append failed (%s)", event_type)

    async def recent(self, limit: int = 20, event_type: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT id, ts, event_type, actor, payload FROM audit_ledger"
        args: list[Any] = []
        if event_type:
            q += " WHERE event_type = $1"
            args.append(event_type)
        q += f" ORDER BY id DESC LIMIT {int(min(max(limit, 1), 200))}"
        async with self._p().acquire() as conn:
            return [dict(r) for r in await conn.fetch(q, *args)]

    async def search(self, text: str, limit: int = 10) -> list[dict[str, Any]]:
        terms = [t for t in re.findall(r"[A-Za-z0-9_\-.]{3,}", text)][:5]
        if not terms:
            return await self.recent(limit)
        clauses = " OR ".join(f"payload ILIKE ${i + 1} OR event_type ILIKE ${i + 1}" for i in range(len(terms)))
        q = f"SELECT id, ts, event_type, actor, payload FROM audit_ledger WHERE {clauses} ORDER BY id DESC LIMIT {int(min(max(limit, 1), 50))}"
        async with self._p().acquire() as conn:
            return [dict(r) for r in await conn.fetch(q, *[f"%{t}%" for t in terms])]

    async def count_since(self, event_type: str, minutes: int) -> int:
        async with self._p().acquire() as conn:
            return int(await conn.fetchval(
                "SELECT count(*) FROM audit_ledger WHERE event_type=$1 AND ts > now() - make_interval(mins => $2)",
                event_type, minutes,
            ))

    async def verify_chain(self, limit: int = 5000) -> dict[str, Any]:
        async with self._p().acquire() as conn:
            rows = await conn.fetch("SELECT id, ts, event_type, actor, payload, prev_hash, hash FROM audit_ledger ORDER BY id DESC LIMIT $1", limit)
        rows = list(reversed(rows))
        for i, r in enumerate(rows):
            if _digest(r["prev_hash"], r["ts"], r["event_type"], r["actor"], r["payload"]) != r["hash"]:
                return {"valid": False, "broken_at": r["id"], "checked": i + 1}
            if i > 0 and r["prev_hash"] != rows[i - 1]["hash"]:
                return {"valid": False, "broken_at": r["id"], "checked": i + 1}
        return {"valid": True, "checked": len(rows)}
