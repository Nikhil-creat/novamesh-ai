"""Redis Streams bus: AES-GCM encrypted payloads, consumer groups, retries, DLQ; Pub/Sub telemetry."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import redis.asyncio as aioredis
from pydantic import BaseModel
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

from src.config import TELEMETRY_CHANNEL, Settings
from src.core import metrics
from src.schemas import utcnow
from src.security.crypto import CryptoError, PayloadCipher

log = logging.getLogger(__name__)
Handler = Callable[[dict[str, Any]], Awaitable[None]]


class RedisBus:
    def __init__(self, settings: Settings, cipher: PayloadCipher) -> None:
        self._s = settings
        self._cipher = cipher
        self.redis: aioredis.Redis = aioredis.from_url(settings.redis_url, decode_responses=True, health_check_interval=15)

    async def connect(self) -> None:
        await self.redis.ping()

    async def close(self) -> None:
        await self.redis.aclose()

    # -------------------------------------------------------------- streams
    def _encode(self, stream: str, payload: BaseModel | dict[str, Any]) -> str:
        data = payload.model_dump_json() if isinstance(payload, BaseModel) else json.dumps(payload, default=str)
        return self._cipher.encrypt(data.encode(), stream.encode())

    def _decode(self, stream: str, token: str) -> dict[str, Any]:
        return json.loads(self._cipher.decrypt(token, stream.encode()))

    async def publish(self, stream: str, payload: BaseModel | dict[str, Any], maxlen: int = 10_000) -> str:
        return await self.redis.xadd(stream, {"d": self._encode(stream, payload)}, maxlen=maxlen, approximate=True)

    async def ensure_group(self, stream: str, group: str) -> None:
        try:
            await self.redis.xgroup_create(stream, group, id="$", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def tail(self, stream: str, count: int = 10) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for _id, fields in await self.redis.xrevrange(stream, count=count):
            try:
                out.append({"stream_id": _id, **self._decode(stream, fields["d"])})
            except (CryptoError, KeyError, json.JSONDecodeError):
                continue
        return out

    async def _dead_letter(self, stream: str, msg_id: str, fields: dict[str, str], err: str) -> None:
        await self.redis.xadd(f"{stream}:dlq", {"d": fields.get("d", ""), "src_id": msg_id, "err": err[:500]}, maxlen=5000, approximate=True)
        metrics.DLQ_MESSAGES.labels(stream).inc()
        log.error("DLQ %s id=%s err=%s", stream, msg_id, err)

    async def consume(
        self,
        stream: str,
        group: str,
        consumer: str,
        handler: Handler,
        *,
        batch: int = 10,
        block_ms: int = 1000,
        max_attempts: int = 3,
    ) -> None:
        """Run forever: handle pending (retry) entries first, then new ones. Poison messages go to <stream>:dlq."""
        await self.ensure_group(stream, group)
        attempts: dict[str, int] = {}
        backlog = True
        while True:
            try:
                resp = await self.redis.xreadgroup(
                    group, consumer, {stream: "0" if backlog else ">"}, count=batch, block=None if backlog else block_ms
                )
                entries = resp[0][1] if resp else []
                if backlog and not entries:
                    backlog = False
                    continue
                failed = False
                for msg_id, fields in entries:
                    try:
                        payload = self._decode(stream, fields["d"])
                    except (CryptoError, KeyError, json.JSONDecodeError) as exc:
                        await self._dead_letter(stream, msg_id, fields, f"undecodable: {exc}")
                        await self.redis.xack(stream, group, msg_id)
                        continue
                    try:
                        await handler(payload)
                        await self.redis.xack(stream, group, msg_id)
                        attempts.pop(msg_id, None)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # noqa: BLE001 - handler isolation
                        attempts[msg_id] = attempts.get(msg_id, 0) + 1
                        if attempts[msg_id] >= max_attempts:
                            await self._dead_letter(stream, msg_id, fields, repr(exc))
                            await self.redis.xack(stream, group, msg_id)
                            attempts.pop(msg_id, None)
                        else:
                            failed = True
                            log.warning("handler failed (%s) attempt %d: %r", stream, attempts[msg_id], exc)
                if failed:
                    backlog = True
                    await asyncio.sleep(0.5)
                elif backlog:
                    backlog = False
            except asyncio.CancelledError:
                raise
            except ResponseError as exc:
                if "NOGROUP" in str(exc):
                    await self.ensure_group(stream, group)
                else:
                    log.exception("stream error %s", stream)
                    await asyncio.sleep(1)
            except (RedisConnectionError, OSError):
                log.warning("redis unavailable for %s, retrying", stream)
                await asyncio.sleep(1)

    # ------------------------------------------------------------ telemetry
    async def publish_telemetry(self, kind: str, data: dict[str, Any]) -> None:
        try:
            await self.redis.publish(TELEMETRY_CHANNEL, json.dumps({"kind": kind, "ts": utcnow().isoformat(), "data": data}, default=str))
        except Exception:  # noqa: BLE001 - telemetry must never break business flow
            log.debug("telemetry publish failed", exc_info=True)
