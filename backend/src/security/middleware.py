"""Pure-ASGI security middleware: headers, body limits, PII redaction + prompt-injection gate, HTTP metrics."""
from __future__ import annotations

import json
import time
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.core import metrics
from src.security.guardrails import InjectionDetector
from src.security.pii import PIIRedactor

_SEC_HEADERS = {
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "cache-control": "no-store",
    "strict-transport-security": "max-age=63072000; includeSubDomains",
    "content-security-policy": "default-src 'none'; frame-ancestors 'none'",
}


class SecurityMiddleware:
    def __init__(
        self, app: ASGIApp, *, redactor: PIIRedactor, detector: InjectionDetector,
        max_json_bytes: int, max_upload_bytes: int, protected_prefixes: tuple[str, ...],
    ) -> None:
        self.app = app
        self.redactor, self.detector = redactor, detector
        self.max_json, self.max_upload = max_json_bytes, max_upload_bytes
        self.prefixes = protected_prefixes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status_holder = {"code": 500}
        method, path = scope["method"], scope["path"]

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["code"] = message["status"]
                headers = MutableHeaders(scope=message)
                for k, v in _SEC_HEADERS.items():
                    headers.setdefault(k, v)
            await send(message)

        try:
            headers = Headers(scope=scope)
            ctype = headers.get("content-type", "")
            clen = int(headers.get("content-length", "0") or 0)
            is_json = ctype.startswith("application/json")
            limit = self.max_json if is_json else self.max_upload
            if clen > limit:
                await JSONResponse({"detail": "payload too large"}, status_code=413)(scope, receive, send_wrapper)
                return

            if method in {"POST", "PUT", "PATCH"} and is_json and path.startswith(self.prefixes):
                body = b""
                more = True
                while more:
                    msg = await receive()
                    if msg["type"] == "http.disconnect":
                        return
                    body += msg.get("body", b"")
                    more = msg.get("more_body", False)
                    if len(body) > limit:
                        await JSONResponse({"detail": "payload too large"}, status_code=413)(scope, receive, send_wrapper)
                        return
                try:
                    parsed: Any = json.loads(body) if body else None
                except json.JSONDecodeError:
                    parsed = None
                if parsed is not None:
                    blocked, cleaned, counts = await self._sanitise(parsed)
                    if blocked:
                        await JSONResponse(
                            {"detail": "request blocked by prompt-injection guardrail", "code": "PROMPT_INJECTION_DETECTED", "signals": blocked},
                            status_code=400,
                        )(scope, receive, send_wrapper)
                        return
                    if counts:
                        metrics.GUARDRAIL_EVENTS.labels("input_redaction").inc()
                    body = json.dumps(cleaned).encode()
                    scope = dict(scope)
                    scope["headers"] = [(k, v) for k, v in scope["headers"] if k != b"content-length"] + [(b"content-length", str(len(body)).encode())]
                    scope.setdefault("state", {})["pii_redactions"] = counts

                sent = False

                async def replay() -> Message:
                    nonlocal sent
                    if not sent:
                        sent = True
                        return {"type": "http.request", "body": body, "more_body": False}
                    return await receive()

                await self.app(scope, replay, send_wrapper)
            else:
                await self.app(scope, receive, send_wrapper)
        finally:
            short = "/".join(path.split("/")[:4])
            metrics.HTTP_REQUESTS.labels(method, short, str(status_holder["code"])).inc()
            metrics.HTTP_LATENCY.labels(method).observe(time.perf_counter() - started)

    async def _sanitise(self, node: Any) -> tuple[list[str], Any, dict[str, int]]:
        blocked: list[str] = []
        counts: dict[str, int] = {}

        async def walk(n: Any) -> Any:
            if isinstance(n, str):
                verdict = self.detector.scan(n)
                if verdict.blocked:
                    blocked.extend(verdict.hits)
                res = await self.redactor.aredact(n)
                for k, v in res.counts.items():
                    counts[k] = counts.get(k, 0) + v
                return res.text
            if isinstance(n, list):
                return [await walk(i) for i in n]
            if isinstance(n, dict):
                return {k: await walk(v) for k, v in n.items()}
            return n

        cleaned = await walk(node)
        return sorted(set(blocked)), cleaned, counts
