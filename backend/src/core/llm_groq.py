"""Groq LPU gateway: OpenAI-compatible client, 429 model fallback, structured JSON, usage tracking."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, ValidationError

from src.config import Settings
from src.core import metrics

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
Message = dict[str, str]
MAX_ATTEMPTS = 3


class LLMError(RuntimeError):
    """Raised when the LLM provider cannot satisfy a request."""


@dataclass(slots=True)
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    fallback_used: bool


@dataclass(slots=True)
class UsageTracker:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    fallbacks: int = 0
    by_model: dict[str, int] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def add(self, res: LLMResult) -> None:
        async with self._lock:
            self.requests += 1
            self.prompt_tokens += res.prompt_tokens
            self.completion_tokens += res.completion_tokens
            self.fallbacks += int(res.fallback_used)
            self.by_model[res.model] = self.by_model.get(res.model, 0) + res.prompt_tokens + res.completion_tokens

    def snapshot(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "fallbacks": self.fallbacks,
            "tokens_by_model": dict(self.by_model),
        }


def _strip_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    return re.sub(r"\s*```$", "", t).strip()


class GroqGateway:
    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._client = AsyncOpenAI(
            api_key=settings.groq_api_key.get_secret_value(),
            base_url=settings.groq_base_url,
            timeout=settings.groq_timeout_s,
            max_retries=0,
        )
        self._sem = asyncio.Semaphore(settings.groq_max_concurrency)
        self.usage = UsageTracker()

    @staticmethod
    def _retry_after(exc: RateLimitError, attempt: int) -> float:
        try:
            raw = exc.response.headers.get("retry-after")
            if raw:
                return min(max(float(raw), 0.5), 10.0)
        except (ValueError, AttributeError):
            pass
        return min(0.5 * 2**attempt, 8.0)

    async def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> LLMResult:
        primary = model or self._s.groq_primary_model
        chain = [primary] if primary == self._s.groq_fallback_model else [primary, self._s.groq_fallback_model]
        last_exc: Exception | None = None
        async with self._sem:
            for idx, name in enumerate(chain):
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    started = time.perf_counter()
                    kwargs: dict[str, Any] = {
                        "model": name,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    if json_mode:
                        kwargs["response_format"] = {"type": "json_object"}
                    try:
                        resp = await self._client.chat.completions.create(**kwargs)
                    except RateLimitError as exc:
                        last_exc = exc
                        metrics.LLM_REQUESTS.labels(name, "rate_limited").inc()
                        if idx + 1 < len(chain):
                            log.warning("429 on %s, falling back to %s", name, chain[idx + 1])
                            break
                        await asyncio.sleep(self._retry_after(exc, attempt))
                        continue
                    except (APIConnectionError, APITimeoutError) as exc:
                        last_exc = exc
                        metrics.LLM_REQUESTS.labels(name, "network_error").inc()
                        await asyncio.sleep(min(0.5 * 2**attempt, 8.0))
                        continue
                    except APIStatusError as exc:
                        last_exc = exc
                        metrics.LLM_REQUESTS.labels(name, f"http_{exc.status_code}").inc()
                        if exc.status_code >= 500:
                            await asyncio.sleep(min(0.5 * 2**attempt, 8.0))
                            continue
                        raise LLMError(f"Groq rejected request ({exc.status_code})") from exc

                    latency = time.perf_counter() - started
                    usage = resp.usage
                    result = LLMResult(
                        text=(resp.choices[0].message.content or ""),
                        model=name,
                        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                        latency_ms=latency * 1000,
                        fallback_used=idx > 0,
                    )
                    metrics.LLM_REQUESTS.labels(name, "ok").inc()
                    metrics.LLM_LATENCY.labels(name).observe(latency)
                    metrics.LLM_TOKENS.labels(name, "prompt").inc(result.prompt_tokens)
                    metrics.LLM_TOKENS.labels(name, "completion").inc(result.completion_tokens)
                    await self.usage.add(result)
                    return result
        raise LLMError("Groq request failed after retries and fallback") from last_exc

    async def chat_json(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> T:
        """Request JSON and validate against a Pydantic schema; one self-repair attempt."""
        msgs = list(messages)
        for _ in range(2):
            res = await self.chat(msgs, model=model, temperature=0.0, max_tokens=max_tokens, json_mode=True)
            try:
                return schema.model_validate_json(_strip_fences(res.text))
            except ValidationError as exc:
                msgs = [
                    *msgs,
                    {"role": "assistant", "content": res.text[:2000]},
                    {"role": "user", "content": f"That JSON failed validation: {str(exc)[:400]}. Return corrected JSON only."},
                ]
        raise LLMError("Structured output failed schema validation")

    async def aclose(self) -> None:
        await self._client.close()
