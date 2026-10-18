"""Prompt-injection detection and response validation."""
from __future__ import annotations

import re
import secrets
import unicodedata
from dataclasses import dataclass, field

from src.core import metrics
from src.security.pii import PIIRedactor

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff\u202a\u202b\u202c\u202d\u202e"), None)

_INJECTION_RULES: list[tuple[str, float, re.Pattern[str]]] = [
    ("override_instructions", 0.9, re.compile(r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|any|system|safety)\b.{0,30}\b(instruction|prompt|rule|guideline|polic)", re.S)),
    ("reveal_prompt", 0.85, re.compile(r"\b(reveal|show|print|repeat|leak|output|display)\b.{0,30}\b(system|hidden|initial|developer)\b.{0,15}\b(prompt|instruction|message)")),
    ("role_hijack", 0.8, re.compile(r"\b(you are now|from now on you are|act as|pretend to be)\b.{0,60}\b(dan|developer mode|jailbroken|unrestricted|no restrictions|without (any )?(restrictions|filters|limits))")),
    ("chat_template_tokens", 0.8, re.compile(r"(<\|im_start\|>|<\|im_end\|>|<\|system\|>|\[/?inst\]|<<sys>>|</?system>)")),
    ("disable_safety", 0.75, re.compile(r"\b(disable|bypass|turn off|circumvent)\b.{0,30}\b(guardrail|safety|filter|moderation|policy|policies|security)")),
    ("exfiltrate", 0.7, re.compile(r"\b(send|post|upload|exfiltrate|forward)\b.{0,40}\b(api[_ ]?key|secret|token|password|credential)")),
    ("jailbreak_word", 0.5, re.compile(r"\b(jailbreak|dan mode|do anything now)\b")),
    ("encoded_payload", 0.4, re.compile(r"\b(base64|rot13|hex)\b.{0,30}\b(decode|execute|run|eval)")),
]


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    return re.sub(r"\s+", " ", t).strip().lower()


@dataclass(slots=True)
class InjectionVerdict:
    score: float
    hits: list[str] = field(default_factory=list)
    blocked: bool = False


class InjectionDetector:
    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold

    def scan(self, text: str) -> InjectionVerdict:
        norm = normalise(text)
        remaining, hits = 1.0, []
        for name, weight, pat in _INJECTION_RULES:
            if pat.search(norm):
                hits.append(name)
                remaining *= 1.0 - weight
        score = round(1.0 - remaining, 3)
        blocked = score >= self.threshold
        if blocked:
            metrics.GUARDRAIL_EVENTS.labels("injection_blocked").inc()
        return InjectionVerdict(score, hits, blocked)


_DANGEROUS_OUTPUT = re.compile(
    r"(rm\s+-rf\s+/|:\(\)\s*\{\s*:\|:&\s*\};:|drop\s+table|truncate\s+table|curl[^\n|]*\|\s*(ba)?sh|chmod\s+-R\s+777\s+/|mkfs\.)", re.I
)


@dataclass(slots=True)
class ResponseVerdict:
    ok: bool
    text: str
    violations: list[str]


class ResponseEvaluator:
    """Validates LLM output: canary leak, secrets/PII, destructive commands, size."""

    def __init__(self, redactor: PIIRedactor, max_chars: int = 6000) -> None:
        self.canary = f"NOVAMESH-CANARY-{secrets.token_hex(8)}"
        self._redactor = redactor
        self._max = max_chars

    def evaluate(self, text: str) -> ResponseVerdict:
        violations: list[str] = []
        if self.canary in text:
            metrics.GUARDRAIL_EVENTS.labels("canary_leak").inc()
            return ResponseVerdict(False, "I can't share internal configuration.", ["system_prompt_leak"])
        if _DANGEROUS_OUTPUT.search(text):
            metrics.GUARDRAIL_EVENTS.labels("dangerous_command").inc()
            return ResponseVerdict(False, "I can't provide destructive commands. Use an approved admin action instead.", ["dangerous_command"])
        red = self._redactor.redact(text)
        if red.redacted:
            violations.extend(f"redacted_{k.lower()}" for k in red.counts)
            metrics.GUARDRAIL_EVENTS.labels("output_redaction").inc()
        out = red.text
        if len(out) > self._max:
            out = out[: self._max] + " …[truncated]"
            violations.append("truncated")
        return ResponseVerdict(True, out, violations)
