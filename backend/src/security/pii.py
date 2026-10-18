"""PII / secret redaction: regex (+Luhn) and optional spaCy NER."""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("PRIVATE_KEY", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----")),
    ("GROQ_KEY", re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b")),
    ("AWS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("BEARER", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}")),
]
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_AADHAAR = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ \-]?){12,18}\d\b")
_PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s().\-]{8,17}\d(?!\w)")
_IPV4 = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")
_DATEISH = re.compile(r"^\d{4}-\d{2}-\d{2}")
NER_LABELS = {"PERSON"}


def luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass(slots=True)
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def redacted(self) -> bool:
        return bool(self.counts)


class PIIRedactor:
    def __init__(self, use_ner: bool = True) -> None:
        self._nlp: Any = None
        if use_ner:
            try:
                import spacy

                self._nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            except Exception:  # noqa: BLE001 - NER is best-effort, regex layer always on
                log.warning("spaCy model unavailable; PII NER disabled (regex layer active)")

    @property
    def ner_enabled(self) -> bool:
        return self._nlp is not None

    def redact(self, text: str) -> RedactionResult:
        counts: dict[str, int] = {}

        def bump(label: str) -> str:
            counts[label] = counts.get(label, 0) + 1
            return f"[REDACTED:{label}]"

        out = text
        for label, pat in _SECRET_PATTERNS:
            out = pat.sub(lambda _m, l=label: bump(l), out)
        out = _CARD.sub(lambda m: bump("CARD") if luhn_ok(m.group()) else m.group(), out)
        out = _SSN.sub(lambda _m: bump("SSN"), out)
        out = _AADHAAR.sub(lambda _m: bump("NATIONAL_ID"), out)
        out = _EMAIL.sub(lambda _m: bump("EMAIL"), out)

        def phone(m: re.Match[str]) -> str:
            s = m.group()
            digits = sum(c.isdigit() for c in s)
            if not 10 <= digits <= 15 or _IPV4.fullmatch(s) or _DATEISH.match(s):
                return s
            return bump("PHONE")

        out = _PHONE.sub(phone, out)

        if self._nlp is not None and out.strip():
            doc = self._nlp(out[:20000])
            spans = [(e.start_char, e.end_char) for e in doc.ents if e.label_ in NER_LABELS and "[REDACTED" not in e.text]
            for start, end in sorted(spans, reverse=True):
                out = out[:start] + bump("PERSON") + out[end:]
        return RedactionResult(out, counts)

    async def aredact(self, text: str) -> RedactionResult:
        return await asyncio.to_thread(self.redact, text)
