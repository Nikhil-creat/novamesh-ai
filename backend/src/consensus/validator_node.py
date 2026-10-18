"""Standalone validator process: `python -m src.consensus.validator_node` (set EMBEDDED_VALIDATORS=false on core)."""
from __future__ import annotations

import asyncio
import logging

from src.config import get_settings
from src.consensus.roster import ROSTER
from src.consensus.validator import ValidatorNode
from src.core.action_catalog import build_action_registry
from src.core.llm_groq import GroqGateway
from src.core.redis_bus import RedisBus
from src.security.crypto import PayloadCipher


async def main() -> None:
    s = get_settings()
    logging.basicConfig(level=s.log_level)
    bus = RedisBus(s, PayloadCipher(s.aes_key))
    await bus.connect()
    llm = GroqGateway(s)
    registry = build_action_registry()  # schema/policy validation only; handlers unused on validators
    nodes = [ValidatorNode(e, s, bus, llm, registry) for e in ROSTER[: s.validator_count]]
    await asyncio.gather(*(n.run() for n in nodes))


if __name__ == "__main__":
    asyncio.run(main())
