"""Async weighted consensus over Redis Streams with HITL gate bookkeeping and audited state machine."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from datetime import timedelta
from typing import Any

from src.config import STREAM_PROPOSALS, STREAM_VOTES, Settings
from src.consensus.roster import ROSTER, RosterEntry
from src.consensus.validator import ValidatorNode
from src.core import metrics
from src.core.actions import ActionError, ActionRegistry
from src.core.ledger import Ledger
from src.core.llm_groq import GroqGateway
from src.core.redis_bus import RedisBus
from src.schemas import (
    ConsensusResult, ConsensusStatus as S, ExecutionProposal, ProposalRecord, Vote, VoteMessage, utcnow,
)

log = logging.getLogger(__name__)
MAX_RECORDS = 500


class StateError(Exception):
    """Illegal state transition (e.g. approving a proposal that is not awaiting HITL)."""


class _Round:
    def __init__(self, expected: set[str]) -> None:
        self.expected = expected
        self.votes: dict[str, VoteMessage] = {}
        self.done = asyncio.Event()


class ConsensusEngine:
    def __init__(self, settings: Settings, bus: RedisBus, llm: GroqGateway, ledger: Ledger, actions: ActionRegistry) -> None:
        self._s, self._bus, self._llm, self._ledger, self._actions = settings, bus, llm, ledger, actions
        self.roster: dict[str, RosterEntry] = {e.validator_id: e for e in ROSTER[: settings.validator_count]}
        self.records: OrderedDict[str, ProposalRecord] = OrderedDict()
        self._rounds: dict[str, _Round] = {}
        self._lock = asyncio.Lock()
        self._tasks: list[asyncio.Task[None]] = []
        self.validators: list[ValidatorNode] = []

    # ------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        if self._s.embedded_validators:
            self.validators = [ValidatorNode(e, self._s, self._bus, self._llm, self._actions) for e in self.roster.values()]
            for v in self.validators:  # groups must exist before the first proposal is published
                await self._bus.ensure_group(STREAM_PROPOSALS, v.group)
        await self._bus.ensure_group(STREAM_VOTES, "coordinator")
        self._tasks = [asyncio.create_task(v.run(), name=v.group) for v in self.validators]
        self._tasks.append(asyncio.create_task(self._bus.consume(STREAM_VOTES, "coordinator", self._s.node_id, self._on_vote), name="vote-collector"))

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # ------------------------------------------------------------ helpers
    def _log(self, rec: ProposalRecord, status: S, **extra: Any) -> None:
        rec.status = status
        rec.history.append({"ts": utcnow().isoformat(), "status": status.value, **extra})

    async def _emit(self, rec: ProposalRecord, actor: str = "consensus") -> None:
        metrics.HITL_PENDING.set(sum(1 for r in self.records.values() if r.status is S.AWAITING_HITL))
        await self._ledger.safe_append(f"CONSENSUS_{rec.status.value}", actor, {
            "proposal_id": rec.proposal.proposal_id, "incident_id": rec.proposal.incident_id,
            "action": rec.proposal.action, "severity": rec.proposal.severity.value,
            "ratio": rec.result.approval_ratio if rec.result else None,
        })
        await self._bus.publish_telemetry("consensus_state", {
            "proposal_id": rec.proposal.proposal_id, "status": rec.status.value, "action": rec.proposal.action,
            "severity": rec.proposal.severity.value, "approval_ratio": rec.result.approval_ratio if rec.result else None,
        })

    async def _on_vote(self, payload: dict[str, Any]) -> None:
        vote = VoteMessage.model_validate(payload)
        rnd = self._rounds.get(vote.proposal_id)
        if rnd is None or vote.validator_id not in rnd.expected:
            return
        rnd.votes.setdefault(vote.validator_id, vote)
        if set(rnd.votes) >= rnd.expected:
            rnd.done.set()

    def _tally(self, proposal_id: str, votes: dict[str, VoteMessage]) -> ConsensusResult:
        total = sum(e.weight for e in self.roster.values())
        w = {Vote.AGREE: 0.0, Vote.DISAGREE: 0.0, Vote.ABSTAIN: 0.0}
        for vid, v in votes.items():
            w[v.vote] += self.roster[vid].weight  # coordinator-side weights are authoritative
        decisive = w[Vote.AGREE] + w[Vote.DISAGREE]
        participation = (decisive + w[Vote.ABSTAIN]) / total if total else 0.0
        ratio = w[Vote.AGREE] / decisive if decisive > 0 else 0.0
        quorum = participation >= 0.5
        return ConsensusResult(
            proposal_id=proposal_id, agree_weight=w[Vote.AGREE], disagree_weight=w[Vote.DISAGREE], abstain_weight=w[Vote.ABSTAIN],
            total_weight=total, participation=round(participation, 4), approval_ratio=round(ratio, 4), threshold=self._s.consensus_threshold,
            quorum_met=quorum, passed=quorum and decisive > 0 and ratio > self._s.consensus_threshold, votes=list(votes.values()),
        )

    # ------------------------------------------------------------ rounds
    async def run_round(self, proposal: ExecutionProposal) -> ProposalRecord:
        started = time.perf_counter()
        rec = ProposalRecord(proposal=proposal)
        self._log(rec, S.PENDING)
        self.records[proposal.proposal_id] = rec
        while len(self.records) > MAX_RECORDS:
            self.records.popitem(last=False)
        rnd = _Round(set(self.roster))
        self._rounds[proposal.proposal_id] = rnd  # register BEFORE broadcasting to avoid vote races
        await self._emit(rec)
        try:
            await self._bus.publish(STREAM_PROPOSALS, proposal)
            try:
                await asyncio.wait_for(rnd.done.wait(), timeout=self._s.consensus_timeout_s)
            except TimeoutError:
                log.warning("consensus timeout for %s with %d/%d votes", proposal.proposal_id, len(rnd.votes), len(rnd.expected))
        finally:
            self._rounds.pop(proposal.proposal_id, None)
        result = self._tally(proposal.proposal_id, rnd.votes)
        rec.result = result
        async with self._lock:
            if not result.passed:
                self._log(rec, S.REJECTED, ratio=result.approval_ratio, quorum=result.quorum_met)
                metrics.CONSENSUS_ROUNDS.labels("rejected").inc()
            elif proposal.requires_hitl:
                rec.hitl_deadline = utcnow() + timedelta(seconds=self._s.hitl_timeout_s)
                self._log(rec, S.AWAITING_HITL, deadline=rec.hitl_deadline.isoformat())
                metrics.CONSENSUS_ROUNDS.labels("awaiting_hitl").inc()
            else:
                self._log(rec, S.PASSED, ratio=result.approval_ratio)
                metrics.CONSENSUS_ROUNDS.labels("passed").inc()
        metrics.CONSENSUS_DURATION.observe(time.perf_counter() - started)
        await self._emit(rec)
        return rec

    # ------------------------------------------------------------ HITL
    async def record_hitl(self, proposal_id: str, approve: bool, actor: str, note: str = "", expired: bool = False) -> ProposalRecord:
        async with self._lock:
            rec = self.records.get(proposal_id)
            if rec is None:
                raise KeyError(proposal_id)
            if rec.status is not S.AWAITING_HITL:
                raise StateError(f"proposal is {rec.status.value}, not awaiting human approval")
            rec.hitl_actor, rec.hitl_note = actor, note
            self._log(rec, S.EXPIRED if expired else (S.APPROVED if approve else S.HITL_DENIED), actor=actor, note=note)
        await self._emit(rec, actor=actor)
        return rec

    def expired_hitl(self) -> list[str]:
        now = utcnow()
        return [pid for pid, r in self.records.items() if r.status is S.AWAITING_HITL and r.hitl_deadline and r.hitl_deadline < now]

    # ------------------------------------------------------------ execution
    async def execute(self, proposal_id: str, actor: str) -> ProposalRecord:
        async with self._lock:
            rec = self.records.get(proposal_id)
            if rec is None:
                raise KeyError(proposal_id)
            if rec.status not in (S.PASSED, S.APPROVED):
                raise StateError(f"cannot execute proposal in state {rec.status.value}")
            self._log(rec, S.EXECUTING, actor=actor)
        try:
            rec.execution = {"ok": True, "result": await self._actions.execute(rec.proposal.action, rec.proposal.args)}
            self._log(rec, S.EXECUTED, actor=actor)
        except ActionError as exc:
            rec.execution = {"ok": False, "error": str(exc)}
            self._log(rec, S.FAILED, actor=actor, error=str(exc))
        await self._emit(rec, actor=actor)
        return rec

    def get(self, proposal_id: str) -> ProposalRecord | None:
        return self.records.get(proposal_id)

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for r in self.records.values():
            counts[r.status.value] = counts.get(r.status.value, 0) + 1
        return {"proposals": len(self.records), "by_status": counts, "validators": list(self.roster)}
