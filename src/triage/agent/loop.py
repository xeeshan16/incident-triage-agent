"""The triage agent loop: a manual Messages-API tool-use loop with explicit guardrails.

Guardrails (each has a test in tests/test_agent_loop.py):

* ``max_iterations``  - hard cap on model calls per run; the model is nudged to submit on the
  last allowed turn.
* ``token_budget``    - cumulative input+output tokens (from API ``usage``) per run.
* ``timeout``         - wall-clock cap on the whole run (asyncio.timeout).
* retries/backoff     - handled by ``RetryingClient`` (429/5xx/overloaded/connection errors).
* untrusted payloads  - alert payloads are delimited/escaped (see prompts.py); the agent has no
  write tools at all.
* schema-forced output - the run only ends successfully via the ``submit_triage`` tool, whose
  input is validated with Pydantic.
* citation provenance - every cited id must have been returned by a tool earlier in this run.
  The model gets one chance to fix unknown ids; after that they are stripped and flagged.

Database access is in short transactions: no transaction is held open across a model call.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from triage.agent.audit import AuditRecorder, ToolCallRecord
from triage.agent.llm import LLMClient, LLMError, LLMResponse, ToolUse
from triage.agent.prompts import (
    FINAL_TURN_NUDGE,
    NO_SUBMIT_NUDGE,
    SYSTEM_PROMPT,
    initial_user_message,
)
from triage.agent.tools import (
    SUBMIT_TRIAGE,
    TOOL_DEFINITIONS,
    CitationInput,
    ProvenanceSet,
    SubmitTriageInput,
    ToolResult,
    execute_tool,
    parse_submission,
)
from triage.db.base import SessionFactory, transaction
from triage.db.models import AgentRun, Incident, TriageProposal
from triage.logging_config import get_logger
from triage.services.errors import ConflictError, NotFoundError

log = get_logger(__name__)

# Termination reasons (stored on agent_runs.termination_reason).
SUBMITTED = "submitted"
MAX_ITERATIONS = "max_iterations"
TOKEN_BUDGET = "token_budget_exceeded"  # noqa: S105 - not a secret
TIMEOUT = "timeout"
LLM_ERROR = "llm_error"
REFUSAL = "refusal"
MAX_TOKENS = "max_tokens"
UNEXPECTED_STOP = "unexpected_stop_reason"
INVALID_CITATIONS = "invalid_citations"
INTERNAL_ERROR = "internal_error"
CANCELLED = "cancelled"

FLAG_INJECTION_MODEL = "prompt_injection_reported_by_model"
FLAG_INJECTION_HEURISTIC = "prompt_injection_heuristic_match"
FLAG_CITATIONS_REMOVED = "unverified_citations_removed"


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str
    max_iterations: int = 8
    token_budget: int = 60_000
    timeout_seconds: float = 90.0
    max_tokens_per_call: int = 2_048
    tool_output_max_chars: int = 12_000


@dataclass(frozen=True, slots=True)
class AgentOutcome:
    run_id: int
    succeeded: bool
    termination_reason: str
    proposal_id: int | None = None
    detail: str | None = None


@dataclass(slots=True)
class _Accepted:
    submission: SubmitTriageInput
    citations: list[CitationInput]
    rejected: list[CitationInput]


@dataclass(slots=True)
class _Stopped:
    reason: str
    detail: str


@dataclass(slots=True)
class _RunState:
    run_id: int
    incident_id: int
    provenance: ProvenanceSet = field(default_factory=ProvenanceSet)
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    last_stop_reason: str | None = None
    citation_retry_used: bool = False

    @property
    def tokens_used(self) -> int:
        return self.input_tokens + self.output_tokens

    def add_usage(self, response: LLMResponse) -> None:
        u = response.usage
        # Cache reads/writes are still input tokens for budgeting purposes.
        self.input_tokens += (
            u.input_tokens + u.cache_creation_input_tokens + u.cache_read_input_tokens
        )
        self.output_tokens += u.output_tokens


def partition_citations(
    citations: list[CitationInput], provenance: ProvenanceSet
) -> tuple[list[CitationInput], list[CitationInput]]:
    """Split citations into (seen-in-this-run, not-seen), de-duplicated, order preserved."""
    seen: set[tuple[str, int]] = set()
    valid: list[CitationInput] = []
    invalid: list[CitationInput] = []
    for c in citations:
        key = (c.kind, c.id)
        if key in seen:
            continue
        seen.add(key)
        (valid if provenance.contains(c.kind, c.id) else invalid).append(c)
    return valid, invalid


def _tool_input_for_audit(raw: object) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {"_non_object_input": repr(raw)[:2_000]}


class TriageAgent:
    def __init__(
        self,
        llm: LLMClient,
        sessions: SessionFactory,
        config: AgentConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._llm = llm
        self._sessions = sessions
        self._config = config
        self._clock = clock
        self._audit = AuditRecorder(sessions)

    async def run(self, incident_id: int) -> AgentOutcome:
        async with transaction(self._sessions) as session:
            incident = await session.get(Incident, incident_id)
            if incident is None:
                raise NotFoundError(f"incident {incident_id} not found")
            if incident.status != "open":
                raise ConflictError(f"incident {incident_id} is {incident.status}")
            run = AgentRun(incident_id=incident_id, model=self._config.model, status="running")
            session.add(run)
            await session.flush()
            state = _RunState(run_id=run.id, incident_id=incident_id)
            first_message = initial_user_message(incident_id, incident.service, incident.title)

        bound = log.bind(agent_run_id=state.run_id, incident_id=incident_id)
        bound.info("agent_run_started", model=self._config.model)

        # Every exit path below must finalise the agent_runs row; a run left in 'running' is an
        # audit-trail lie. Unexpected exceptions and cancellation finalise, then propagate.
        timeout_cm = asyncio.timeout(self._config.timeout_seconds)
        result: _Accepted | _Stopped
        try:
            async with timeout_cm:
                result = await self._loop(state, first_message)
        except TimeoutError:
            if not timeout_cm.expired():
                await self._finish_failed(state, _Stopped(INTERNAL_ERROR, "unexpected timeout"))
                raise
            result = _Stopped(TIMEOUT, f"run exceeded {self._config.timeout_seconds}s")
        except LLMError as exc:
            result = _Stopped(LLM_ERROR, str(exc))
        except asyncio.CancelledError:
            # Shielded so the status write itself is not cancelled; then re-raise.
            await asyncio.shield(self._finish_failed(state, _Stopped(CANCELLED, "cancelled")))
            raise
        except Exception as exc:
            await self._finish_failed(state, _Stopped(INTERNAL_ERROR, type(exc).__name__))
            raise

        if isinstance(result, _Stopped):
            outcome = await self._finish_failed(state, result)
        else:
            try:
                outcome = await self._finish_succeeded(state, result)
            except BaseException as exc:
                # The proposal transaction rolled back; record the run as failed.
                reason = CANCELLED if isinstance(exc, asyncio.CancelledError) else INTERNAL_ERROR
                await asyncio.shield(
                    self._finish_failed(state, _Stopped(reason, type(exc).__name__))
                )
                raise
        bound.info(
            "agent_run_finished",
            succeeded=outcome.succeeded,
            termination_reason=outcome.termination_reason,
            iterations=state.iterations,
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
        )
        return outcome

    # ------------------------------------------------------------------ loop

    async def _loop(self, state: _RunState, first_message: str) -> _Accepted | _Stopped:
        cfg = self._config
        messages: list[dict[str, Any]] = [{"role": "user", "content": first_message}]

        for iteration in range(1, cfg.max_iterations + 1):
            state.iterations = iteration
            started = self._clock()
            response = await self._llm.create(
                model=cfg.model,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                max_tokens=cfg.max_tokens_per_call,
            )
            latency_ms = int((self._clock() - started) * 1000)
            state.add_usage(response)
            state.last_stop_reason = response.stop_reason
            await self._audit.llm_call(state.run_id, iteration, response, latency_ms)

            if response.stop_reason == "refusal":
                return _Stopped(REFUSAL, "model declined to respond")
            if response.stop_reason == "max_tokens":
                # A tool_use block may be truncated; its input cannot be trusted.
                return _Stopped(MAX_TOKENS, "response hit max_tokens_per_call")
            if response.stop_reason not in ("tool_use", "end_turn", "stop_sequence"):
                return _Stopped(UNEXPECTED_STOP, f"stop_reason={response.stop_reason}")

            # The API rejects an assistant message with empty content, and an empty end_turn
            # can happen. Substitute a minimal text block so the transcript stays valid and
            # roles keep alternating.
            assistant_content = response.raw_content or [{"type": "text", "text": "(no output)"}]
            messages.append({"role": "assistant", "content": assistant_content})
            is_last = iteration == cfg.max_iterations

            # 1) A submission is judged against what the model had seen *before* this turn,
            #    so parallel tool calls in the same turn cannot launder citations.
            records: list[ToolCallRecord] = []
            results: dict[str, ToolResult] = {}
            for tu in response.tool_uses:
                if tu.name != SUBMIT_TRIAGE:
                    continue
                verdict = self._judge_submission(state, tu, is_last=is_last)
                records.append(
                    ToolCallRecord(
                        iteration=iteration,
                        tool_use_id=tu.id,
                        tool_name=tu.name,
                        input=_tool_input_for_audit(tu.input),
                        output_chars=len(verdict.content) if isinstance(verdict, ToolResult) else 0,
                        is_error=not isinstance(verdict, _Accepted),
                        latency_ms=0,
                    )
                )
                if isinstance(verdict, _Accepted | _Stopped):
                    await self._audit.tool_calls(state.run_id, records)
                    return verdict
                results[tu.id] = verdict

            # 2) Budget is checked after a possible submission: tokens already spent on a
            #    valid final answer are sunk, and discarding it would waste them.
            if state.tokens_used > cfg.token_budget:
                await self._audit.tool_calls(state.run_id, records)
                return _Stopped(
                    TOKEN_BUDGET, f"used {state.tokens_used} tokens > budget {cfg.token_budget}"
                )

            # 3) Read-only tools.
            for tu in response.tool_uses:
                if tu.name == SUBMIT_TRIAGE:
                    continue
                result, record = await self._run_tool(iteration, tu, state.provenance)
                results[tu.id] = result
                records.append(record)
            await self._audit.tool_calls(state.run_id, records)

            # 4) Next user turn: every tool_use gets a tool_result, results first.
            content: list[dict[str, Any]] = [
                {
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": results[tu.id].content,
                    **({"is_error": True} if results[tu.id].is_error else {}),
                }
                for tu in response.tool_uses
            ]
            if not response.tool_uses:
                content.append({"type": "text", "text": NO_SUBMIT_NUDGE})
            if iteration == cfg.max_iterations - 1:
                content.append({"type": "text", "text": FINAL_TURN_NUDGE})
            messages.append({"role": "user", "content": content})

        return _Stopped(MAX_ITERATIONS, f"no valid submission after {cfg.max_iterations} turns")

    def _judge_submission(
        self, state: _RunState, tu: ToolUse, *, is_last: bool
    ) -> _Accepted | _Stopped | ToolResult:
        parsed = parse_submission(tu.input)
        if isinstance(parsed, ToolResult):
            return parsed  # schema error -> back to the model
        valid, invalid = partition_citations(parsed.citations, state.provenance)
        if not invalid:
            return _Accepted(parsed, valid, [])
        can_retry = not state.citation_retry_used and not is_last
        if can_retry:
            state.citation_retry_used = True
            body = {
                "error": "unverified_citations",
                "unknown": [c.model_dump() for c in invalid],
                "message": "These ids were not returned by any tool in this session. "
                "Resubmit citing only ids you have seen in tool results.",
            }
            return ToolResult(json.dumps(body), is_error=True)
        if valid:
            return _Accepted(parsed, valid, invalid)
        return _Stopped(INVALID_CITATIONS, "no citation could be verified against tool results")

    async def _run_tool(
        self, iteration: int, tu: ToolUse, provenance: ProvenanceSet
    ) -> tuple[ToolResult, ToolCallRecord]:
        started = self._clock()
        try:
            async with transaction(self._sessions) as session:
                result = await execute_tool(
                    session,
                    tu.name,
                    tu.input,
                    provenance,
                    max_output_chars=self._config.tool_output_max_chars,
                )
        except Exception:
            log.exception("agent_tool_failed", tool=tu.name)
            result = ToolResult('{"error":"internal_tool_error"}', is_error=True)
        latency_ms = int((self._clock() - started) * 1000)
        record = ToolCallRecord(
            iteration=iteration,
            tool_use_id=tu.id,
            tool_name=tu.name,
            input=_tool_input_for_audit(tu.input),
            output_chars=len(result.content),
            is_error=result.is_error,
            latency_ms=latency_ms,
        )
        return result, record

    # ------------------------------------------------------------------ finish

    async def _finish_succeeded(self, state: _RunState, accepted: _Accepted) -> AgentOutcome:
        sub = accepted.submission
        flags: list[str] = []
        if sub.suspected_prompt_injection:
            flags.append(FLAG_INJECTION_MODEL)
        if state.provenance.suspicious_alerts:
            flags.append(FLAG_INJECTION_HEURISTIC)
        if accepted.rejected:
            flags.append(FLAG_CITATIONS_REMOVED)

        async with transaction(self._sessions) as session:
            proposal = TriageProposal(
                incident_id=state.incident_id,
                agent_run_id=state.run_id,
                status="pending_review",
                summary=sub.summary,
                probable_cause=sub.probable_cause,
                severity_level=sub.severity_level,
                severity_rationale=sub.severity_rationale,
                confidence=sub.confidence,
                next_steps=sub.next_steps,
                citations=[c.model_dump() for c in accepted.citations],
                rejected_citations=[c.model_dump() for c in accepted.rejected],
                flags=flags,
            )
            session.add(proposal)
            await self._update_run(session, state, status="succeeded", reason=SUBMITTED)
            await session.flush()
            proposal_id = proposal.id
        return AgentOutcome(state.run_id, True, SUBMITTED, proposal_id=proposal_id)

    async def _finish_failed(self, state: _RunState, stopped: _Stopped) -> AgentOutcome:
        try:
            async with transaction(self._sessions) as session:
                await self._update_run(
                    session, state, status="failed", reason=stopped.reason, error=stopped.detail
                )
        except Exception:
            log.exception("agent_run_finalise_failed", agent_run_id=state.run_id)
        return AgentOutcome(state.run_id, False, stopped.reason, detail=stopped.detail)

    async def _update_run(
        self,
        session: AsyncSession,
        state: _RunState,
        *,
        status: str,
        reason: str,
        error: str | None = None,
    ) -> None:
        run = await session.get_one(AgentRun, state.run_id)
        run.status = status
        run.termination_reason = reason
        run.last_stop_reason = state.last_stop_reason
        run.iterations = state.iterations
        run.input_tokens = state.input_tokens
        run.output_tokens = state.output_tokens
        run.error = error
        run.finished_at = datetime.now(UTC)
