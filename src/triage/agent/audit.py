"""Persists the agent audit trail (LLM calls and tool calls) in their own short transactions.

Audit rows are written as the run progresses rather than at the end, so a crash or timeout
mid-run still leaves a record of what the agent did up to that point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from triage.agent.llm import LLMResponse
from triage.db.base import SessionFactory, transaction
from triage.db.models import AgentLLMCall, AgentToolCall


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    iteration: int
    tool_use_id: str
    tool_name: str
    input: dict[str, Any]
    output_chars: int
    is_error: bool
    latency_ms: int


class AuditRecorder:
    def __init__(self, sessions: SessionFactory) -> None:
        self._sessions = sessions

    async def llm_call(
        self, run_id: int, iteration: int, response: LLMResponse, latency_ms: int
    ) -> None:
        u = response.usage
        async with transaction(self._sessions) as session:
            session.add(
                AgentLLMCall(
                    run_id=run_id,
                    iteration=iteration,
                    stop_reason=response.stop_reason,
                    input_tokens=u.input_tokens
                    + u.cache_creation_input_tokens
                    + u.cache_read_input_tokens,
                    output_tokens=u.output_tokens,
                    attempts=response.attempts,
                    latency_ms=latency_ms,
                )
            )

    async def tool_calls(self, run_id: int, records: list[ToolCallRecord]) -> None:
        if not records:
            return
        async with transaction(self._sessions) as session:
            session.add_all(
                AgentToolCall(
                    run_id=run_id,
                    iteration=r.iteration,
                    tool_use_id=r.tool_use_id,
                    tool_name=r.tool_name,
                    input=r.input,
                    output_chars=r.output_chars,
                    is_error=r.is_error,
                    latency_ms=r.latency_ms,
                )
                for r in records
            )
