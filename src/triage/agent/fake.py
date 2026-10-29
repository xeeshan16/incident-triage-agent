"""Deterministic, scripted ``LLMClient`` used by tests and offline evals.

A script is a list of steps. Each step is either an ``LLMResponse``, an exception to raise, or a
callable that receives the conversation so far and returns one of those (handy for responses
that must reference ids returned by earlier tool calls).
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from triage.agent.llm import LLMResponse, ToolUse, Usage

Step = LLMResponse | BaseException | Callable[[list[dict[str, Any]]], "LLMResponse"]

_ids = itertools.count(1)


def tool_call(name: str, tool_input: object, *, usage: tuple[int, int] = (500, 80)) -> LLMResponse:
    return tool_calls([(name, tool_input)], usage=usage)


def tool_calls(
    calls: Sequence[tuple[str, object]], *, usage: tuple[int, int] = (500, 80), text: str = ""
) -> LLMResponse:
    uses = [ToolUse(id=f"toolu_fake_{next(_ids)}", name=n, input=i) for n, i in calls]
    raw: list[dict[str, Any]] = []
    if text:
        raw.append({"type": "text", "text": text})
    raw.extend({"type": "tool_use", "id": u.id, "name": u.name, "input": u.input} for u in uses)
    return LLMResponse(
        stop_reason="tool_use",
        usage=Usage(*usage),
        raw_content=raw,
        text=text,
        tool_uses=uses,
    )


def text_reply(
    text: str, *, stop_reason: str = "end_turn", usage: tuple[int, int] = (500, 40)
) -> LLMResponse:
    return LLMResponse(
        stop_reason=stop_reason,
        usage=Usage(*usage),
        raw_content=[{"type": "text", "text": text}],
        text=text,
    )


def last_tool_result(messages: list[dict[str, Any]], tool_name: str) -> Any:
    """Find the JSON body of the most recent tool_result for ``tool_name`` in the transcript."""
    names: dict[str, str] = {}
    for m in messages:
        if m["role"] == "assistant" and isinstance(m["content"], list):
            for b in m["content"]:
                if b.get("type") == "tool_use":
                    names[b["id"]] = b["name"]
    for m in reversed(messages):
        if m["role"] != "user" or not isinstance(m["content"], list):
            continue
        for b in m["content"]:
            if b.get("type") == "tool_result" and names.get(b["tool_use_id"]) == tool_name:
                return json.loads(b["content"])
    raise LookupError(f"no tool_result for {tool_name}")


@dataclass
class ScriptedLLM:
    steps: list[Step]
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def create(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> LLMResponse:
        # Snapshot the request (messages is mutated by the loop after we return).
        self.calls.append(
            {
                "model": model,
                "system": system,
                "messages": json.loads(json.dumps(messages, default=str)),
                "tools": [t["name"] for t in tools],
                "max_tokens": max_tokens,
            }
        )
        if not self.steps:
            raise AssertionError("ScriptedLLM ran out of scripted steps")
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        if callable(step):
            return step(messages)
        return step
