"""A deliberately small LLM client interface.

The agent loop depends only on ``LLMClient``. Production uses ``AnthropicClient`` (official SDK,
Messages API); tests and evals use a scripted fake. Retries live in ``RetryingClient`` rather
than in the SDK (we construct the SDK with ``max_retries=0``) so backoff behaviour is explicit,
observable in the audit trail, and unit-testable.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import anthropic
from anthropic.types import MessageParam, ToolParam


@dataclass(frozen=True, slots=True)
class ToolUse:
    id: str
    name: str
    input: object


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )


@dataclass(frozen=True, slots=True)
class LLMResponse:
    stop_reason: str | None
    usage: Usage
    # Content blocks exactly as returned, as API params. Sent back verbatim as the assistant
    # turn so that any block type we do not interpret (e.g. thinking) round-trips intact.
    raw_content: list[dict[str, Any]]
    text: str = ""
    tool_uses: list[ToolUse] = field(default_factory=list)
    attempts: int = 1


class LLMError(Exception):
    """Non-retryable failure talking to the model provider."""


class LLMTransientError(LLMError):
    """Retryable failure: 429, 5xx, overloaded, connection error or request timeout."""

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMClient(Protocol):
    async def create(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> LLMResponse: ...


# ----------------------------------------------------------------------------- Anthropic

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def _retry_after(exc: anthropic.APIStatusError) -> float | None:
    value = exc.response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


class AnthropicClient:
    """Adapter from the official ``anthropic`` SDK to ``LLMClient``."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
        sdk: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self._sdk = sdk or anthropic.AsyncAnthropic(
            api_key=api_key, timeout=timeout_seconds, max_retries=0
        )

    async def create(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> LLMResponse:
        try:
            # Messages/tools are built as plain dicts; the SDK's TypedDict params are structural,
            # so cast rather than re-declaring every block type here.
            message = await self._sdk.messages.create(
                model=model,
                system=system,
                messages=cast("list[MessageParam]", messages),
                tools=cast("list[ToolParam]", tools),
                # Forced tool_choice ("any"/"tool") is rejected by current Sonnet/Opus models,
                # so the loop relies on "auto" + strict tool schemas + nudges instead.
                tool_choice={"type": "auto"},
                max_tokens=max_tokens,
            )
        except anthropic.APIStatusError as exc:
            if exc.status_code in _RETRYABLE_STATUS:
                raise LLMTransientError(
                    f"HTTP {exc.status_code}: {exc.message}", retry_after=_retry_after(exc)
                ) from exc
            raise LLMError(f"HTTP {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:  # includes APITimeoutError
            raise LLMTransientError(f"connection error: {exc}") from exc

        raw: list[dict[str, Any]] = []
        texts: list[str] = []
        tool_uses: list[ToolUse] = []
        for block in message.content:
            raw.append(block.to_dict())
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(ToolUse(id=block.id, name=block.name, input=block.input))
        u = message.usage
        return LLMResponse(
            stop_reason=message.stop_reason,
            usage=Usage(
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                cache_creation_input_tokens=u.cache_creation_input_tokens or 0,
                cache_read_input_tokens=u.cache_read_input_tokens or 0,
            ),
            raw_content=raw,
            text="\n".join(texts),
            tool_uses=tool_uses,
        )


# ----------------------------------------------------------------------------- retries

Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_retries: int = 3
    base_seconds: float = 0.5
    max_seconds: float = 8.0

    def delay(self, attempt: int, retry_after: float | None, rng: random.Random) -> float:
        """Delay before retry number ``attempt`` (1-based). Full jitter, capped."""
        if retry_after is not None:
            return min(retry_after, self.max_seconds)
        ceiling = min(self.max_seconds, self.base_seconds * (2 ** (attempt - 1)))
        return rng.uniform(0, ceiling)


class RetryingClient:
    """Wraps an ``LLMClient`` with exponential backoff on ``LLMTransientError``."""

    def __init__(
        self,
        inner: LLMClient,
        policy: RetryPolicy,
        *,
        sleep: Sleep = asyncio.sleep,
        rng: random.Random | None = None,
        on_retry: Callable[[int, float, LLMTransientError], None] | None = None,
    ) -> None:
        self._inner = inner
        self._policy = policy
        self._sleep = sleep
        self._rng = rng or random.Random()  # noqa: S311 - jitter, not crypto
        self._on_retry = on_retry

    async def create(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_tokens: int,
    ) -> LLMResponse:
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._inner.create(
                    model=model,
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                )
            except LLMTransientError as exc:
                if attempt > self._policy.max_retries:
                    raise
                delay = self._policy.delay(attempt, exc.retry_after, self._rng)
                if self._on_retry is not None:
                    self._on_retry(attempt, delay, exc)
                await self._sleep(delay)
                continue
            return LLMResponse(
                stop_reason=response.stop_reason,
                usage=response.usage,
                raw_content=response.raw_content,
                text=response.text,
                tool_uses=response.tool_uses,
                attempts=attempt,
            )
