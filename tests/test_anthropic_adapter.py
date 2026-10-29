"""AnthropicClient adapter, exercised through the real ``anthropic`` SDK with a mock transport.

No network: the SDK's HTTP client is replaced by an httpx2 MockTransport that returns canned
Messages API responses. This checks request shape and response/error mapping, not the model.
"""

from __future__ import annotations

import json
from typing import Any

import anthropic
import httpx2
import pytest

from triage.agent.llm import AnthropicClient, LLMError, LLMTransientError
from triage.agent.tools import TOOL_DEFINITIONS

MESSAGE = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "content": [
        {"type": "text", "text": "Checking alerts."},
        {
            "type": "tool_use",
            "id": "toolu_01",
            "name": "get_incident_alerts",
            "input": {"incident_id": 7},
        },
    ],
    "usage": {
        "input_tokens": 1200,
        "output_tokens": 85,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 300,
    },
}


def _client(handler: Any) -> tuple[AnthropicClient, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def wrapped(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        result: httpx2.Response = handler(request)
        return result

    sdk = anthropic.AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(wrapped)),
    )
    return AnthropicClient(sdk=sdk), seen


async def _create(client: AnthropicClient) -> Any:
    return await client.create(
        model="claude-sonnet-5-5",
        system="sys",
        messages=[{"role": "user", "content": "hi"}],
        tools=TOOL_DEFINITIONS,
        max_tokens=512,
    )


async def test_request_shape_and_response_mapping() -> None:
    client, seen = _client(lambda _: httpx2.Response(200, json=MESSAGE))
    resp = await _create(client)

    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/messages"
    assert body["model"] == "claude-sonnet-5-5"
    assert body["tool_choice"] == {"type": "auto"}
    assert body["max_tokens"] == 512
    assert all(t["strict"] is True for t in body["tools"])
    assert {t["name"] for t in body["tools"]} >= {"submit_triage", "search_runbooks"}

    assert resp.stop_reason == "tool_use"
    assert resp.text == "Checking alerts."
    assert [(t.id, t.name, t.input) for t in resp.tool_uses] == [
        ("toolu_01", "get_incident_alerts", {"incident_id": 7})
    ]
    assert resp.usage.input_tokens == 1200
    assert resp.usage.cache_read_input_tokens == 300
    assert resp.usage.total == 1585
    # raw_content round-trips as API params for the next assistant turn.
    assert resp.raw_content[1] == {
        "type": "tool_use",
        "id": "toolu_01",
        "name": "get_incident_alerts",
        "input": {"incident_id": 7},
    }


@pytest.mark.parametrize(
    ("status", "error_type"),
    [(429, "rate_limit_error"), (500, "api_error"), (529, "overloaded_error")],
)
async def test_retryable_statuses_raise_transient(status: int, error_type: str) -> None:
    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            status,
            headers={"retry-after": "2"},
            json={"type": "error", "error": {"type": error_type, "message": "slow down"}},
        )

    client, _ = _client(handler)
    with pytest.raises(LLMTransientError) as info:
        await _create(client)
    assert info.value.retry_after == 2.0
    assert str(status) in str(info.value)


async def test_bad_request_is_not_retryable() -> None:
    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            400,
            json={"type": "error", "error": {"type": "invalid_request_error", "message": "no"}},
        )

    client, _ = _client(handler)
    with pytest.raises(LLMError) as info:
        await _create(client)
    assert not isinstance(info.value, LLMTransientError)


async def test_connection_error_is_transient() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    client, _ = _client(handler)
    with pytest.raises(LLMTransientError):
        await _create(client)


def test_build_llm_client_does_not_need_a_key_at_startup() -> None:
    from triage.config import Settings
    from triage.main import build_llm_client

    # The app must start (healthz/readyz/ingest work) without ANTHROPIC_API_KEY.
    build_llm_client(Settings(anthropic_api_key=None))
