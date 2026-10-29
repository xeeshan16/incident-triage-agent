"""Regression tests for defects found in the post-build review (see REVIEW.md, findings log)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event

from triage.agent.fake import ScriptedLLM
from triage.config import Settings
from triage.main import create_app

ALERT = {
    "service": "payments-api",
    "severity": "high",
    "title": "Pool exhausted",
    "fingerprint": "fp",
}


# --- F1: request transaction must commit before the response is sent ---------------------------


async def test_commit_happens_before_response_start(
    settings: Settings, engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from triage.services import alerts as alert_service

    events: list[str] = []
    real_ingest = alert_service.ingest_alert

    async def recording_ingest(session: Any, *args: Any, **kwargs: Any) -> Any:
        # Record when *this request's* transaction commits (other code paths, e.g. the
        # readiness probe or pool pre-ping, also commit and must not confuse the test).
        event.listen(session.sync_session, "after_commit", lambda _: events.append("commit"))
        return await real_ingest(session, *args, **kwargs)

    monkeypatch.setattr(alert_service, "ingest_alert", recording_ingest)
    app = create_app(settings, llm=ScriptedLLM([]))

    async def recording_app(scope: Any, receive: Any, send: Any) -> None:
        async def _send(message: Any) -> None:
            if message["type"] == "http.response.start":
                events.append("response.start")
            await send(message)

        await app(scope, receive, _send)

    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=recording_app), base_url="http://t") as c,
    ):
        r = await c.post("/alerts", json=ALERT)
    assert r.status_code == 201
    assert events == ["commit", "response.start"], events


# --- F2: input Postgres rejects must be a 422, not a 500 -------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {**ALERT, "title": "bad\u0000title"},
        {**ALERT, "fingerprint": "fp\u0000"},
        {**ALERT, "payload": {"msg": "a\u0000b"}},
        {**ALERT, "payload": {"nested": [{"k\u0000": 1}]}},
    ],
)
async def test_nul_bytes_in_alerts_are_rejected(client: AsyncClient, body: dict[str, Any]) -> None:
    assert (await client.post("/alerts", json=body)).status_code == 422


async def test_nul_bytes_in_runbooks_and_reviews_are_rejected(client: AsyncClient) -> None:
    r = await client.post("/runbooks", json={"title": "t", "body": "b\u0000"})
    assert r.status_code == 422
    r = await client.post("/runbooks", json={"title": "t", "body": "b", "tags": ["x\u0000"]})
    assert r.status_code == 422
    assert (await client.post("/triage/1/approve", json={"reviewer": "z\u0000"})).status_code == 422
    assert (await client.get("/incidents", params={"service": "a\u0000"})).status_code == 422


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/incidents/99999999999999999999"),
        ("POST", "/incidents/99999999999999999999/triage"),
        ("GET", "/triage/99999999999999999999"),
        ("POST", "/triage/99999999999999999999/reject"),
        ("GET", "/incidents/0"),
        ("GET", "/incidents/-1"),
    ],
)
async def test_out_of_range_ids_are_422(client: AsyncClient, method: str, path: str) -> None:
    kwargs: dict[str, Any] = {"json": {"reviewer": "z"}} if method == "POST" else {}
    assert (await client.request(method, path, **kwargs)).status_code == 422


# --- F3: future-dated alerts must not hold an incident's grouping window open -----------------


async def test_far_future_received_at_is_rejected(client: AsyncClient) -> None:
    far = (datetime.now(UTC) + timedelta(days=365)).isoformat()
    assert (await client.post("/alerts", json={**ALERT, "received_at": far})).status_code == 422
    skew = (datetime.now(UTC) + timedelta(seconds=30)).isoformat()  # small clock skew is fine
    assert (await client.post("/alerts", json={**ALERT, "received_at": skew})).status_code == 201
