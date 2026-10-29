"""Regression tests for defects found in the post-build review (see REVIEW.md, findings log)."""

from __future__ import annotations

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
