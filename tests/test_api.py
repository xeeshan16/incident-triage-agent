"""HTTP API tests (ASGI in-process, real Postgres, scripted LLM)."""

from __future__ import annotations

from typing import Any

from httpx import AsyncClient

from tests.test_agent_loop import submit_citing_seen_alerts
from triage.agent.fake import ScriptedLLM, text_reply, tool_call

ALERT = {
    "service": "payments-api",
    "severity": "high",
    "title": "Postgres connection pool exhausted",
    "fingerprint": "pg-pool-exhausted",
    "payload": {"pool_size": 20, "waiting": 143},
}


async def _post_alert(client: AsyncClient, **overrides: Any) -> dict[str, Any]:
    r = await client.post("/alerts", json={**ALERT, **overrides})
    assert r.status_code in (200, 201), r.text
    body: dict[str, Any] = r.json()
    return body


async def test_health_and_ready(client: AsyncClient) -> None:
    assert (await client.get("/healthz")).json() == {"status": "ok"}
    r = await client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["database"] == "ok"


async def test_readyz_reports_db_failure(settings: Any, engine: Any) -> None:
    from httpx import ASGITransport

    from triage.main import create_app

    bad = settings.model_copy(
        update={"database_url": "postgresql+asyncpg://nobody:wrong@127.0.0.1:1/none"}
    )
    app = create_app(bad, llm=ScriptedLLM([]))
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c,
    ):
        r = await c.get("/readyz")
    assert r.status_code == 503
    assert r.json()["status"] == "unavailable"


async def test_request_id_is_echoed_or_generated(client: AsyncClient) -> None:
    r = await client.get("/healthz", headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"
    r = await client.get("/healthz", headers={"X-Request-ID": "x" * 200})
    assert r.headers["X-Request-ID"] != "x" * 200
    assert len(r.headers["X-Request-ID"]) == 32


async def test_ingest_dedup_and_list(client: AsyncClient) -> None:
    first = await _post_alert(client)
    assert first["new_incident"] is True
    assert first["deduplicated"] is False
    r = await client.post("/alerts", json=ALERT)
    assert r.status_code == 200
    assert r.json()["deduplicated"] is True
    assert r.json()["alert"]["duplicate_count"] == 1

    page = (await client.get("/incidents")).json()
    assert len(page["items"]) == 1
    assert page["items"][0]["alert_count"] == 1

    assert (await client.get("/incidents", params={"status": "actioned"})).json()["items"] == []
    assert (await client.get("/incidents", params={"service": "nope"})).json()["items"] == []


async def test_ingest_validation(client: AsyncClient) -> None:
    bad_cases: list[dict[str, Any]] = [
        {**ALERT, "severity": "sev0"},
        {**ALERT, "service": "has spaces"},
        {**ALERT, "unexpected": 1},
        {**ALERT, "received_at": "2026-01-01T00:00:00"},  # naive timestamp
        {**ALERT, "payload": {"x": "y" * 40_000}},
        {k: v for k, v in ALERT.items() if k != "fingerprint"},
    ]
    for body in bad_cases:
        r = await client.post("/alerts", json=body)
        assert r.status_code == 422, body


async def test_get_incident_detail_and_404(client: AsyncClient) -> None:
    created = await _post_alert(client)
    r = await client.get(f"/incidents/{created['incident_id']}")
    assert r.status_code == 200
    body = r.json()
    assert len(body["alerts"]) == 1
    assert body["alerts"][0]["payload"] == ALERT["payload"]
    assert body["proposals"] == []
    assert (await client.get("/incidents/999999")).status_code == 404


async def test_triage_approve_flow(client: AsyncClient, scripted: ScriptedLLM) -> None:
    created = await _post_alert(client)
    incident_id = created["incident_id"]
    scripted.steps.extend(
        [tool_call("get_incident_alerts", {"incident_id": incident_id}), submit_citing_seen_alerts]
    )
    r = await client.post(f"/incidents/{incident_id}/triage")
    assert r.status_code == 201, r.text
    proposal = r.json()
    assert proposal["status"] == "pending_review"
    assert {"kind": "alert", "id": created["alert"]["id"]} in proposal["citations"]

    # Agent proposing does not action the incident.
    incident = (await client.get(f"/incidents/{incident_id}")).json()
    assert incident["status"] == "open"
    assert len(incident["proposals"]) == 1

    r = await client.post(
        f"/triage/{proposal['id']}/approve", json={"reviewer": "zeeshan", "note": "agreed"}
    )
    assert r.status_code == 200
    assert r.json()["status"] == "approved"
    assert r.json()["reviewer"] == "zeeshan"
    assert r.json()["reviewed_at"] is not None
    assert (await client.get(f"/incidents/{incident_id}")).json()["status"] == "actioned"

    # Double review conflicts; triage of an actioned incident conflicts.
    r = await client.post(f"/triage/{proposal['id']}/reject", json={"reviewer": "someone"})
    assert r.status_code == 409
    assert (await client.post(f"/incidents/{incident_id}/triage")).status_code == 409


async def test_triage_reject_keeps_incident_open(
    client: AsyncClient, scripted: ScriptedLLM
) -> None:
    created = await _post_alert(client)
    incident_id = created["incident_id"]
    scripted.steps.extend(
        [tool_call("get_incident_alerts", {"incident_id": incident_id}), submit_citing_seen_alerts]
    )
    proposal = (await client.post(f"/incidents/{incident_id}/triage")).json()
    r = await client.post(
        f"/triage/{proposal['id']}/reject", json={"reviewer": "zeeshan", "note": "wrong cause"}
    )
    assert r.status_code == 200
    assert r.json()["status"] == "rejected"
    assert (await client.get(f"/incidents/{incident_id}")).json()["status"] == "open"
    assert (await client.get(f"/triage/{proposal['id']}")).json()["review_note"] == "wrong cause"


async def test_review_validation_and_404(client: AsyncClient) -> None:
    assert (await client.post("/triage/1/approve", json={"reviewer": ""})).status_code == 422
    assert (await client.post("/triage/1/approve", json={})).status_code == 422
    assert (await client.post("/triage/424242/approve", json={"reviewer": "z"})).status_code == 404
    assert (await client.get("/triage/424242")).status_code == 404


async def test_triage_failure_returns_502_with_run_id(
    client: AsyncClient, scripted: ScriptedLLM
) -> None:
    created = await _post_alert(client)
    scripted.steps.append(text_reply("", stop_reason="refusal"))
    r = await client.post(f"/incidents/{created['incident_id']}/triage")
    assert r.status_code == 502
    assert r.json()["termination_reason"] == "refusal"
    assert isinstance(r.json()["agent_run_id"], int)


async def test_triage_unknown_incident_404(client: AsyncClient) -> None:
    assert (await client.post("/incidents/999/triage")).status_code == 404


async def test_unhandled_error_returns_500_with_request_id(
    client: AsyncClient, scripted: ScriptedLLM
) -> None:
    created = await _post_alert(client)
    scripted.steps.append(RuntimeError("kaboom"))
    r = await client.post(f"/incidents/{created['incident_id']}/triage")
    assert r.status_code == 500
    assert r.json() == {"detail": "internal server error"}
    assert "X-Request-ID" in r.headers


async def test_runbooks_create_list_search(client: AsyncClient) -> None:
    r = await client.post(
        "/runbooks",
        json={
            "title": "Postgres connection pool exhaustion",
            "service": "payments-api",
            "body": "Inspect pg_stat_activity for idle in transaction sessions.",
            "tags": ["Postgres", " pool "],
        },
    )
    assert r.status_code == 201
    assert r.json()["tags"] == ["postgres", "pool"]
    await client.post(
        "/runbooks",
        json={"title": "Rolling back a bad deploy", "body": "Use the deploy tool to roll back."},
    )
    listing = (await client.get("/runbooks")).json()
    assert [rb["title"] for rb in listing["items"]] == [
        "Postgres connection pool exhaustion",
        "Rolling back a bad deploy",
    ]
    hits = (await client.get("/runbooks", params={"q": "connection pool"})).json()["items"]
    assert [h["title"] for h in hits] == ["Postgres connection pool exhaustion"]
    assert hits[0]["rank"] > 0
    assert "**" in hits[0]["snippet"] or hits[0]["snippet"]
    assert (await client.get("/runbooks", params={"q": "x"})).status_code == 422
