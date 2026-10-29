"""Direct tests of each read-only tool against Postgres, including FTS and trigram search."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from tests.conftest import NOW, add_alert
from triage.agent.tools import ProvenanceSet, ToolResult, execute_tool, parse_submission
from triage.db.base import SessionFactory
from triage.db.models import Incident, Runbook, TriageProposal
from triage.services.runbooks import to_or_query


async def _call(sessions: SessionFactory, name: str, args: object) -> tuple[Any, ProvenanceSet]:
    prov = ProvenanceSet()
    async with sessions() as s:
        result = await execute_tool(s, name, args, prov, max_output_chars=50_000)
    return json.loads(result.content), prov


async def _seed_runbooks(sessions: SessionFactory) -> dict[str, int]:
    rows = [
        Runbook(
            title="Postgres connection pool exhaustion",
            service="payments-api",
            body="Symptoms: too many clients, pool timeout. Check pg_stat_activity.",
            tags=["postgres", "database"],
        ),
        Runbook(
            title="Rolling back a bad deploy",
            service=None,
            body="If error rate rose right after a deployment, roll back via the deploy tool.",
            tags=["deploy", "rollback"],
        ),
        Runbook(
            title="Ledger reconciliation lag",
            service="ledger",
            body="Reconciliation jobs falling behind; check the queue consumer and database.",
            tags=["queue"],
        ),
    ]
    async with sessions.begin() as s:
        s.add_all(rows)
    return {r.title: r.id for r in rows}


async def test_get_incident_alerts_records_provenance(sessions: SessionFactory) -> None:
    a1, inc = await add_alert(sessions, fingerprint="a")
    a2, _ = await add_alert(sessions, fingerprint="b", at=NOW + timedelta(minutes=1))
    body, prov = await _call(sessions, "get_incident_alerts", {"incident_id": inc})
    assert [a["id"] for a in body["alerts"]] == [a1, a2]
    assert prov.alert == {a1, a2}
    assert prov.incident == {inc}
    assert body["alerts"][0]["title"].startswith("<untrusted_text>")


async def test_get_incident_alerts_not_found_is_error(sessions: SessionFactory) -> None:
    prov = ProvenanceSet()
    async with sessions() as s:
        r = await execute_tool(
            s, "get_incident_alerts", {"incident_id": 99}, prov, max_output_chars=1000
        )
    assert r.is_error
    assert json.loads(r.content)["error"] == "not_found"
    assert prov.incident == set()


async def test_search_runbooks_fts_ranking_and_service_scope(sessions: SessionFactory) -> None:
    ids = await _seed_runbooks(sessions)
    body, prov = await _call(
        sessions, "search_runbooks", {"query": "postgres pool timeout", "service": "payments-api"}
    )
    titles = [r["title"] for r in body["results"]]
    assert titles[0] == "Postgres connection pool exhaustion"
    assert "Ledger reconciliation lag" not in titles  # other service excluded
    assert prov.runbook == {r["id"] for r in body["results"]}

    # Generic (service IS NULL) runbooks are always eligible.
    body, _ = await _call(
        sessions, "search_runbooks", {"query": "deployment rollback", "service": "payments-api"}
    )
    assert [r["id"] for r in body["results"]] == [ids["Rolling back a bad deploy"]]

    # Stemming: "deployments" matches "deployment".
    body, _ = await _call(sessions, "search_runbooks", {"query": "deployments"})
    assert body["results"]


async def test_search_runbooks_hostile_query_is_safe(sessions: SessionFactory) -> None:
    await _seed_runbooks(sessions)
    for q in ["'; DROP TABLE runbooks; --", "!!! && || :* <-> (((", '"unterminated', "or or"]:
        body, _ = await _call(sessions, "search_runbooks", {"query": q})
        assert "results" in body
    async with sessions() as s:
        assert await s.get(Runbook, 1) is not None


def test_to_or_query() -> None:
    assert to_or_query("db pool exhausted") == "db or pool or exhausted"
    assert to_or_query('-"quoted" OR x') == "quoted or x"
    assert to_or_query("a " * 50).count(" or ") == 15


async def test_find_similar_incidents(sessions: SessionFactory) -> None:
    # Past incidents (spaced beyond grouping window) on same service.
    _, old_same_fp = await add_alert(sessions, at=NOW - timedelta(days=3))
    _, old_similar = await add_alert(
        sessions,
        title="Postgres connection pool exhausted on replica",
        fingerprint="other-fp",
        at=NOW - timedelta(days=2),
    )
    _, unrelated = await add_alert(
        sessions, title="TLS certificate expiring", fingerprint="tls", at=NOW - timedelta(days=1)
    )
    _, other_service = await add_alert(sessions, service="ledger", at=NOW)
    _, current = await add_alert(sessions, at=NOW)
    async with sessions.begin() as s:
        for i in (old_same_fp, old_similar, unrelated):
            (await s.get_one(Incident, i)).status = "actioned"
    async with sessions.begin() as s:
        from triage.db.models import AgentRun

        run = AgentRun(incident_id=old_same_fp, model="m", status="succeeded")
        s.add(run)
        await s.flush()
        s.add(
            TriageProposal(
                incident_id=old_same_fp,
                agent_run_id=run.id,
                status="approved",
                summary="Pool leak in refunds worker.",
                probable_cause="leak",
                severity_level="high",
                severity_rationale="r",
                confidence="high",
                next_steps=["x"],
                citations=[],
            )
        )

    body, prov = await _call(sessions, "find_similar_incidents", {"incident_id": current})
    ids = [i["id"] for i in body["incidents"]]
    assert ids[0] == old_same_fp  # same fingerprint ranks first
    assert old_similar in ids
    assert unrelated not in ids
    assert other_service not in ids
    assert current not in ids
    assert body["incidents"][0]["approved_triage_summary"] == "Pool leak in refunds worker."
    assert prov.incident == set(ids)


async def test_service_recent_history_window(sessions: SessionFactory) -> None:
    _, recent = await add_alert(sessions, at=NOW - timedelta(days=2))
    _, old = await add_alert(sessions, fingerprint="z", at=NOW - timedelta(days=40))
    body, prov = await _call(
        sessions, "get_service_recent_history", {"service": "payments-api", "days": 14}
    )
    assert [i["id"] for i in body["incidents"]] == [recent]
    assert old not in prov.incident


def test_parse_submission_rejects_extra_fields_and_bad_types() -> None:
    result = parse_submission({"summary": 1, "surprise": True})
    assert isinstance(result, ToolResult)
    body = json.loads(result.content)
    locs = {d["loc"] for d in body["details"]}
    assert {"summary", "surprise", "citations", "next_steps"} <= locs


def test_strict_api_schemas_avoid_unsupported_keywords() -> None:
    """Strict tool use rejects numeric/length constraints; keep them in Pydantic only."""
    from triage.agent.tools import TOOL_DEFINITIONS

    forbidden = {"minimum", "maximum", "minLength", "maxLength", "maxItems"}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            assert not forbidden & node.keys(), node
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False, node
            if "minItems" in node:
                assert node["minItems"] in (0, 1)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for tool in TOOL_DEFINITIONS:
        assert tool["strict"] is True
        walk(tool["input_schema"])
