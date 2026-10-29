"""Alert ingestion: dedup window, grouping window, severity escalation, concurrency."""

from __future__ import annotations

import asyncio
from datetime import timedelta

from sqlalchemy import func, select

from tests.conftest import NOW, add_alert
from triage.db.base import SessionFactory
from triage.db.models import Alert, Incident


async def test_first_alert_opens_incident(sessions: SessionFactory) -> None:
    alert_id, incident_id = await add_alert(sessions)
    async with sessions() as s:
        incident = await s.get_one(Incident, incident_id)
        assert incident.status == "open"
        assert incident.alert_count == 1
        assert incident.fingerprint == "pg-pool-exhausted"
        assert (await s.get_one(Alert, alert_id)).incident_id == incident_id


async def test_same_fingerprint_within_window_is_deduplicated(sessions: SessionFactory) -> None:
    a1, i1 = await add_alert(sessions, at=NOW)
    a2, i2 = await add_alert(sessions, at=NOW + timedelta(minutes=5), severity="critical")
    assert (a1, i1) == (a2, i2)
    async with sessions() as s:
        alert = await s.get_one(Alert, a1)
        incident = await s.get_one(Incident, i1)
        assert alert.duplicate_count == 1
        assert alert.last_received_at == NOW + timedelta(minutes=5)
        assert incident.alert_count == 1  # dedup does not create a new alert row
        assert incident.severity == "critical"  # but does escalate severity
        assert await s.scalar(select(func.count()).select_from(Alert)) == 1


async def test_dedup_window_slides_with_last_received(sessions: SessionFactory) -> None:
    # Each duplicate refreshes last_received_at, so a steady flapping alert keeps deduping.
    a1, _ = await add_alert(sessions, at=NOW)
    for minutes in (8, 16, 24):
        a, _ = await add_alert(sessions, at=NOW + timedelta(minutes=minutes))
        assert a == a1


async def test_same_fingerprint_after_dedup_window_is_new_alert_same_incident(
    sessions: SessionFactory,
) -> None:
    a1, i1 = await add_alert(sessions, at=NOW)
    a2, i2 = await add_alert(sessions, at=NOW + timedelta(minutes=15))
    assert a1 != a2
    assert i1 == i2  # still within the 30 min grouping window


async def test_different_fingerprint_same_service_groups(sessions: SessionFactory) -> None:
    _, i1 = await add_alert(sessions, fingerprint="pg-pool", at=NOW)
    _, i2 = await add_alert(sessions, fingerprint="latency-p99", at=NOW + timedelta(minutes=2))
    assert i1 == i2
    async with sessions() as s:
        assert (await s.get_one(Incident, i1)).alert_count == 2


async def test_outside_group_window_opens_new_incident(sessions: SessionFactory) -> None:
    _, i1 = await add_alert(sessions, fingerprint="a", at=NOW)
    _, i2 = await add_alert(sessions, fingerprint="b", at=NOW + timedelta(minutes=45))
    assert i1 != i2


async def test_different_service_never_groups(sessions: SessionFactory) -> None:
    _, i1 = await add_alert(sessions, service="payments-api", at=NOW)
    _, i2 = await add_alert(sessions, service="ledger", at=NOW)
    assert i1 != i2


async def test_actioned_incident_is_not_reused(sessions: SessionFactory) -> None:
    _, i1 = await add_alert(sessions, at=NOW)
    async with sessions.begin() as s:
        (await s.get_one(Incident, i1)).status = "actioned"
    a2, i2 = await add_alert(sessions, at=NOW + timedelta(minutes=1))
    assert i2 != i1
    async with sessions() as s:
        assert (await s.get_one(Alert, a2)).duplicate_count == 0


async def test_concurrent_identical_alerts_open_one_incident(sessions: SessionFactory) -> None:
    results = await asyncio.gather(*(add_alert(sessions, at=NOW) for _ in range(8)))
    assert len({incident for _, incident in results}) == 1
    assert len({alert for alert, _ in results}) == 1
    async with sessions() as s:
        alert = (await s.scalars(select(Alert))).one()
        assert alert.duplicate_count == 7


async def test_late_backdated_alert_does_not_join_newer_incident(
    sessions: SessionFactory,
) -> None:
    _, newer = await add_alert(sessions, at=NOW)
    a_old, older = await add_alert(sessions, at=NOW - timedelta(days=3))
    assert older != newer
    async with sessions() as s:
        assert (await s.get_one(Alert, a_old)).duplicate_count == 0


async def test_slightly_out_of_order_alert_extends_incident_backwards(
    sessions: SessionFactory,
) -> None:
    _, i1 = await add_alert(sessions, fingerprint="a", at=NOW)
    _, i2 = await add_alert(sessions, fingerprint="b", at=NOW - timedelta(minutes=5))
    assert i1 == i2
    async with sessions() as s:
        incident = await s.get_one(Incident, i1)
        assert incident.first_seen_at == NOW - timedelta(minutes=5)
        assert incident.last_seen_at == NOW
