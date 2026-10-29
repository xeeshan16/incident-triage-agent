"""Alert ingestion: fingerprint dedup and grouping into incidents.

Rules (deliberately simple and documented in the README):

* **Dedup** - an alert whose fingerprint matches an alert on an *open* incident that was last
  received within ``dedup_window`` is a duplicate: we bump ``duplicate_count`` and
  ``last_received_at`` on the existing alert instead of storing a new row.
  The window is applied in both directions so a late, back-dated alert is not folded into an
  alert that started after it.
* **Grouping** - otherwise the alert joins the most recently active *open* incident for the
  same service whose ``last_seen_at`` is within ``group_window``; failing that a new incident
  is opened. Incident severity is the highest severity among its alerts.

Concurrent ingests for the same service are serialised with a transaction-scoped advisory lock
so two identical alerts arriving at once cannot both open an incident.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from triage.db.models import SEVERITIES, Alert, Incident
from triage.schemas.api import AlertIn

_SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}  # lower index = more severe


def more_severe(a: str, b: str) -> str:
    return a if _SEVERITY_RANK[a] <= _SEVERITY_RANK[b] else b


@dataclass(frozen=True, slots=True)
class IngestOutcome:
    alert: Alert
    incident: Incident
    deduplicated: bool
    new_incident: bool


async def ingest_alert(
    session: AsyncSession,
    data: AlertIn,
    *,
    dedup_window: timedelta,
    group_window: timedelta,
    now: datetime | None = None,
) -> IngestOutcome:
    """Store (or dedup) one alert. Caller owns the transaction."""
    received_at = data.received_at or now or datetime.now(UTC)

    # Serialise ingest per service for the rest of this transaction.
    await session.execute(
        select(func.pg_advisory_xact_lock(func.hashtextextended(f"ingest:{data.service}", 0)))
    )

    duplicate = await session.scalar(
        select(Alert)
        .join(Incident, Incident.id == Alert.incident_id)
        .where(
            Alert.fingerprint == data.fingerprint,
            Alert.service == data.service,
            Incident.status == "open",
            # Both bounds matter: late/back-dated alerts must not attach to newer alerts.
            Alert.last_received_at >= received_at - dedup_window,
            Alert.received_at <= received_at + dedup_window,
        )
        .order_by(Alert.last_received_at.desc())
        .limit(1)
    )
    if duplicate is not None:
        incident = await session.get_one(Incident, duplicate.incident_id)
        duplicate.duplicate_count += 1
        duplicate.last_received_at = max(duplicate.last_received_at, received_at)
        incident.first_seen_at = min(incident.first_seen_at, received_at)
        incident.last_seen_at = max(incident.last_seen_at, received_at)
        incident.severity = more_severe(incident.severity, data.severity)
        await session.flush()
        return IngestOutcome(duplicate, incident, deduplicated=True, new_incident=False)

    existing = await session.scalar(
        select(Incident)
        .where(
            Incident.service == data.service,
            Incident.status == "open",
            Incident.last_seen_at >= received_at - group_window,
            Incident.first_seen_at <= received_at + group_window,
        )
        .order_by(Incident.last_seen_at.desc())
        .limit(1)
    )
    new_incident = existing is None
    if existing is None:
        incident = Incident(
            service=data.service,
            title=data.title,
            severity=data.severity,
            status="open",
            fingerprint=data.fingerprint,
            alert_count=0,
            first_seen_at=received_at,
            last_seen_at=received_at,
        )
        session.add(incident)
        await session.flush()
    else:
        incident = existing
        incident.severity = more_severe(incident.severity, data.severity)
        incident.first_seen_at = min(incident.first_seen_at, received_at)
        incident.last_seen_at = max(incident.last_seen_at, received_at)

    alert = Alert(
        incident_id=incident.id,
        service=data.service,
        severity=data.severity,
        title=data.title,
        fingerprint=data.fingerprint,
        payload=data.payload,
        duplicate_count=0,
        received_at=received_at,
        last_received_at=received_at,
    )
    session.add(alert)
    incident.alert_count += 1
    await session.flush()
    return IngestOutcome(alert, incident, deduplicated=False, new_incident=new_incident)
