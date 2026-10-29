"""Incident queries."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from triage.db.models import Incident
from triage.services.errors import NotFoundError


async def list_incidents(
    session: AsyncSession,
    *,
    status: str | None,
    service: str | None,
    limit: int,
    offset: int,
) -> Sequence[Incident]:
    stmt = select(Incident).order_by(Incident.last_seen_at.desc(), Incident.id.desc())
    if status is not None:
        stmt = stmt.where(Incident.status == status)
    if service is not None:
        stmt = stmt.where(Incident.service == service)
    result = await session.scalars(stmt.limit(limit).offset(offset))
    return result.all()


async def get_incident_detail(session: AsyncSession, incident_id: int) -> Incident:
    # selectinload: one extra query per relationship rather than one per row (no N+1).
    incident = await session.scalar(
        select(Incident)
        .where(Incident.id == incident_id)
        .options(selectinload(Incident.alerts), selectinload(Incident.proposals))
    )
    if incident is None:
        raise NotFoundError(f"incident {incident_id} not found")
    return incident
