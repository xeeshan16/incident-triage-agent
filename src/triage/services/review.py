"""Human-in-the-loop review of triage proposals. The agent never calls into this module."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from triage.db.models import Incident, TriageProposal
from triage.services.errors import ConflictError, NotFoundError


async def get_proposal(session: AsyncSession, proposal_id: int) -> TriageProposal:
    proposal = await session.get(TriageProposal, proposal_id)
    if proposal is None:
        raise NotFoundError(f"triage proposal {proposal_id} not found")
    return proposal


async def review_proposal(
    session: AsyncSession,
    proposal_id: int,
    *,
    decision: Literal["approved", "rejected"],
    reviewer: str,
    note: str,
) -> TriageProposal:
    """Approve or reject a pending proposal. Approval marks the incident ``actioned``.

    The proposal row is locked (SELECT ... FOR UPDATE) so two reviewers racing on the same
    proposal cannot both succeed; the loser gets a ConflictError.
    """
    proposal = await session.scalar(
        select(TriageProposal).where(TriageProposal.id == proposal_id).with_for_update()
    )
    if proposal is None:
        raise NotFoundError(f"triage proposal {proposal_id} not found")
    if proposal.status != "pending_review":
        raise ConflictError(f"triage proposal {proposal_id} is already {proposal.status}")

    proposal.status = decision
    proposal.reviewer = reviewer
    proposal.review_note = note
    proposal.reviewed_at = datetime.now(UTC)

    if decision == "approved":
        incident = await session.scalar(
            select(Incident).where(Incident.id == proposal.incident_id).with_for_update()
        )
        if incident is None:  # pragma: no cover - FK makes this unreachable
            raise NotFoundError(f"incident {proposal.incident_id} not found")
        incident.status = "actioned"
    await session.flush()
    return proposal
