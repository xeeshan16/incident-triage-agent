"""HTTP routes. Handlers are thin: validate, call a service, shape the response."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Annotated, Literal

from fastapi import APIRouter, Path, Query, Response, status
from sqlalchemy import text

from triage.api.deps import AgentDep, SessionDep, SessionFactoryDep, SettingsDep
from triage.schemas.api import (
    PG_BIGINT_MAX,
    AlertIn,
    AlertIngestResult,
    AlertOut,
    IncidentDetail,
    IncidentStatus,
    IncidentSummary,
    Page,
    ReviewIn,
    RunbookIn,
    RunbookOut,
    RunbookSearchHit,
    Text,
    TriageProposalOut,
)
from triage.services import alerts as alert_service
from triage.services import incidents as incident_service
from triage.services import review as review_service
from triage.services import runbooks as runbook_service
from triage.services.errors import AgentRunFailedError

router = APIRouter()

READINESS_TIMEOUT_SECONDS = 2.0

Limit = Annotated[int, Query(ge=1, le=200)]
Offset = Annotated[int, Query(ge=0, le=100_000)]
# Bounded to Postgres BIGINT so out-of-range ids are a 422, not a driver error (500).
IncidentId = Annotated[int, Path(ge=1, le=PG_BIGINT_MAX)]
ProposalId = Annotated[int, Path(ge=1, le=PG_BIGINT_MAX)]
ServiceFilter = Annotated[Text | None, Query(max_length=100)]


@router.get("/healthz", tags=["ops"])
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", tags=["ops"])
async def readyz(factory: SessionFactoryDep, response: Response) -> dict[str, str]:
    # Bounded: an unreachable database host otherwise blocks on TCP connect for ~60s, well past
    # any orchestrator probe timeout.
    try:
        async with asyncio.timeout(READINESS_TIMEOUT_SECONDS), factory() as session:
            await session.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 - any DB failure (incl. timeout) means not ready
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unavailable", "database": "error"}
    return {"status": "ok", "database": "ok"}


# ----------------------------------------------------------------------------- alerts


@router.post(
    "/alerts",
    status_code=status.HTTP_201_CREATED,
    response_model=AlertIngestResult,
    tags=["alerts"],
)
async def ingest_alert(
    body: AlertIn, session: SessionDep, settings: SettingsDep, response: Response
) -> AlertIngestResult:
    outcome = await alert_service.ingest_alert(
        session,
        body,
        dedup_window=timedelta(seconds=settings.dedup_window_seconds),
        group_window=timedelta(seconds=settings.group_window_seconds),
    )
    if outcome.deduplicated:
        response.status_code = status.HTTP_200_OK
    return AlertIngestResult(
        alert=AlertOut.model_validate(outcome.alert),
        incident_id=outcome.incident.id,
        deduplicated=outcome.deduplicated,
        new_incident=outcome.new_incident,
    )


# ----------------------------------------------------------------------------- incidents


@router.get("/incidents", response_model=Page[IncidentSummary], tags=["incidents"])
async def list_incidents(
    session: SessionDep,
    status_: Annotated[IncidentStatus | None, Query(alias="status")] = None,
    service: ServiceFilter = None,
    limit: Limit = 50,
    offset: Offset = 0,
) -> Page[IncidentSummary]:
    rows = await incident_service.list_incidents(
        session, status=status_, service=service, limit=limit, offset=offset
    )
    return Page[IncidentSummary](
        items=[IncidentSummary.model_validate(r) for r in rows], limit=limit, offset=offset
    )


@router.get("/incidents/{incident_id}", response_model=IncidentDetail, tags=["incidents"])
async def get_incident(incident_id: IncidentId, session: SessionDep) -> IncidentDetail:
    incident = await incident_service.get_incident_detail(session, incident_id)
    return IncidentDetail.model_validate(incident)


@router.post(
    "/incidents/{incident_id}/triage",
    response_model=TriageProposalOut,
    status_code=status.HTTP_201_CREATED,
    tags=["triage"],
    responses={502: {"description": "Agent run ended without a valid proposal"}},
)
async def triage_incident(
    incident_id: IncidentId, agent: AgentDep, factory: SessionFactoryDep
) -> TriageProposalOut:
    # No request-scoped transaction here: the agent manages short transactions itself so that
    # no DB connection is held for the duration of model calls.
    outcome = await agent.run(incident_id)
    if not outcome.succeeded or outcome.proposal_id is None:
        raise AgentRunFailedError(
            run_id=outcome.run_id,
            reason=outcome.termination_reason,
            detail=outcome.detail or "",
        )
    async with factory() as session:
        proposal = await review_service.get_proposal(session, outcome.proposal_id)
        return TriageProposalOut.model_validate(proposal)


# ----------------------------------------------------------------------------- review


async def _review(
    session: SessionDep,
    proposal_id: int,
    body: ReviewIn,
    decision: Literal["approved", "rejected"],
) -> TriageProposalOut:
    proposal = await review_service.review_proposal(
        session, proposal_id, decision=decision, reviewer=body.reviewer, note=body.note
    )
    return TriageProposalOut.model_validate(proposal)


@router.get("/triage/{proposal_id}", response_model=TriageProposalOut, tags=["triage"])
async def get_triage(proposal_id: ProposalId, session: SessionDep) -> TriageProposalOut:
    return TriageProposalOut.model_validate(await review_service.get_proposal(session, proposal_id))


@router.post("/triage/{proposal_id}/approve", response_model=TriageProposalOut, tags=["triage"])
async def approve_triage(
    proposal_id: ProposalId, body: ReviewIn, session: SessionDep
) -> TriageProposalOut:
    return await _review(session, proposal_id, body, "approved")


@router.post("/triage/{proposal_id}/reject", response_model=TriageProposalOut, tags=["triage"])
async def reject_triage(
    proposal_id: ProposalId, body: ReviewIn, session: SessionDep
) -> TriageProposalOut:
    return await _review(session, proposal_id, body, "rejected")


# ----------------------------------------------------------------------------- runbooks


@router.post(
    "/runbooks",
    response_model=RunbookOut,
    status_code=status.HTTP_201_CREATED,
    tags=["runbooks"],
)
async def create_runbook(body: RunbookIn, session: SessionDep) -> RunbookOut:
    return RunbookOut.model_validate(await runbook_service.create_runbook(session, body))


@router.get("/runbooks", tags=["runbooks"])
async def list_runbooks(
    session: SessionDep,
    q: Annotated[Text | None, Query(min_length=2, max_length=200)] = None,
    service: ServiceFilter = None,
    limit: Limit = 50,
    offset: Offset = 0,
) -> Page[RunbookOut] | Page[RunbookSearchHit]:
    """List runbooks, or full-text search them when ``q`` is given (ranked, no offset)."""
    if q is not None:
        hits = await runbook_service.search_runbooks(session, q, service=service, limit=limit)
        return Page[RunbookSearchHit](
            items=[
                RunbookSearchHit(
                    id=h.id,
                    title=h.title,
                    service=h.service,
                    tags=h.tags,
                    rank=h.rank,
                    snippet=h.snippet,
                )
                for h in hits
            ],
            limit=limit,
            offset=0,
        )
    rows = await runbook_service.list_runbooks(session, service=service, limit=limit, offset=offset)
    return Page[RunbookOut](
        items=[RunbookOut.model_validate(r) for r in rows], limit=limit, offset=offset
    )
