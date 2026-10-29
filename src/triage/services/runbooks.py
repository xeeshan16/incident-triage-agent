"""Runbook storage and Postgres full-text search."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from triage.db.models import Runbook
from triage.schemas.api import RunbookIn


async def create_runbook(session: AsyncSession, data: RunbookIn) -> Runbook:
    runbook = Runbook(title=data.title, service=data.service, body=data.body, tags=data.tags)
    session.add(runbook)
    await session.flush()
    await session.refresh(runbook)
    return runbook


async def list_runbooks(
    session: AsyncSession, *, service: str | None, limit: int, offset: int
) -> Sequence[Runbook]:
    stmt = select(Runbook).order_by(Runbook.id)
    if service is not None:
        stmt = stmt.where(Runbook.service == service)
    return (await session.scalars(stmt.limit(limit).offset(offset))).all()


_WORD = re.compile(r"\w[\w\-.]*", re.UNICODE)


def to_or_query(text: str, *, max_terms: int = 16) -> str:
    """Turn free text into a websearch_to_tsquery OR-expression: 'a b c' -> 'a or b or c'.

    Only word-ish tokens survive, so quotes/minus signs in the input cannot change the meaning.
    """
    words = [w for w in _WORD.findall(text) if w.lower() != "or"][:max_terms]
    return " or ".join(words)


@dataclass(frozen=True, slots=True)
class RunbookHit:
    id: int
    title: str
    service: str | None
    tags: list[str]
    rank: float
    snippet: str


async def search_runbooks(
    session: AsyncSession, query: str, *, service: str | None, limit: int
) -> list[RunbookHit]:
    """Rank runbooks with websearch_to_tsquery + ts_rank_cd over the weighted tsvector.

    websearch_to_tsquery never raises on user syntax, so arbitrary model/user input is safe.
    The query is always a bound parameter. When ``service`` is given, runbooks scoped to that
    service and generic (service IS NULL) runbooks are both eligible.

    Terms are OR-ed (see ``to_or_query``): an AND query almost never matches when the caller
    passes a handful of symptom words, and ts_rank_cd already rewards documents matching more
    of them.
    """
    tsq = func.websearch_to_tsquery("english", to_or_query(query))
    rank = func.ts_rank_cd(Runbook.search_vector, tsq).label("rank")
    snippet = func.ts_headline(
        "english",
        Runbook.body,
        tsq,
        "MaxFragments=2, MaxWords=25, MinWords=8, StartSel=**, StopSel=**",
    ).label("snippet")
    stmt = (
        select(Runbook.id, Runbook.title, Runbook.service, Runbook.tags, rank, snippet)
        .where(Runbook.search_vector.op("@@")(tsq))
        .order_by(rank.desc(), Runbook.id)
        .limit(limit)
    )
    if service is not None:
        stmt = stmt.where(or_(Runbook.service == service, Runbook.service.is_(None)))
    rows = (await session.execute(stmt)).all()
    return [
        RunbookHit(
            id=r.id,
            title=r.title,
            service=r.service,
            tags=list(r.tags),
            rank=float(r.rank),
            snippet=r.snippet,
        )
        for r in rows
    ]
