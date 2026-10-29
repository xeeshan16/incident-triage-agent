"""FastAPI dependencies. App-scoped resources live on ``app.state`` (created in lifespan)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from triage.agent.loop import TriageAgent
from triage.config import Settings
from triage.db.base import SessionFactory


def get_settings_dep(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_session_factory(request: Request) -> SessionFactory:
    factory: SessionFactory = request.app.state.session_factory
    return factory


async def get_tx_session(
    factory: Annotated[SessionFactory, Depends(get_session_factory)],
) -> AsyncIterator[AsyncSession]:
    """One transaction per request: committed if the handler returns, rolled back if it raises."""
    async with factory() as session, session.begin():
        yield session


def get_agent(request: Request) -> TriageAgent:
    agent: TriageAgent = request.app.state.agent
    return agent


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
SessionDep = Annotated[AsyncSession, Depends(get_tx_session)]
SessionFactoryDep = Annotated[SessionFactory, Depends(get_session_factory)]
AgentDep = Annotated[TriageAgent, Depends(get_agent)]
