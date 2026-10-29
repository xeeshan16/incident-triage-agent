"""Test fixtures. Tests run against a real PostgreSQL database (TEST_DATABASE_URL).

The schema is created once per session by running the Alembic migrations (so migrations are
exercised on every run), and every test starts from truncated tables.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from alembic import command
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from triage.agent.fake import ScriptedLLM
from triage.agent.loop import AgentConfig, TriageAgent
from triage.config import Settings
from triage.db.base import SessionFactory, make_engine, make_session_factory, transaction
from triage.main import create_app
from triage.schemas.api import AlertIn
from triage.services.alerts import ingest_alert

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://triage:triage@localhost:5432/triage_test"
)
TABLES = (
    "triage_proposals",
    "agent_tool_calls",
    "agent_llm_calls",
    "agent_runs",
    "alerts",
    "incidents",
    "runbooks",
)


def _alembic_config() -> Config:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = Config(os.path.join(root, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(root, "migrations"))
    cfg.attributes["database_url"] = TEST_DATABASE_URL
    cfg.attributes["configure_logger"] = False
    return cfg


@pytest.fixture(scope="session")
def migrated() -> None:
    cfg = _alembic_config()
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")


@pytest.fixture
async def engine(migrated: None) -> AsyncIterator[AsyncEngine]:
    eng = make_engine(TEST_DATABASE_URL, pool_size=5, max_overflow=5)
    async with eng.begin() as conn:
        await conn.execute(text(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE"))
    yield eng
    await eng.dispose()


@pytest.fixture
def sessions(engine: AsyncEngine) -> SessionFactory:
    return make_session_factory(engine)


@pytest.fixture
async def session(sessions: SessionFactory) -> AsyncIterator[AsyncSession]:
    async with sessions() as s:
        yield s


@pytest.fixture
def settings() -> Settings:
    return Settings(
        database_url=TEST_DATABASE_URL,
        anthropic_api_key=None,
        log_json=True,
        log_level="WARNING",
    )


@pytest.fixture
def agent_config() -> AgentConfig:
    return AgentConfig(
        model="test-model",
        max_iterations=6,
        token_budget=20_000,
        timeout_seconds=5.0,
        max_tokens_per_call=1_024,
        tool_output_max_chars=12_000,
    )


@pytest.fixture
def make_agent(sessions: SessionFactory, agent_config: AgentConfig) -> Any:
    def _make(llm: ScriptedLLM, config: AgentConfig | None = None) -> TriageAgent:
        return TriageAgent(llm, sessions, config or agent_config)

    return _make


@pytest.fixture
def scripted() -> ScriptedLLM:
    return ScriptedLLM(steps=[])


@pytest.fixture
async def client(
    settings: Settings, engine: AsyncEngine, scripted: ScriptedLLM
) -> AsyncIterator[AsyncClient]:
    app = create_app(settings, llm=scripted)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


# Relative to wall-clock so 'last N days' tool windows behave; fixed for the whole session.
NOW = datetime.now(UTC).replace(microsecond=0) - timedelta(hours=1)


async def add_alert(
    sessions: SessionFactory,
    *,
    service: str = "payments-api",
    severity: str = "high",
    title: str = "Postgres connection pool exhausted",
    fingerprint: str = "pg-pool-exhausted",
    payload: dict[str, Any] | None = None,
    at: datetime | None = None,
) -> tuple[int, int]:
    """Ingest one alert through the real service; returns (alert_id, incident_id)."""
    data = AlertIn(
        service=service,
        severity=severity,
        title=title,
        fingerprint=fingerprint,
        payload=payload or {},
        received_at=at or NOW,
    )
    async with transaction(sessions) as s:
        out = await ingest_alert(
            s, data, dedup_window=timedelta(minutes=10), group_window=timedelta(minutes=30)
        )
        return out.alert.id, out.incident.id
