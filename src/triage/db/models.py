"""ORM models. The schema itself is owned by Alembic migrations (see migrations/)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from triage.db.base import Base

SEVERITIES = ("critical", "high", "medium", "low", "info")
INCIDENT_STATUSES = ("open", "actioned")
PROPOSAL_STATUSES = ("pending_review", "approved", "rejected")
RUN_STATUSES = ("running", "succeeded", "failed")


def _in(column: str, values: tuple[str, ...]) -> str:
    quoted = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({quoted})"


class Incident(Base):
    __tablename__ = "incidents"
    __table_args__ = (
        CheckConstraint(_in("severity", SEVERITIES), name="severity"),
        CheckConstraint(_in("status", INCIDENT_STATUSES), name="status"),
        Index("ix_incidents_service_status_last_seen", "service", "status", "last_seen_at"),
        Index(
            "ix_incidents_title_trgm",
            "title",
            postgresql_using="gin",
            postgresql_ops={"title": "gin_trgm_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    service: Mapped[str] = mapped_column(String(100))
    title: Mapped[str] = mapped_column(String(500))
    severity: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), server_default="open")
    fingerprint: Mapped[str] = mapped_column(String(200))
    alert_count: Mapped[int] = mapped_column(Integer, server_default="0")
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    alerts: Mapped[list[Alert]] = relationship(
        back_populates="incident", order_by="Alert.received_at"
    )
    proposals: Mapped[list[TriageProposal]] = relationship(
        back_populates="incident", order_by="TriageProposal.created_at"
    )


class Alert(Base):
    __tablename__ = "alerts"
    __table_args__ = (
        CheckConstraint(_in("severity", SEVERITIES), name="severity"),
        Index("ix_alerts_fingerprint_last_received", "fingerprint", "last_received_at"),
        Index("ix_alerts_incident_id", "incident_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"))
    service: Mapped[str] = mapped_column(String(100))
    severity: Mapped[str] = mapped_column(String(16))
    title: Mapped[str] = mapped_column(String(500))
    fingerprint: Mapped[str] = mapped_column(String(200))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    duplicate_count: Mapped[int] = mapped_column(Integer, server_default="0")
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    incident: Mapped[Incident] = relationship(back_populates="alerts")


class Runbook(Base):
    __tablename__ = "runbooks"
    __table_args__ = (
        Index("ix_runbooks_search_vector", "search_vector", postgresql_using="gin"),
        Index("ix_runbooks_service", "service"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(300))
    service: Mapped[str | None] = mapped_column(String(100), nullable=True)
    body: Mapped[str] = mapped_column(Text)
    tags: Mapped[list[str]] = mapped_column(ARRAY(String(50)), server_default="{}")
    search_vector: Mapped[str] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', coalesce(title, '')), 'A') || "
            "setweight(to_tsvector('english', triage_tags_to_text(tags)), 'B') || "
            "setweight(to_tsvector('english', coalesce(body, '')), 'C')",
            persisted=True,
        ),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AgentRun(Base):
    __tablename__ = "agent_runs"
    __table_args__ = (
        CheckConstraint(_in("status", RUN_STATUSES), name="status"),
        Index("ix_agent_runs_incident_id", "incident_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"))
    model: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(16), server_default="running")
    termination_reason: Mapped[str | None] = mapped_column(String(50), nullable=True)
    last_stop_reason: Mapped[str | None] = mapped_column(String(50), nullable=True)
    iterations: Mapped[int] = mapped_column(Integer, server_default="0")
    input_tokens: Mapped[int] = mapped_column(Integer, server_default="0")
    output_tokens: Mapped[int] = mapped_column(Integer, server_default="0")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    tool_calls: Mapped[list[AgentToolCall]] = relationship(
        back_populates="run", order_by="AgentToolCall.id"
    )
    llm_calls: Mapped[list[AgentLLMCall]] = relationship(
        back_populates="run", order_by="AgentLLMCall.id"
    )


class AgentLLMCall(Base):
    """One Messages API request/response within an agent run."""

    __tablename__ = "agent_llm_calls"
    __table_args__ = (Index("ix_agent_llm_calls_run_id", "run_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"))
    iteration: Mapped[int] = mapped_column(Integer)
    stop_reason: Mapped[str | None] = mapped_column(String(50), nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    attempts: Mapped[int] = mapped_column(Integer, server_default="1")
    latency_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    run: Mapped[AgentRun] = relationship(back_populates="llm_calls")


class AgentToolCall(Base):
    __tablename__ = "agent_tool_calls"
    __table_args__ = (Index("ix_agent_tool_calls_run_id", "run_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"))
    iteration: Mapped[int] = mapped_column(Integer)
    tool_use_id: Mapped[str] = mapped_column(String(100))
    tool_name: Mapped[str] = mapped_column(String(100))
    input: Mapped[dict[str, Any]] = mapped_column(JSONB)
    output_chars: Mapped[int] = mapped_column(Integer)
    is_error: Mapped[bool] = mapped_column(Boolean, server_default="false")
    latency_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    run: Mapped[AgentRun] = relationship(back_populates="tool_calls")


class TriageProposal(Base):
    __tablename__ = "triage_proposals"
    __table_args__ = (
        CheckConstraint(_in("status", PROPOSAL_STATUSES), name="status"),
        CheckConstraint(_in("severity_level", SEVERITIES), name="severity_level"),
        Index("ix_triage_proposals_incident_id", "incident_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    incident_id: Mapped[int] = mapped_column(ForeignKey("incidents.id", ondelete="CASCADE"))
    agent_run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(16), server_default="pending_review")
    summary: Mapped[str] = mapped_column(Text)
    probable_cause: Mapped[str] = mapped_column(Text)
    severity_level: Mapped[str] = mapped_column(String(16))
    severity_rationale: Mapped[str] = mapped_column(Text)
    confidence: Mapped[str] = mapped_column(String(16))
    next_steps: Mapped[list[str]] = mapped_column(JSONB)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    rejected_citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, server_default="[]")
    flags: Mapped[list[str]] = mapped_column(JSONB, server_default="[]")
    reviewer: Mapped[str | None] = mapped_column(String(100), nullable=True)
    review_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    incident: Mapped[Incident] = relationship(back_populates="proposals")
