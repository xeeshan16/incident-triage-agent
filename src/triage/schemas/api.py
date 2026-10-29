"""Request/response models for the HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Severity = Literal["critical", "high", "medium", "low", "info"]
IncidentStatus = Literal["open", "actioned"]
ProposalStatus = Literal["pending_review", "approved", "rejected"]

_MAX_PAYLOAD_BYTES = 32_768


class AlertIn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    service: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9._\-]+$")
    severity: Severity
    title: str = Field(min_length=1, max_length=500)
    fingerprint: str = Field(min_length=1, max_length=200)
    payload: dict[str, Any] = Field(default_factory=dict)
    received_at: datetime | None = Field(
        default=None, description="Defaults to server time. Must be timezone-aware if given."
    )

    @field_validator("payload")
    @classmethod
    def _payload_size(cls, v: dict[str, Any]) -> dict[str, Any]:
        import json

        if len(json.dumps(v, default=str)) > _MAX_PAYLOAD_BYTES:
            raise ValueError(f"payload exceeds {_MAX_PAYLOAD_BYTES} bytes when serialised")
        return v

    @field_validator("received_at")
    @classmethod
    def _aware(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            raise ValueError("received_at must include a timezone offset")
        return v


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    incident_id: int
    service: str
    severity: Severity
    title: str
    fingerprint: str
    payload: dict[str, Any]
    duplicate_count: int
    received_at: datetime
    last_received_at: datetime


class AlertIngestResult(BaseModel):
    alert: AlertOut
    incident_id: int
    deduplicated: bool = Field(
        description="True if this alert matched an existing alert's fingerprint within the "
        "dedup window and only bumped its duplicate counter."
    )
    new_incident: bool


class IncidentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    service: str
    title: str
    severity: Severity
    status: IncidentStatus
    fingerprint: str
    alert_count: int
    first_seen_at: datetime
    last_seen_at: datetime


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["alert", "runbook", "incident"]
    id: int


class TriageProposalOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    incident_id: int
    agent_run_id: int
    status: ProposalStatus
    summary: str
    probable_cause: str
    severity_level: Severity
    severity_rationale: str
    confidence: str
    next_steps: list[str]
    citations: list[Citation]
    rejected_citations: list[Citation]
    flags: list[str]
    reviewer: str | None
    review_note: str | None
    reviewed_at: datetime | None
    created_at: datetime


class IncidentDetail(IncidentSummary):
    alerts: list[AlertOut]
    proposals: list[TriageProposalOut]


class ReviewIn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    reviewer: str = Field(min_length=1, max_length=100)
    note: str = Field(default="", max_length=4_000)


class RunbookIn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=300)
    service: str | None = Field(default=None, max_length=100)
    body: str = Field(min_length=1, max_length=50_000)
    tags: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("tags")
    @classmethod
    def _tags(cls, v: list[str]) -> list[str]:
        cleaned = [t.strip().lower() for t in v if t.strip()]
        if any(len(t) > 50 for t in cleaned):
            raise ValueError("tags must be at most 50 characters")
        return cleaned


class RunbookOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    service: str | None
    body: str
    tags: list[str]
    created_at: datetime


class RunbookSearchHit(BaseModel):
    id: int
    title: str
    service: str | None
    tags: list[str]
    rank: float
    snippet: str


class Page[T](BaseModel):
    items: list[T]
    limit: int
    offset: int
