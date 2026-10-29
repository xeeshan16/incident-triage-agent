"""Domain errors raised by the service layer and mapped to HTTP responses in triage.api."""

from __future__ import annotations


class DomainError(Exception):
    """Base class for expected, user-facing errors."""


class NotFoundError(DomainError):
    pass


class ConflictError(DomainError):
    pass


class AgentRunFailedError(DomainError):
    """The agent run finished without producing an acceptable proposal."""

    def __init__(self, *, run_id: int, reason: str, detail: str) -> None:
        super().__init__(f"agent run {run_id} failed: {reason}")
        self.run_id = run_id
        self.reason = reason
        self.detail = detail
