"""Initial schema: incidents, alerts, runbooks (FTS), agent audit tables, triage proposals.

Revision ID: 0001
Revises:
Create Date: 2026-10-29 21:22:20.095003
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # pg_trgm backs find_similar_incidents (trigram similarity on incident titles).
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    # array_to_string() is only STABLE, so it cannot appear in a generated column. Tags are a
    # plain text[], for which joining is genuinely immutable, so wrap it in an IMMUTABLE function.
    op.execute(
        "CREATE FUNCTION triage_tags_to_text(text[]) RETURNS text "
        "LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$ SELECT array_to_string($1, ' ') $$"
    )
    op.create_table(
        "incidents",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("service", sa.String(length=100), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="open", nullable=False),
        sa.Column("fingerprint", sa.String(length=200), nullable=False),
        sa.Column("alert_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "severity IN ('critical', 'high', 'medium', 'low', 'info')",
            name=op.f("ck_incidents_severity"),
        ),
        sa.CheckConstraint("status IN ('open', 'actioned')", name=op.f("ck_incidents_status")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_incidents")),
    )
    op.create_index(
        "ix_incidents_service_status_last_seen",
        "incidents",
        ["service", "status", "last_seen_at"],
        unique=False,
    )
    op.create_index(
        "ix_incidents_title_trgm",
        "incidents",
        ["title"],
        unique=False,
        postgresql_using="gin",
        postgresql_ops={"title": "gin_trgm_ops"},
    )
    op.create_table(
        "runbooks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("service", sa.String(length=100), nullable=True),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column(
            "tags", postgresql.ARRAY(sa.String(length=50)), server_default="{}", nullable=False
        ),
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed(
                "setweight(to_tsvector('english', coalesce(title, '')), 'A') || "
                "setweight(to_tsvector('english', triage_tags_to_text(tags)), 'B') || "
                "setweight(to_tsvector('english', coalesce(body, '')), 'C')",
                persisted=True,
            ),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_runbooks")),
    )
    op.create_index(
        "ix_runbooks_search_vector",
        "runbooks",
        ["search_vector"],
        unique=False,
        postgresql_using="gin",
    )
    op.create_index("ix_runbooks_service", "runbooks", ["service"], unique=False)
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("incident_id", sa.BigInteger(), nullable=False),
        sa.Column("model", sa.String(length=100), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="running", nullable=False),
        sa.Column("termination_reason", sa.String(length=50), nullable=True),
        sa.Column("last_stop_reason", sa.String(length=50), nullable=True),
        sa.Column("iterations", sa.Integer(), server_default="0", nullable=False),
        sa.Column("input_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("output_tokens", sa.Integer(), server_default="0", nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('running', 'succeeded', 'failed')", name=op.f("ck_agent_runs_status")
        ),
        sa.ForeignKeyConstraint(
            ["incident_id"],
            ["incidents.id"],
            name=op.f("fk_agent_runs_incident_id_incidents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_runs")),
    )
    op.create_index("ix_agent_runs_incident_id", "agent_runs", ["incident_id"], unique=False)
    op.create_table(
        "alerts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("incident_id", sa.BigInteger(), nullable=False),
        sa.Column("service", sa.String(length=100), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("title", sa.String(length=500), nullable=False),
        sa.Column("fingerprint", sa.String(length=200), nullable=False),
        sa.Column(
            "payload", postgresql.JSONB(astext_type=sa.Text()), server_default="{}", nullable=False
        ),
        sa.Column("duplicate_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_received_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "severity IN ('critical', 'high', 'medium', 'low', 'info')",
            name=op.f("ck_alerts_severity"),
        ),
        sa.ForeignKeyConstraint(
            ["incident_id"],
            ["incidents.id"],
            name=op.f("fk_alerts_incident_id_incidents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_alerts")),
    )
    op.create_index(
        "ix_alerts_fingerprint_last_received",
        "alerts",
        ["fingerprint", "last_received_at"],
        unique=False,
    )
    op.create_index("ix_alerts_incident_id", "alerts", ["incident_id"], unique=False)
    op.create_table(
        "agent_llm_calls",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("iteration", sa.Integer(), nullable=False),
        sa.Column("stop_reason", sa.String(length=50), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("attempts", sa.Integer(), server_default="1", nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["agent_runs.id"],
            name=op.f("fk_agent_llm_calls_run_id_agent_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_llm_calls")),
    )
    op.create_index("ix_agent_llm_calls_run_id", "agent_llm_calls", ["run_id"], unique=False)
    op.create_table(
        "agent_tool_calls",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("iteration", sa.Integer(), nullable=False),
        sa.Column("tool_use_id", sa.String(length=100), nullable=False),
        sa.Column("tool_name", sa.String(length=100), nullable=False),
        sa.Column("input", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("output_chars", sa.Integer(), nullable=False),
        sa.Column("is_error", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["agent_runs.id"],
            name=op.f("fk_agent_tool_calls_run_id_agent_runs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_agent_tool_calls")),
    )
    op.create_index("ix_agent_tool_calls_run_id", "agent_tool_calls", ["run_id"], unique=False)
    op.create_table(
        "triage_proposals",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("incident_id", sa.BigInteger(), nullable=False),
        sa.Column("agent_run_id", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="pending_review", nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("probable_cause", sa.Text(), nullable=False),
        sa.Column("severity_level", sa.String(length=16), nullable=False),
        sa.Column("severity_rationale", sa.Text(), nullable=False),
        sa.Column("confidence", sa.String(length=16), nullable=False),
        sa.Column("next_steps", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("citations", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "rejected_citations",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="[]",
            nullable=False,
        ),
        sa.Column(
            "flags", postgresql.JSONB(astext_type=sa.Text()), server_default="[]", nullable=False
        ),
        sa.Column("reviewer", sa.String(length=100), nullable=True),
        sa.Column("review_note", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "severity_level IN ('critical', 'high', 'medium', 'low', 'info')",
            name=op.f("ck_triage_proposals_severity_level"),
        ),
        sa.CheckConstraint(
            "status IN ('pending_review', 'approved', 'rejected')",
            name=op.f("ck_triage_proposals_status"),
        ),
        sa.ForeignKeyConstraint(
            ["agent_run_id"],
            ["agent_runs.id"],
            name=op.f("fk_triage_proposals_agent_run_id_agent_runs"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["incident_id"],
            ["incidents.id"],
            name=op.f("fk_triage_proposals_incident_id_incidents"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_triage_proposals")),
    )
    op.create_index(
        "ix_triage_proposals_incident_id", "triage_proposals", ["incident_id"], unique=False
    )


def downgrade() -> None:
    # The pg_trgm extension is intentionally left installed: it is database-wide and may be
    # used by other schemas. Everything this revision created is dropped.
    op.drop_index("ix_triage_proposals_incident_id", table_name="triage_proposals")
    op.drop_table("triage_proposals")
    op.drop_index("ix_agent_tool_calls_run_id", table_name="agent_tool_calls")
    op.drop_table("agent_tool_calls")
    op.drop_index("ix_agent_llm_calls_run_id", table_name="agent_llm_calls")
    op.drop_table("agent_llm_calls")
    op.drop_index("ix_alerts_incident_id", table_name="alerts")
    op.drop_index("ix_alerts_fingerprint_last_received", table_name="alerts")
    op.drop_table("alerts")
    op.drop_index("ix_agent_runs_incident_id", table_name="agent_runs")
    op.drop_table("agent_runs")
    op.drop_index("ix_runbooks_service", table_name="runbooks")
    op.drop_index("ix_runbooks_search_vector", table_name="runbooks", postgresql_using="gin")
    op.drop_table("runbooks")
    op.drop_index(
        "ix_incidents_title_trgm",
        table_name="incidents",
        postgresql_using="gin",
        postgresql_ops={"title": "gin_trgm_ops"},
    )
    op.drop_index("ix_incidents_service_status_last_seen", table_name="incidents")
    op.drop_table("incidents")
    op.execute("DROP FUNCTION triage_tags_to_text(text[])")
