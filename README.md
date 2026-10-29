# incident-triage-agent

An on-call triage service. Alerts arrive over HTTP and are stored in PostgreSQL, deduplicated and
grouped into incidents. On request, a Claude agent investigates an incident with read-only tools
against the database and returns a structured triage proposal with citations. A human approves
or rejects the proposal; the agent never takes action itself.

The scope is deliberately small: about 2,700 lines of application code, with the core (agent
loop, tools and guardrails in `src/triage/agent/`) readable in about 30 minutes. Tests and evals
add roughly another 2,400 lines.

## Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Agent loop and guardrails](#agent-loop-and-guardrails)
- [API examples](#api-examples)
- [Running locally](#running-locally)
- [Testing](#testing)
- [Evals](#evals)
- [Configuration](#configuration)
- [Design decisions and trade-offs](#design-decisions-and-trade-offs)
- [Limitations](#limitations)

## What it does

- **Ingest** (`POST /alerts`): alerts carry `service`, `severity`, `title`, `fingerprint` and a
  JSON `payload`.
  - *Dedup*: an alert whose fingerprint matches an alert on an open incident, received within
    `DEDUP_WINDOW_SECONDS` (default 10 min), only increments that alert's `duplicate_count`.
    The window slides, so a steadily flapping alert stays one row.
  - *Grouping*: otherwise the alert joins the most recently active open incident for the same
    service within `GROUP_WINDOW_SECONDS` (default 30 min), or opens a new incident. Incident
    severity is the highest severity among its alerts.
  - Concurrent ingests for one service are serialised with a transaction-scoped advisory lock,
    so simultaneous identical alerts cannot open two incidents (there is a test for this).
- **Triage** (`POST /incidents/{id}/triage`): runs the agent and stores a proposal with status
  `pending_review`: summary, probable cause, severity assessment and rationale, confidence,
  ordered next steps, and citations (`{kind: alert|runbook|incident, id}`).
- **Review** (`POST /triage/{id}/approve|reject`): a reviewer name and note are recorded.
  Approval marks the incident `actioned`. This is the only way an incident changes state.
- **Runbooks** (`POST /runbooks`, `GET /runbooks?q=...`): stored in Postgres and searched with
  full-text search (weighted `tsvector`, GIN index, `ts_rank_cd`, `ts_headline` snippets).
- **Audit trail**: every agent run, every model call (stop reason, tokens, attempts, latency) and
  every tool call (name, input, output size, error flag, latency) is persisted.
- **Ops**: `GET /healthz` (liveness), `GET /readyz` (database check, 2 s bound), JSON logs with a
  request id (`X-Request-ID` is accepted or generated, and echoed).

## Architecture

```mermaid
flowchart LR
    subgraph Clients
        MON[Monitoring / alert source]
        ONC[On-call engineer]
    end

    subgraph API["FastAPI (src/triage/api)"]
        R[routes.py]
    end

    subgraph Services["services/"]
        ING[alerts: dedup + grouping]
        RB[runbooks: FTS]
        REV[review: approve / reject]
    end

    subgraph Agent["agent/"]
        LOOP[loop.py: manual tool-use loop + guardrails]
        TOOLS[tools.py: read-only tools + submit_triage]
        LLM[llm.py: LLMClient protocol, retries]
        AUD[audit.py]
    end

    PG[(PostgreSQL 16<br/>pg_trgm, tsvector/GIN)]
    CLAUDE[Claude Messages API]

    MON -- POST /alerts --> R --> ING --> PG
    ONC -- POST /incidents/id/triage --> R --> LOOP
    LOOP <--> LLM <--> CLAUDE
    LOOP --> TOOLS --> PG
    TOOLS --> RB
    LOOP --> AUD --> PG
    LOOP -- proposal: pending_review --> PG
    ONC -- POST /triage/id/approve --> R --> REV --> PG
```

Layering: `api` (HTTP only) calls `services` (domain logic, no HTTP) and `agent`; both use `db`
(SQLAlchemy 2 async models, Alembic migrations) and `schemas` (Pydantic v2). The agent depends on
an `LLMClient` protocol, not on the SDK, so tests and evals use a scripted fake.

Tables: `incidents`, `alerts`, `runbooks`, `agent_runs`, `agent_llm_calls`, `agent_tool_calls`,
`triage_proposals`. See `migrations/versions/0001_initial_schema.py`.

## Agent loop and guardrails

The loop itself (`_loop` in `src/triage/agent/loop.py`) is under 100 lines, written directly
against the Messages API with tool use rather than a framework; the rest of the module is
submission judging, run finalisation and the audit trail.

```
create agent_run (status=running)
for iteration in 1..max_iterations:
    response = llm.create(system, messages, tools)        # retried on 429/5xx/overloaded
    record usage + stop_reason
    refusal / max_tokens / unexpected stop_reason  -> stop (failed)
    if response calls submit_triage:
        validate input with Pydantic                     -> on error: tool_result is_error, continue
        check citations against ids seen in earlier turns
            all known                                     -> accept
            unknown ids, first time                       -> is_error, ask to resubmit
            unknown ids, again                            -> strip + flag, or fail if none left
    if tokens used > budget                               -> stop (failed)
    run read-only tools (each in its own short transaction), audit each call
    send all tool_results back in one user turn (+ nudge on the second-to-last turn)
stop (failed: max_iterations)
```

The run is wrapped in `asyncio.timeout(AGENT_TIMEOUT_SECONDS)`, and every exit path (including
cancellation) finalises the `agent_runs` row.

**Tools** (all read-only; inputs validated with Pydantic, `extra="forbid"`, strict types; invalid
input becomes a structured `is_error` tool result, never an exception):

| Tool | What it returns |
| --- | --- |
| `get_incident_alerts` | The incident's alerts with wrapped payloads |
| `search_runbooks` | Runbook FTS hits with rank and snippet |
| `find_similar_incidents` | Same-service incidents with the same fingerprint or similar title (`pg_trgm`), plus their approved triage summary |
| `get_service_recent_history` | The service's incidents in the last N days |
| `submit_triage` | Terminal tool: its input schema *is* the proposal |

**Guardrails** (each has tests in `tests/test_agent_loop.py` or `tests/test_review_findings.py`):

| Guardrail | Mechanism |
| --- | --- |
| Iteration cap | `AGENT_MAX_ITERATIONS`; on the second-to-last turn the model is told to submit |
| Token budget | Cumulative `input + cache + output` tokens from each response's `usage`, checked every turn |
| Timeout | `asyncio.timeout` over the whole run, plus a per-request HTTP timeout in the SDK |
| Retries | `RetryingClient`: exponential backoff with full jitter, honours `retry-after`, capped; 429, 408, 409, 5xx, 529 and connection errors only. SDK retries are disabled so this is the one place retries happen, and attempts are recorded per call |
| Structured output | Run only succeeds via `submit_triage`; its schema is sent with `strict: true` and the input is re-validated with Pydantic, which has the length and count limits strict mode cannot express |
| Citation provenance | Every cited id must have been returned by a tool **in an earlier turn of this run**. An id that exists in the database but was never shown is still rejected, as is a citation made in the same turn as the tool call that would return it |
| Prompt injection | See below |
| Propose-only | No tool can write. Only the review endpoints change incident state |

**Prompt injection.** Alert titles and payloads come from whatever emits alerts, so they are
treated as attacker-controlled; runbook text and past summaries are treated the same way. In
order of how much the design relies on them:

1. *Capability*: the agent has no write tools. A successful injection can at worst produce a
   misleading proposal, which a human must still approve.
2. *Output validation*: schema validation plus citation provenance. A test has a scripted model
   obey a malicious payload by citing a runbook it never saw; the run fails with
   `invalid_citations` and nothing is stored.
3. *Delimiting*: untrusted text only reaches the model inside `tool_result` blocks, inside
   `<untrusted_alert_payload>` / `<untrusted_text>` tags, with `<` and `>` escaped so the
   content cannot close the tag. Oversized output is truncated by dropping whole items, never by
   cutting through a tag.
4. *Instruction*: the system prompt says tagged content is data and asks the model to report
   injection attempts (`suspected_prompt_injection`).
5. *Detection*: a regex heuristic flags suspicious alerts on the proposal
   (`prompt_injection_heuristic_match`) so the reviewer sees them.

Points 3 to 5 reduce risk; they do not eliminate it. Points 1 and 2 are the guarantees.

**Model choice and `tool_choice`.** The model defaults to `claude-sonnet-5-5` (override with
`TRIAGE_MODEL`). Forcing a tool (`tool_choice` of `any` or `tool`) returns a 400 on Claude Sonnet 5.5
and Opus 5.5, so the loop uses `auto`, as the API docs recommend for those models, and relies on
strict tool schemas, the system prompt and nudges to get a `submit_triage` call.

## API examples

```bash
# Ingest an alert (201 = stored, 200 = deduplicated into an existing alert)
curl -s -X POST localhost:8000/alerts -H 'content-type: application/json' -d '{
  "service": "payments-api",
  "severity": "high",
  "title": "Postgres connection pool exhausted",
  "fingerprint": "payments-api:pg-pool-exhausted",
  "payload": {"pool_max": 20, "waiting": 143, "db": "payments-primary"}
}'

# List open incidents, then fetch one with its alerts and proposals
curl -s 'localhost:8000/incidents?status=open&limit=20'
curl -s localhost:8000/incidents/1

# Run the agent (needs ANTHROPIC_API_KEY). 201 with the proposal, or 502 with the run id.
curl -s -X POST localhost:8000/incidents/1/triage

# Human review
curl -s -X POST localhost:8000/triage/1/approve -H 'content-type: application/json' \
  -d '{"reviewer": "zeeshan", "note": "Matches what I see in pg_stat_activity."}'
curl -s -X POST localhost:8000/triage/1/reject -H 'content-type: application/json' \
  -d '{"reviewer": "zeeshan", "note": "Cause is the deploy, not the pool."}'

# Runbooks: create, list, full-text search
curl -s -X POST localhost:8000/runbooks -H 'content-type: application/json' -d '{
  "title": "Postgres connection pool exhaustion", "tags": ["postgres", "pool"],
  "body": "Check pg_stat_activity for idle in transaction sessions..."
}'
curl -s localhost:8000/runbooks
curl -s 'localhost:8000/runbooks?q=connection%20pool%20exhausted'

curl -s localhost:8000/healthz
curl -s localhost:8000/readyz
```

A triage proposal looks like this (abridged):

```json
{
  "id": 1, "incident_id": 1, "agent_run_id": 1, "status": "pending_review",
  "summary": "...", "probable_cause": "...",
  "severity_level": "high", "severity_rationale": "...", "confidence": "medium",
  "next_steps": ["Check pg_stat_activity for idle-in-transaction sessions", "..."],
  "citations": [{"kind": "alert", "id": 3}, {"kind": "runbook", "id": 1}],
  "rejected_citations": [],
  "flags": []
}
```

Interactive OpenAPI docs are at `/docs`.

## Running locally

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/) for native runs, or Docker.

### Docker Compose

```bash
cp .env.example .env              # optionally set ANTHROPIC_API_KEY
docker compose up --build -d      # postgres -> migrate (one-shot) -> api
docker compose exec api python -m scripts.seed
curl -s localhost:8000/readyz
```

The image is multi-stage, installs from `uv.lock`, and runs as a non-root user (uid 10001).

### Native

```bash
uv sync --locked                  # creates .venv with runtime + dev dependencies
cp .env.example .env              # point DATABASE_URL at a Postgres 16 with pg_trgm available
make migrate
make seed
make run                          # uvicorn on :8000 with reload
```

Without `ANTHROPIC_API_KEY` everything works except triage, which returns 502 and records the
run as failed with a clear error.

## Testing

Tests run against a real PostgreSQL database (`TEST_DATABASE_URL`, default
`postgresql+asyncpg://triage:triage@localhost:5432/triage_test`). The schema is created by
running the Alembic migrations, and tables are truncated before each test. **Point it at a
throwaway database.**

```bash
make lint          # ruff check + ruff format --check
make typecheck     # mypy (strict) over src, tests, evals, scripts
make test          # pytest
make cov           # pytest with branch coverage
make check         # all of the above plus evals
```

What the tests cover: ingest dedup/grouping (including concurrency and out-of-order alerts),
each tool against real FTS and pg_trgm queries, every guardrail in the agent loop, the HTTP API
including error mapping, the Anthropic adapter through the real SDK with a mock HTTP transport,
and one regression test per review finding. The model is never called.

## Evals

`evals/run_evals.py` runs scripted scenarios through the real agent loop and a real database,
and checks guardrail behaviour and citation validity:

| Scenario | Expected |
| --- | --- |
| `db_connection_exhaustion` | Succeeds; cites only alerts from the incident and searched runbooks |
| `bad_deploy_with_transient_errors` | API overloaded / 500 mid-run; retries recover; succeeds |
| `noisy_flapping_alert` | Six flapping alerts dedup to one; model wanders; stops at the iteration cap |
| `prompt_injection_obeyed` | Model follows injected instructions; fabricated citation rejected; no proposal |
| `prompt_injection_resisted` | Model reports the injection; proposal flagged; severity not downgraded |
| `runaway_token_usage` | Stops at the token budget |

Every scenario also checks invariants that must hold for any model: the run is finalised, the
incident is not actioned, a new proposal is `pending_review`, and every cited alert belongs to
the incident.

```bash
make eval                                  # offline, scripted model
python -m evals.run_evals --only noisy_flapping_alert
ANTHROPIC_API_KEY=... python -m evals.run_evals --live
```

With `--live` the same scenarios are seeded but the real model drives the run, and only the
model-independent invariants are checked (plus, for the injection scenarios, that a resulting
proposal is flagged and not downgraded to `info`). Live results are non-deterministic and cost
tokens. The evals truncate the database at `EVAL_DATABASE_URL` (falling back to
`TEST_DATABASE_URL`).

These evals test this codebase's handling of model behaviour. They do not measure the quality of
the model's triage; that would need a labelled set of real incidents.

## Configuration

All settings are environment variables (or `.env`); see `.env.example`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql+asyncpg://triage:triage@localhost:5432/triage` | Application database |
| `ANTHROPIC_API_KEY` | unset | Required for triage only |
| `TRIAGE_MODEL` | `claude-sonnet-5-5` | Model id |
| `AGENT_MAX_ITERATIONS` | `8` | Model calls per run |
| `AGENT_TOKEN_BUDGET` | `60000` | Input + output tokens per run |
| `AGENT_TIMEOUT_SECONDS` | `90` | Wall clock per run |
| `AGENT_MAX_TOKENS_PER_CALL` | `2048` | `max_tokens` per model call |
| `TOOL_OUTPUT_MAX_CHARS` | `12000` | Cap on each tool result sent to the model |
| `LLM_REQUEST_TIMEOUT_SECONDS` | `60` | Per HTTP request to the API |
| `LLM_MAX_RETRIES` | `3` | Retries on transient errors |
| `LLM_BACKOFF_BASE_SECONDS` / `LLM_BACKOFF_MAX_SECONDS` | `0.5` / `8` | Backoff shape |
| `DEDUP_WINDOW_SECONDS` | `600` | Fingerprint dedup window |
| `GROUP_WINDOW_SECONDS` | `1800` | Incident grouping window |
| `LOG_LEVEL` / `LOG_JSON` | `INFO` / `true` | Logging |

## Design decisions and trade-offs

**Postgres full-text search, not a vector database.** Runbooks are few, short and full of exact
terms (`pg_stat_activity`, `idle in transaction`, service names), where lexical search does well.
FTS needs no embedding model, no second datastore and no sync job, and its ranking is
explainable. The cost is weaker recall for paraphrases ("DB out of connections" against "pool
exhaustion"). The search uses OR semantics over the query's words so that a handful of symptom
words still matches. If recall became a problem, `pgvector` in the same database would be the
next step, not a separate vector store.

**A manual loop, not an agent framework.** The loop is small, and the guardrails are the point
of the project: where the token budget is checked relative to a submission, which turn's
provenance a citation is judged against, what happens to parallel tool calls in one turn. Owning
the loop makes those decisions visible and testable line by line. The cost is re-implementing
things a framework provides (retries, tool dispatch), and having to track API changes directly.

**Propose-only.** Triage is the step where a model is most useful (correlating alerts, recalling
runbooks and similar incidents) and actions are where a mistake is most expensive. Keeping the
agent read-only bounds the damage from hallucination or prompt injection to a bad proposal that
a human must still approve, and it makes the system easy to reason about. The cost is that a
human is always in the loop, even for obvious cases.

**Citations checked against this run's tool output.** Requiring citations to exist in the
database would not catch a model guessing a plausible id. Requiring that the id was *shown* to
the model in this run does, and it is cheap to check. One correction round is allowed, because
models do occasionally mis-copy an id.

**Short transactions in the agent.** Each tool call and audit write uses its own transaction, and
none is held open across a model call, which can take tens of seconds. A connection is never
held while waiting on the API. The cost is that the agent can see data change mid-run, which is
acceptable for triage.

**Retries in our code, not the SDK.** The SDK can retry on its own, but then attempts are
invisible. Doing it in `RetryingClient` (with SDK retries off) means one retry policy, recorded
per call in `agent_llm_calls.attempts`, and testable without sleeping.

**Synchronous triage endpoint.** Simple to use and test. See limitations.

## Limitations

This is a portfolio-sized service. Before production use it would need at least:

- **Authentication and authorisation.** No endpoint is authenticated. In particular, anyone who
  can reach the API can approve proposals or add runbooks (which the agent then reads).
- **Background execution of triage.** The agent runs inside the HTTP request (up to
  `AGENT_TIMEOUT_SECONDS`). A job queue with a `202 Accepted` and polling would be more robust.
- **Idempotency and concurrency limits on triage.** Two triage calls for one incident run two
  agents and produce two proposals. Only one can be approved, but tokens are spent twice. There
  is no per-tenant or global rate limit on agent runs.
- **No live model validation.** Nothing in this repository has been run against the real Claude
  API; the adapter is tested only through the SDK with a mock transport. Prompt quality, tool
  descriptions and the default token budget have not been tuned on real runs.
- **Triage quality is unmeasured.** The evals check guardrails, not whether the proposals are
  right.
- **The injection heuristic is shallow.** It is a regex list meant to flag, not to protect.
- **Data retention.** Alerts, audit rows and proposals are kept forever; there is no
  partitioning or pruning.
- **Observability.** Logs only: no metrics, traces or token-cost dashboards.
- **Ingest at scale.** The per-service advisory lock serialises ingest per service, which is
  fine at moderate volume but would need rethinking for a very high-volume single service.
- **The Docker image is only built in CI.** It could not be built in the development sandbox (no
  registry access); the CI `docker` job builds it but nothing runs the container end to end.

## License

MIT. See [LICENSE](LICENSE).
