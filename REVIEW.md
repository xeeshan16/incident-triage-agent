# How the AI output was reviewed

The code in this repository was written with AI assistance (Claude). This document records,
truthfully, which checks were applied to that output, what the review found, and what is still
outstanding. It is deliberately specific: each finding names the commit that fixed it and the
regression test that pins it.

Two kinds of review are described separately:

1. **Automated gates and a structured self-review** carried out during the build session by the
   AI assistant itself (sections 1 and 2). A self-review by the same system that wrote the code
   has obvious blind spots; treat it as a first pass, not as independent assurance.
2. **Maintainer review** by the repository owner (section 3), which is still to be done.

## 1. Review process

### Automated gates

Every change had to pass all of these before it was committed (CI runs the same set):

| Gate | Command | What it catches |
| --- | --- | --- |
| Lint | `ruff check .` (rules incl. bugbear, bandit `S`, async, simplify) | Common bugs, unsafe patterns, unused code |
| Format | `ruff format --check .` | Style drift |
| Types | `mypy` in strict mode over `src/`, `tests/`, `evals/`, `scripts/` | Type errors, unhandled `None`, wrong union handling |
| Migrations | `alembic upgrade head` / `alembic check` / `downgrade base` / `upgrade head` | Schema drift from models, irreversible migrations |
| Tests | `pytest` against **real PostgreSQL 16** (no SQLite) | Behaviour, including FTS, pg_trgm, advisory locks, row locks |
| Evals | `python -m evals.run_evals` | Guardrails and citation validity across scripted scenarios |
| Docker build | `docker build .` (CI only) | Image builds from the lockfile |

Notes on the gates:

- The test database is built by running the Alembic migrations, so migrations are exercised on
  every test run.
- Two guardrails were mutation-checked by hand: disabling tag escaping and disabling the
  citation-provenance check each made the expected tests fail (1 and 5 tests respectively), then
  the code was restored.
- The Docker build could not be run in the build sandbox (registry access was blocked). It is
  covered only by the CI job, which has not run yet because nothing has been pushed.

### Manual review checklist

After the first complete version, the whole codebase was reviewed against this checklist. Where
a check could be made executable, a probe script or test was written rather than relying on
reading alone.

- **SQL injection**: every query built with SQLAlchemy expressions and bound parameters; no
  string-formatted SQL reaching the database with external input; FTS input cannot alter query
  semantics.
- **Transaction boundaries**: each request is exactly one transaction; commit happens before the
  response; no transaction is held open across a model call; failure paths roll back.
- **Async session handling**: no session shared across tasks; sessions closed on all paths;
  connection pool not held during slow I/O.
- **Secret handling**: API key never logged, never in `repr`, never persisted in the audit trail.
- **Prompt injection**: all attacker-influenced text reaches the model only inside escaped
  delimiters inside `tool_result` blocks; the agent has no write capability; output and
  citations validated.
- **Error paths**: bad input gives 4xx, not 5xx; external failures give a recorded, explained
  failure; no state left half-written (e.g. runs stuck in `running`).
- **N+1 queries**: query count does not grow with row count on list/detail endpoints and tools.
- **Migration reversibility**: downgrade removes everything upgrade created; autogenerate shows
  no drift.

## 2. Findings log

Every entry below is a real defect found in the self-review pass and fixed in a separate commit.
Each fix commit adds a regression test (most are in `tests/test_review_findings.py`) that fails
on the pre-fix code; that failure was observed before each fix was applied.

| # | Category | What was wrong | Fix | Commit |
| --- | --- | --- | --- | --- |
| F1 | Transaction boundaries | The per-request session dependency used `yield`, and with the installed FastAPI its teardown (the commit) ran **after** the response had started. A failed commit could follow a `201` already sent. Confirmed by recording event order: `['response.start', 'commit']`. | Dependency declared with `scope="function"` so commit happens before the response. | `78e42d4` (test tidy-up in `7193ead`) |
| F2 | Error paths | NUL characters (`\u0000`) in titles, payloads, runbooks or reviewer names, and path ids beyond Postgres `BIGINT`, reached the database and produced **500** errors. | Schema-level validators reject NUL anywhere (including nested JSON keys/values); path ids bounded to `1..2^63-1`. Both now return 422. | `7aaddcf` |
| F3 | Error paths / data integrity | A caller-supplied `received_at` far in the future (e.g. year 2999) was accepted, which would set `incident.last_seen_at` into the future and keep that incident's dedup/grouping window open indefinitely. | Reject timestamps more than 5 minutes ahead of server time. | `03f72b5` |
| F4 | Error paths / async | If the request was cancelled (client disconnect, shutdown) mid-run, or inserting the proposal failed, the `agent_runs` row was left with `status = 'running'` forever. | All exit paths finalise the run (`cancelled` / `internal_error`); the status write is shielded from cancellation; the original exception is re-raised. | `0481842` |
| F5 | Transaction boundaries / logic | Two pending proposals for one incident could both be approved; the second silently re-actioned an incident that was already actioned. | Approval requires the incident to still be `open`, checked under the row lock (lock order proposal then incident). Rejecting a stale proposal is still allowed. | `87a3f4f` |
| F6 | Prompt injection | Only alert data was wrapped as untrusted. Runbook titles, tags and snippets (creatable through the unauthenticated `POST /runbooks`) and past approved triage summaries (model output that can echo injected text) reached the model unescaped. | Same escaping wrapper applied to all of them; system prompt updated to name them. | `3365662` |
| F7 | Error paths (LLM protocol) | An `end_turn` response with no content blocks was echoed back as an empty assistant message, which the Messages API rejects with a 400 on the next call. | Substitute a minimal text block so the transcript stays valid and roles alternate. | `bb85a3b` |
| F8 | Error paths / secret handling | With no `ANTHROPIC_API_KEY`, the SDK raised a raw `TypeError`/`CredentialsError` on first use, giving an opaque 500. The SDK could also silently pick up a credentials profile from the host, making behaviour machine-dependent. | Build the SDK client only when the key is set; otherwise raise `LLMError`, which records the run as failed and returns 502 with the run id. | `a85900f` |
| F9 | Prompt injection / correctness | Oversized tool output was cut with a raw string slice. The JSON became invalid, an `<untrusted_alert_payload>` wrapper could be left unterminated (so later text would appear to be outside it), and provenance still contained ids the model never received, so those ids could be cited. | Per-payload cap applied **inside** the wrapper; whole list items dropped until the output fits (marked `truncated` / `omitted_items`); provenance records only ids actually emitted. | `6ab24b5` |
| F10 | Error paths / operability | With the database host unreachable, `/readyz` blocked on TCP connect for about 60 s before answering. Measured: 60.1 s before, 2.0 s after. | 2-second timeout around the readiness query. | `9218e70` |

A process slip, recorded here because it is relevant to how much the gates can be trusted:
the F3 fix (`03f72b5`) was committed after running lint, types and tests but **not** the evals.
The eval scenarios generated timestamps slightly in the future and started failing. This was
caught at the next full gate run and fixed in `8f2851d`; from then on the evals were part of
every pre-commit gate run.

Earlier in the build, before the first commit, the gates also caught problems that never reached
a commit. These are noted for completeness but are not part of the findings log: mypy found an
attribute access on the wrong union member in the citation-handling path that would have
crashed when citations were rejected; a test found that a leading `-` in a runbook search query
became a negation operator; and a test found that late, back-dated alerts could be grouped into
a newer incident (the dedup and grouping windows were one-sided).

### Categories with no findings

- **SQL injection**: none found. All queries use SQLAlchemy expressions with bound parameters.
  The only string-built SQL is fixed DDL in the migration and `TRUNCATE` statements over a
  hard-coded table list in tests, the eval runner and the seed script. Runbook search input goes
  through `websearch_to_tsquery` (which does not raise on arbitrary syntax) after being reduced
  to word tokens; a test sends hostile queries (`'; DROP TABLE ...`, unbalanced operators).
- **N+1 queries**: none found. Incident detail uses `selectinload`; `find_similar_incidents`
  fetches approved summaries for all hits in one query. Measured with a statement counter:
  incident detail plus list issued 4 statements with 1 alert and 4 statements with 8 alerts.
- **Secret handling** (beyond F8): none found. The key is a `SecretStr`; `repr`,
  `model_dump()` and `model_dump_json()` were checked and do not reveal it; it is not logged and
  is not written to any audit table.
- **Migration reversibility**: none found. `upgrade` / `downgrade base` / `upgrade` round-trips
  cleanly and `alembic check` reports no drift. The downgrade intentionally leaves the
  `pg_trgm` extension installed (it is database-wide); this is documented in the migration.
- **Async session handling** (beyond F1 and F4): none found. Sessions are never shared between
  tasks; the agent opens short transactions per step and holds no connection across model calls.

### Known weaknesses not fixed

These were identified but deliberately left, and are listed in the README limitations:

- No authentication or authorisation on any endpoint, including approve/reject.
- `POST /incidents/{id}/triage` runs the agent synchronously inside the request.
- The prompt-injection heuristic is a small regex list; it flags, it does not protect.
- Nothing here has been run against the live Claude API. The adapter is tested against the
  real `anthropic` SDK with a mock HTTP transport only.

## Maintainer review (Zeeshan)

> TODO: to be completed by the maintainer. Nothing in this section has been filled in yet.

- [ ] Read the codebase, starting with `src/triage/agent/loop.py` and `src/triage/agent/tools.py`
- [ ] Re-run all gates locally: `make check`
- [ ] Verify each finding F1-F10 against its commit and regression test
- [ ] Run the eval scenarios against the live API: `python -m evals.run_evals --live`
- [ ] Check the CI run on GitHub after the first push (including the Docker build job)
- [ ] Record any additional findings, and what was changed, below

Notes:

<!-- Maintainer notes go here. -->
