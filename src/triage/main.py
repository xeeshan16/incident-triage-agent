"""Application factory, lifespan, request-id middleware and error handlers."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from triage.agent.llm import AnthropicClient, LLMClient, RetryingClient, RetryPolicy
from triage.agent.loop import AgentConfig, TriageAgent
from triage.api.routes import router
from triage.config import Settings, get_settings
from triage.db.base import make_engine, make_session_factory
from triage.logging_config import configure_logging, get_logger
from triage.services.errors import AgentRunFailedError, ConflictError, NotFoundError

log = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


def build_llm_client(settings: Settings) -> LLMClient:
    api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
    inner = AnthropicClient(api_key=api_key, timeout_seconds=settings.llm_request_timeout_seconds)
    return retrying(inner, settings)


def retrying(inner: LLMClient, settings: Settings) -> RetryingClient:
    def on_retry(attempt: int, delay: float, exc: Exception) -> None:
        log.warning("llm_retry", attempt=attempt, delay_s=round(delay, 3), error=str(exc))

    return RetryingClient(
        inner,
        RetryPolicy(
            max_retries=settings.llm_max_retries,
            base_seconds=settings.llm_backoff_base_seconds,
            max_seconds=settings.llm_backoff_max_seconds,
        ),
        on_retry=on_retry,
    )


def agent_config(settings: Settings) -> AgentConfig:
    return AgentConfig(
        model=settings.triage_model,
        max_iterations=settings.agent_max_iterations,
        token_budget=settings.agent_token_budget,
        timeout_seconds=settings.agent_timeout_seconds,
        max_tokens_per_call=settings.agent_max_tokens_per_call,
        tool_output_max_chars=settings.tool_output_max_chars,
    )


def create_app(settings: Settings | None = None, llm: LLMClient | None = None) -> FastAPI:
    """Build the app. ``llm`` is injectable so tests can supply a scripted fake client."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, json=settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = make_engine(
            settings.database_url,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
        )
        factory = make_session_factory(engine)
        client = llm or build_llm_client(settings)
        app.state.settings = settings
        app.state.session_factory = factory
        app.state.agent = TriageAgent(client, factory, agent_config(settings))
        log.info("startup", model=settings.triage_model)
        try:
            yield
        finally:
            await engine.dispose()

    app = FastAPI(title="incident-triage-agent", version="0.1.0", lifespan=lifespan)
    app.include_router(router)

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        # Accept a caller-supplied id only if it is short and printable; otherwise mint one.
        request_id = (
            incoming if 0 < len(incoming) <= 64 and incoming.isprintable() else uuid.uuid4().hex
        )
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("request_failed", method=request.method, path=request.url.path)
            response = JSONResponse({"detail": "internal server error"}, status_code=500)
        response.headers[REQUEST_ID_HEADER] = request_id
        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return response

    @app.exception_handler(NotFoundError)
    async def _not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ConflictError)
    async def _conflict(_: Request, exc: ConflictError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(AgentRunFailedError)
    async def _agent_failed(_: Request, exc: AgentRunFailedError) -> JSONResponse:
        return JSONResponse(
            {
                "detail": "triage agent did not produce an acceptable proposal",
                "agent_run_id": exc.run_id,
                "termination_reason": exc.reason,
            },
            status_code=502,
        )

    return app
