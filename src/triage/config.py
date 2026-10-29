"""Application settings, loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODEL = "claude-sonnet-5-5"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- infrastructure ---
    database_url: str = "postgresql+asyncpg://triage:triage@localhost:5432/triage"
    db_pool_size: int = Field(default=5, ge=1)
    db_max_overflow: int = Field(default=5, ge=0)
    log_level: str = "INFO"
    log_json: bool = True

    # --- alert ingestion ---
    dedup_window_seconds: int = Field(default=600, ge=1)
    group_window_seconds: int = Field(default=1800, ge=1)

    # --- LLM / agent ---
    anthropic_api_key: SecretStr | None = None
    triage_model: str = DEFAULT_MODEL
    agent_max_iterations: int = Field(default=8, ge=1, le=50)
    agent_token_budget: int = Field(default=60_000, ge=1_000)
    agent_timeout_seconds: float = Field(default=90.0, gt=0)
    agent_max_tokens_per_call: int = Field(default=2_048, ge=256)
    llm_request_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0, le=10)
    llm_backoff_base_seconds: float = Field(default=0.5, ge=0)
    llm_backoff_max_seconds: float = Field(default=8.0, ge=0)
    tool_output_max_chars: int = Field(default=12_000, ge=1_000)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
