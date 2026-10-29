.DEFAULT_GOAL := help
UV ?= uv
RUN := $(UV) run --locked

.PHONY: help install lint format typecheck test cov eval check migrate downgrade seed run up down docker-build

help: ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

install: ## Create .venv with locked runtime + dev dependencies
	$(UV) sync --locked

lint: ## Ruff lint + format check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format: ## Apply ruff fixes and formatting
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

typecheck: ## mypy --strict on package, tests, evals and scripts
	$(RUN) mypy

test: ## Run tests against TEST_DATABASE_URL (real Postgres)
	$(RUN) pytest

cov: ## Tests with branch coverage report
	$(RUN) pytest --cov --cov-report=term-missing

eval: ## Offline eval scenarios (scripted model, real Postgres; truncates the eval DB)
	$(RUN) python -m evals.run_evals

check: lint typecheck cov eval ## Everything CI runs, minus the Docker build

migrate: ## alembic upgrade head on DATABASE_URL
	$(RUN) alembic upgrade head

downgrade: ## alembic downgrade -1 on DATABASE_URL
	$(RUN) alembic downgrade -1

seed: ## Load sample runbooks and alerts into DATABASE_URL
	$(RUN) python -m scripts.seed

run: ## Run the API locally with reload
	$(RUN) uvicorn triage.main:create_app --factory --reload --port 8000

up: ## docker compose up (api + postgres)
	docker compose up --build -d

down: ## docker compose down
	docker compose down

docker-build: ## Build the production image
	docker build -t incident-triage-agent:local .
