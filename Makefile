.PHONY: help setup dev lint test test-one up down restart logs ps \
	scale-workers eval-runtime eval-load eval-retrieval eval-generation \
	eval-offline eval-dataset ingest shadow-reindex backfill-index-version \
	export-data import-data

# Dev API auth headers (DEV_AUTH=true locally). Override: make ingest TENANT=t2 USER=u2
TENANT ?= dev-tenant
USER ?= dev-user
API ?= http://localhost:8000

help:
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Copy .env and install deps
	cp -n .env.example .env || true
	uv sync --extra dev

dev: ## Run API with reload (host, no docker)
	uv run uvicorn app.main:app --app-dir backend --reload

lint: ## Ruff check
	uv run ruff check backend evaluation tests

test: ## Run full test suite
	uv run --extra dev pytest -q

test-one: ## Run one test: make test-one T=tests/test_api.py::test_name
	uv run --extra dev pytest -q $(T)

up: ## Start full stack (api, frontend, postgres, redis, minio, neo4j, worker)
	docker compose up --build -d

down: ## Stop and remove stack
	docker compose down

restart: down up ## Restart the stack

logs: ## Tail stack logs
	docker compose logs -f

ps: ## Show stack container status
	docker compose ps

scale-workers: ## Scale worker replicas: make scale-workers N=3
	docker compose up -d --scale worker=$(N) worker

eval-runtime: ## Runtime report
	uv run python evaluation/runtime_report.py

eval-load: ## Load test: make eval-load COUNT=100000 CONCURRENCY=20
	uv run python evaluation/load_test.py --count $(or $(COUNT),100000) --concurrency $(or $(CONCURRENCY),20)

eval-retrieval: ## Retrieval eval
	uv run python evaluation/retrieval_eval.py

eval-generation: ## Generation eval
	uv run python evaluation/generation_eval.py

eval-offline: ## Offline eval (retrieval + generation)
	uv run python evaluation/offline_eval.py

eval-dataset: ## Build multihop eval dataset
	uv run python evaluation/build_multihop_dataset.py

ingest: ## Upload a document: make ingest FILE=path/to/doc.pdf
	curl -sS -X POST $(API)/api/v1/documents \
		-H "X-Tenant-ID: $(TENANT)" -H "X-User-ID: $(USER)" \
		-F "file=@$(FILE)"

shadow-reindex: ## Enqueue a shadow reindex: make shadow-reindex VERSION=v2
	uv run python scripts/enqueue_shadow_reindex.py --version $(VERSION)

backfill-index-version: ## One-time migration for pre-versioning records
	uv run python scripts/backfill_index_version.py

export-data: ## Export postgres/neo4j/weaviate to data/backup for moving to another system
	uv run python scripts/export_data.py

import-data: ## Import a data/backup produced by export-data into this stack
	uv run python scripts/import_data.py
