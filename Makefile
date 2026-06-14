# inferah-bench — run-it-yourself entrypoints.
# Requires: docker (for postgres), a venv with `pip install -r requirements.txt`,
# and ANTHROPIC_API_KEY / OPENAI_API_KEY in your environment or .env.
PY ?= .venv/bin/python

.PHONY: help db seed dry-run full-run score report test clean-db

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

db: ## start the throwaway Postgres (docker)
	docker compose up -d postgres
	@echo "waiting for postgres..." && sleep 3

seed: ## generate + load all 28 cases into Postgres ($PG_URL)
	$(PY) -m cases.seed

dry-run: ## 3 cases x configured arms x 1 run, prints answers + cost estimate
	$(PY) -m bench.cli dry-run

full-run: ## the full grid (uses the raw.jsonl cache; only missing cells call the API)
	$(PY) -m bench.cli full-run

score: ## (re)score results/raw.jsonl into results/scores.parquet
	$(PY) -m bench.cli score

report: ## print the main comparison tables from results/scores.parquet
	$(PY) -m bench.cli report

test: ## unit tests (scoring + completeness gate; needs Postgres for the gate)
	$(PY) -m pytest tests/ -q

clean-db: ## stop + wipe the throwaway Postgres
	docker compose down
