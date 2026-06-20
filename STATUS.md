# Reproducibility & status

What's pinned, what's verified, and what ships — so a result here points at an
immutable, re-runnable state.

## Pinned & reproducible
- `requirements.txt` pins the engine to an immutable tag:
  `inferah-engine @ ...@v0.3.3`. A fresh `pip install -r requirements.txt`
  always resolves the same engine commit.
- Case generators are byte-for-byte deterministic (two independent builds hash
  identical); `cases/labels.json` is regenerated from them and validates
  against `cases/labels.schema.json` (`python -m cases.build_labels`).
- `make db && make seed && make report` reproduces the engine-side table from a
  clean machine **with no API key** (Postgres on :5544). CI runs the full path
  — Postgres + seed + tests — on every push.
- Custom agents plug in via one method (`examples/custom_agent.py`).

## What ships vs. what's regenerated
- **Ships:** `bench/`, `cases/` (incl. `labels.json`, `docs/`), `tests/`,
  `examples/`, `benchmark.ipynb`, `results/scores.parquet` (reference
  aggregate), `results/audit_d_v0.json`, `results/cost_per_answer.json`,
  `results/exploratory_opus_hardset.json`, README, LICENSE, Makefile,
  `docker-compose.yml`, `requirements.txt`, `.env.example`.
- **Regenerated locally (gitignored):** `results/raw.jsonl` and per-shard runs
  — these are one operator's model answers; you generate your own with
  `make full-run`. `.env`, `.venv/`, caches.

## Scope of the published results
- Headline grid (28 cases × 5 runs) is graded on `claude-sonnet-4-6`. A
  frontier-model probe (`claude-opus-4-8`, hard set) is in the README's
  exploratory section, clearly separated from the headline.
- Multi-provider transport (OpenAI + Google) is implemented and smoke-tested
  (each answered one case end to end — a single call, nothing more); there is
  no scored non-Anthropic headline grid. Left to anyone with a funded key.
  Cross-provider generalization is stated as a hypothesis (README limitation c2).
- The score has a high trivial-baseline floor (~0.47) and a coarse grounding
  gate; compare gaps and non-overlapping CIs, not absolute numbers. See
  `results/headline_stats.json` and `python -m bench.stats`.

## Frozen vs. iterated
- Cases, labels (ground-truth values), scoring, and the arm A/B/D2 prompts are
  unchanged since v0. Arm D was iterated v0 → D2 → D3; all versions stay in the
  table. See the README "Methodology" and "Limitations" sections.

Apache-2.0 (same as the engine).
