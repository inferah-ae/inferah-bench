# inferah-bench

[![ci](https://github.com/inferah-ae/inferah-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/inferah-ae/inferah-bench/actions/workflows/ci.yml)

A controlled benchmark for **causal correctness** of LLM answers to *"why did
this metric change?"* — the everyday analytics question where a confident,
plausible, wrong answer is worse than no answer.

**What it measures.** Given a metric that moved between two periods over a
real Postgres table, can an agent name the *actual* driver — or honestly
abstain when the cause is outside the data? Every case has **ground truth by
construction** (the driver is injected by the generator), so scoring is
**deterministic** — no LLM-judge, no rubric drift. The scorer also recomputes
every number the agent cites straight from Postgres and fails fabrications.

**What it compares.** A bare SQL agent, the same agent handed exhaustive
documentation, and an agent whose only tool is a deterministic decomposition
engine ([inferah-engine](https://github.com/inferah-ae/inferah-engine)). The
question it answers: *do docs cure confident wrongness — and what's left after
them?*

## Headline result

Mean score (0–1), 28 cases × 5 runs, temperature 0:

| arm | what it is | model | **ALL** |
|---|---|---|---|
| **A** | bare agent, `run_sql` only | claude-sonnet-4-6 | 0.70 |
| **B** | agent + exhaustive docs, `run_sql` | claude-sonnet-4-6 | 0.70 |
| **D** | agent + engine, LLM narrates text (v0) | claude-sonnet-4-6 | 0.78 |
| **D2** | agent + engine, LLM narrates JSON + completeness gate (v0.2) | claude-sonnet-4-6 | 0.90 |
| **D3** | engine + **code** narrator, zero LLM in the answer loop (v0.3) | — (deterministic) | **0.94** |

By failure type:

| type | injected cause | A | B | D | D2 | D3 |
|---|---|---|---|---|---|---|
| T1 | 100% in one segment (control) | 0.97 | 0.89 | 0.90 | 0.96 | 1.00 |
| T2 | one factor of GMV=buyers×freq×AOV | 0.89 | 0.41 | 0.65 | 0.84 | 1.00 |
| T3 | Simpson (rate vs mix) | 0.85 | 0.76 | 0.88 | 0.90 | 0.90 |
| T4 | driver outside the columns → abstain | 0.76 | 0.66 | 1.00 | 1.00 | 1.00 |
| T5 | two independent drivers | 0.74 | 0.73 | 0.63 | 0.67 | 0.70 |
| T6 | move within noise → no_driver | 0.24 | 0.60 | 1.00 | 1.00 | 1.00 |
| T7 | incomplete data → abstain | 0.44 | 0.88 | 0.40 | 0.95 | 0.95 |
| **ALL** | | **0.70** | **0.70** | **0.78** | **0.90** | **0.94** |

Components:

| arm/model | action | driver | grounding | sum-share | stability | $/run |
|---|---|---|---|---|---|---|
| A / sonnet-4-6 | 0.73 | 0.60 | 0.71 | 0.97 | 0.72 | 0.045 |
| B / sonnet-4-6 | 0.75 | 0.56 | 0.79 | 0.98 | 0.69 | 0.048 |
| D2 / sonnet-4-6 | 1.00 | 0.82 | 0.89 | 0.96 | 0.94 | 0.010 |
| **D3 / code** | **1.00** | **0.89** | 0.89 | **1.00** | **1.00** | **0.00** |

*(Single-model grid. The multi-provider transport is implemented and verified
on single calls for OpenAI and Google — `bench/llm.py`; arms A/B run on
`gpt-5.5` and `gemini-3.1-pro-preview` end to end on one case each. A full
non-Anthropic grid was not run on budget: in this agentic harness gpt-5.5
costs ≈ $28 for 280 cells — each case re-sends the growing transcript + the
large arm-B doc across up to 15 tool calls, with no OpenAI-side prompt cache,
and gpt-5.5 also emits ~2.3k reasoning tokens/cell — and the Gemini key was
free-tier (quota-exhausted under load). Add models via `BENCH_MODELS` on a
funded key and re-run; the columns appear automatically.)*

Three findings:
1. **Exhaustive documentation did not raise the average** (A 0.70 → B 0.70).
   It cured the tail (noise T6 0.24→0.60, data gaps T7 0.44→0.88) and broke
   the middle by the same caution (T2 0.89→0.41: false "no driver" on real,
   uniformly-spread drivers). Docs *reshuffle* errors; they don't remove them.
2. **The bare agent confidently explains noise** in ~70% of T6 runs and
   fabricates a cited number in up to 75% of T5 runs.
3. **The deterministic layer wins on honesty, stability, and cost** —
   abstains correctly when the cause is off-data, never invents an
   explanation, reproduces exactly, and (as code, D3) costs nothing. The LLM
   narrator (D2) only *degraded* the engine's output; replacing it with code
   (D3) was a strict improvement everywhere (audit: 26 narrator errors, 0
   engine errors).

## Quick start (run it yourself)

```bash
git clone <this repo> && cd inferah-bench
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # pulls inferah-engine from GitHub
cp .env.example .env                      # add ANTHROPIC_API_KEY (and OPENAI_API_KEY)

make db          # throwaway Postgres in docker (port 5544)
make seed        # generate + load all 28 cases
make dry-run     # 3 cases x arms x 1 run: raw answers + cost estimate
make full-run    # the full grid (cached in results/raw.jsonl)
make report      # the comparison tables above
make test        # unit tests (scoring + completeness gate)
```

Re-running never re-burns tokens: every (case, arm, run, model) is cached in
`results/raw.jsonl`; only missing cells call the API. The engine arms D2/D3
and the data are free/deterministic, so `make seed && make report` reproduces
the engine side with no key at all.

### Plug in your own agent

Implement one method and the scorer treats you like a built-in arm:

```python
class Agent:
    def answer(self, question: str, conn) -> dict:
        # conn: read-only SQLAlchemy connection scoped to one case schema
        # return the answer schema (see cases/labels.schema.json)
        ...
```

Full working example: [`examples/custom_agent.py`](examples/custom_agent.py)
(`python -m examples.custom_agent` scores it on all 28 cases).

## Cases — 28 = 7 types × 4 parameterizations

One data model (order grain, generic e-commerce, fictional names):
`order_id, ts_day, period (p0/p1), country, city, order_type, category, gmv,
user_id`. Each case is its own schema `case_01 … case_28`. Generators
(`cases/generators.py`) are byte-for-byte deterministic; ground truth in
`cases/labels.json` (validated against `cases/labels.schema.json`).

| type | what's injected | expected |
|---|---|---|
| T1 | 100% of the move in one segment (drop + spike) | explain, volume |
| T2 | one factor of GMV = buyers × freq × AOV | explain |
| T3 | Simpson: per-segment rates flat, mix moved (+ reverse-Simpson) | explain mix / no_driver |
| T4 | true driver outside the columns (NULL-segment / uniform external) | **abstain** (unmapped_dimension) |
| T5 | two independent drivers (70/30, 50/50) | explain, both drivers |
| T6 | move within daily noise (+ noise hiding a tiny real effect) | **no_driver** |
| T7 | incomplete p1 (missing day / source / mid-period truncation) | **abstain** (data_gap) |

## Scoring (`bench/scoring.py`)

Per run: **action 30% + driver 40% + grounding 20% + sum-of-shares 10%**.
`mechanism_correct` (rate vs mix vs volume — the Simpson test) and
**stability** (modal-answer share across runs) are reported separately, not
scored. Matching rules are frozen and documented in the module docstring
(string normalization; factor-child credit; T5 partial credit; abstain-reason
half credit; parse failure → 0).

## Methodology — what changed across versions

| version | change | arms touched |
|---|---|---|
| **v0** | initial grid: A, B, D (engine tool → rendered text → LLM narrates) | A, B, D |
| **v0.2** | audit of D's failures (26 narrator-error, 0 engine-error) → **D2**: structured JSON narrator interface + one completeness gate | **D2 only**; A/B/D frozen |
| **v0.3** | **D3**: replace the LLM narrator with a pure code mapping (zero LLM in the answer loop); multi-model transport for A/B; run-it-yourself packaging | **D3 only**; A/B/D/D2 frozen |

Arms A and B — their prompts, the cases, the labels, and the scoring — have
not changed since v0. Arm D was iterated (v0 → D2 → D3); **all versions stay
in the table** for honest methodology.

## Limitations (stated plainly)

- **(a) Synthetic data.** These are constructed cases with injected ground
  truth, not real warehouse data. They isolate specific failure modes; they
  do not prove behavior on messy production tables.
- **(b) The benchmark author is the author of the engine baseline (arm D).**
  That is a conflict of interest — which is exactly why this is public with a
  one-method plug-in interface: **run it, add your agent, and try to beat or
  break it.** PRs that improve any arm or add a case are welcome.
- **(c) Arm D was iterated twice (v0 → D2 → D3); arms A/B were frozen at v0**
  and are very likely improvable by better prompting. The asymmetry is
  disclosed, all D versions are shown, and A/B prompts are in the repo for
  anyone to improve.
- **(c2) Single-model results.** Arms A/B were fully run only on
  `claude-sonnet-4-6`, so "LLM agents are confidently wrong" is, strictly,
  shown for one model. The multi-provider transport is built and verified on
  single calls (OpenAI + Google); the full cross-model grid is left to anyone
  with a funded key (see the note under the components table). Until then the
  generalization across providers is a hypothesis, not a result.
- **(d) Compound cases (T5).** The engine's greedy walk reports the single
  dominant driver, not both — a known limitation (engine roadmap:
  multi-driver decomposition), not a scoring artifact.

## Not included

arm C (verification gates inside the agent loop) · new case types · web
leaderboard · prompt iteration after seeing results (that would p-hack the
benchmark).

Apache-2.0. The decomposition engine is the sibling repo,
[inferah-engine](https://github.com/inferah-ae/inferah-engine).
