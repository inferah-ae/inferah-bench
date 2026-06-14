"""
Command line for the run-it-yourself flow (driven by the Makefile):

    python -m bench.cli dry-run    # 3 cases x arms x 1 run, raw answers + cost
    python -m bench.cli full-run   # the full grid (cached)
    python -m bench.cli score      # raw.jsonl -> scores.parquet
    python -m bench.cli report     # print comparison tables

Which arms/models run is config, not code: BENCH_ARMS and BENCH_MODELS env
vars (comma-separated), defaulting to the frozen v0.x grid. Arms A/B run per
model in BENCH_MODELS; engine arms D/D2/D3 run once (model tag is intrinsic).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys

from bench.config import pg_url
from bench.runner import (ROOT, fact_banks, run_benchmark, score_records,
                          usage_summary)

LABELS = json.loads((ROOT / "cases" / "labels.json").read_text())
DRY_CASES = ["case_01", "case_09", "case_21"]   # T1 control, T3 Simpson, T6 noise

# SQL arms run once per model; engine arms carry their own model tag.
SQL_ARMS = tuple(os.environ.get("BENCH_ARMS", "A,B").split(","))
SQL_MODELS = tuple(os.environ.get("BENCH_MODELS", "claude-sonnet-4-6").split(","))
ENGINE_ARMS = (("D2", "claude-sonnet-4-6"), ("D3", "code-narrator"))


def _run(arms_models, n_runs, cases, progress=True):
    url = pg_url()
    recs = []
    for arm in SQL_ARMS:
        for model in SQL_MODELS:
            recs += run_benchmark(LABELS, url, arms=(arm,), n_runs=n_runs,
                                  cases=cases, model=model, progress=progress)
    for arm, model in ENGINE_ARMS:
        nr = 1 if arm == "D3" else n_runs
        recs += run_benchmark(LABELS, url, arms=(arm,), n_runs=nr,
                              cases=cases, model=model, progress=progress)
    return recs


def dry_run():
    recs = _run(None, n_runs=1, cases=DRY_CASES, progress=False)
    by_id = {l["case_id"]: l for l in LABELS}
    fresh = [r for r in recs if r["case_id"] in DRY_CASES]
    for r in fresh:
        exp = by_id[r["case_id"]]["expected"]
        ans = r["answer"]
        print(f"\n{r['case_id']} [{by_id[r['case_id']]['type']}] "
              f"arm {r['arm']} ({r.get('model')})")
        print("  expected:", json.dumps(exp, ensure_ascii=False))
        print("  answer:  ", json.dumps(ans, ensure_ascii=False)
              if ans else f"PARSE ERROR: {r['raw_text'][:160]}")
    u = usage_summary(fresh)
    full = len(SQL_ARMS) * len(SQL_MODELS) * 28 * 5 + 28 * 5 + 28
    print(f"\ndry: {u['n_calls']} calls, ${u['cost_usd']:.2f}")
    if u["n_calls"]:
        print(f"full grid ~{full} calls, est ~${u['cost_usd']/u['n_calls']*full:.0f} "
              f"(upper bound; caching + free D3 reduce it)")


def full_run():
    recs = _run(None, n_runs=5, cases=None)
    u = usage_summary(recs)
    print(f"{u['n_calls']} records | ${u['cost_usd']:.2f}")


def score():
    from bench.runner import load_cache
    recs = list(load_cache().values())
    df = score_records(recs, LABELS, fact_banks(LABELS, pg_url()))
    df.to_parquet(ROOT / "results" / "scores.parquet")
    print(f"scored {len(df)} records -> results/scores.parquet")


def report():
    import pandas as pd
    p = ROOT / "results" / "scores.parquet"
    if not p.exists():
        score()
    df = pd.read_parquet(p)
    df["col"] = df.apply(
        lambda r: r.arm if r.arm.startswith("D") else f"{r.arm}/{r.model}",
        axis=1)
    pd.set_option("display.precision", 2)
    main = df.pivot_table(index="type", columns="col", values="score",
                          aggfunc="mean")
    main.loc["ALL"] = df.groupby("col")["score"].mean()
    print("\n=== main: mean score (type x arm/model) ===")
    print(main.round(2).to_string())
    comp = df.groupby("col")[["action_correct", "driver_score",
                              "grounding_pass", "sum_share_pass"]].mean()
    comp["stability"] = df.groupby("col")["modal_share"].mean()
    print("\n=== components ===")
    print(comp.round(2).to_string())


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    {"dry-run": dry_run, "full-run": full_run, "score": score,
     "report": report}[cmd]()


if __name__ == "__main__":
    main()
