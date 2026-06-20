"""
Honest statistics for the headline grid: arm means with cluster-bootstrap 95%
CIs (resampling the 28 cases, not the 140 runs — runs within a case are not
independent), plus trivial-baseline scores so the floor is explicit.

CIs are recomputed from the committed `results/scores.parquet` (no DB, no API
key). Trivial baselines need the scorer + a fact bank (Postgres), so their
values are read from the committed `results/headline_stats.json`; pass
`--recompute-baselines` with $PG_URL set to regenerate them.

    python -m bench.stats                      # print arm CIs + baselines
    python -m bench.stats --recompute-baselines
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
STATS = ROOT / "results" / "headline_stats.json"


def _ci(per_case_means, B=5000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(per_case_means, dtype=float)
    n = len(v)
    bs = [rng.choice(v, n, replace=True).mean() for _ in range(B)]
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def arm_cis(scores: pd.DataFrame) -> dict:
    scores = scores.copy()
    scores["col"] = scores.apply(
        lambda r: r.arm if str(r.arm).startswith("D") else f"{r.arm}/{r.model}",
        axis=1)
    out = {}
    for c in sorted(scores.col.unique()):
        sub = scores[scores.col == c]
        pcm = sub.groupby("case_id").score.mean().values
        m, lo, hi = _ci(pcm)
        out[c] = {"mean": round(m, 3), "ci95": [round(lo, 3), round(hi, 3)],
                  "n_cases": int(sub.case_id.nunique())}
    return out


def trivial_baselines(pg_url: str) -> dict:
    """Score three zero-intelligence agents through the real scorer."""
    from bench.runner import fact_banks
    from bench.scoring import score_run
    labels = json.loads((ROOT / "cases" / "labels.json").read_text())
    banks = fact_banks(labels, pg_url)
    agents = {
        "always_abstain_data_gap": lambda l: {
            "action": "abstain", "drivers": [], "abstain_reason": "data_gap",
            "cited_numbers": []},
        "always_explain_broad_aov_rate": lambda l: {
            "action": "explain", "abstain_reason": None, "cited_numbers": [],
            "drivers": [{"dimension": None, "segment": None, "factor": "aov",
                         "mechanism": "rate", "share_of_move": 1.0}]},
        "always_no_driver": lambda l: {
            "action": "no_driver", "drivers": [], "abstain_reason": None,
            "cited_numbers": []},
    }
    out = {}
    for name, make in agents.items():
        per = [score_run(l, make(l), banks[l["case_id"]])["score"] for l in labels]
        m, lo, hi = _ci(per)
        out[name] = {"mean": round(m, 3), "ci95": [round(lo, 3), round(hi, 3)]}
    return out


def main():
    scores = pd.read_parquet(ROOT / "results" / "scores.parquet")
    stats = {"arms": arm_cis(scores)}
    if "--recompute-baselines" in sys.argv:
        import os
        from bench.config import pg_url
        stats["trivial_baselines"] = trivial_baselines(pg_url())
        STATS.write_text(json.dumps(stats, indent=2) + "\n")
        print("wrote", STATS)
    else:
        if STATS.exists():
            stats["trivial_baselines"] = json.loads(STATS.read_text()).get(
                "trivial_baselines", {})
    print("arm means with 95% CI (cluster-bootstrap over cases):")
    for c, s in stats["arms"].items():
        print(f"  {c:<12} {s['mean']:.3f}  {s['ci95']}")
    for name, s in stats.get("trivial_baselines", {}).items():
        print(f"  [baseline] {name:<30} {s['mean']:.3f}  {s['ci95']}")


if __name__ == "__main__":
    main()
