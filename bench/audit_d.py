"""
v0.2 Fix 0 — audit of arm D failures on T2/T5 (and any other type on demand).

For every D run scoring < 1.0 we replay the engine LOCALLY (free, deterministic)
on the same case and put three things side by side: the engine's own verified
decomposition, the model's JSON answer, and ground truth. Each failing run is
classified into exactly one bucket:

  engine_error    the engine's terminal decomposition itself contradicts the
                  expected main driver (factor/mechanism) — MUST be shown in
                  full and stops the v0.2 work until reviewed;
  narrator_error  the engine was right, the model misrepresented it (listed
                  the tree path as drivers, flipped factor/mechanism, made up
                  shares not consistent with the engine output);
  label_error     the engine was right AND the model reported it faithfully —
                  the loss comes from the label/scoring being stricter than a
                  greedy walk can satisfy (e.g. T5 wants both drivers, the
                  engine honestly reports the dominant one).

Priority: engine_error > narrator_error > label_error.
"""
from __future__ import annotations

import json
import pathlib

from inferah_engine import investigate
from inferah_engine.datasource import PostgresSource
from inferah_engine.loader import load_pack

from bench.engine_arm import PACK_PATH, PERIODS
from bench.scoring import _driver_tuple, _factor_matches, _norm, score_run

ROOT = pathlib.Path(__file__).resolve().parents[1]

# the engine's terminal winner id -> (factor, mechanism) in answer-schema terms
LEAF_MAP = {
    "orders": ("orders", "volume"),
    "buyers": ("buyers", "volume"),
    "freq": ("freq", "volume"),
    "aov": ("aov", "rate"),
    "aov_rate": ("aov", "rate"),
    "aov_mix": ("aov", "mix"),
    "gmv": ("gmv", "volume"),
}


def replay_engine(pg_url: str, schema: str):
    """Re-run the deterministic walk; return (Result, compact summary dict)."""
    pack = load_pack(PACK_PATH)
    src = PostgresSource(pg_url, f"{schema}.orders", "ts_day", PERIODS,
                         pack.measures)
    try:
        res = investigate(src, pack.tree)
        drills = [(s.scope.split("·")[0].strip().split("∩")[-1].strip())
                  for s in res.steps if s.kind == "segment"]
        # the last drilled WHERE filter actually applied (from final scope)
        last_scope = res.steps[-1].scope if res.steps else "all"
        winner = res.steps[-1].winner if res.steps else None
        factor, mech = LEAF_MAP.get(winner, (winner, None))
        return res, {
            "abstained": res.abstained,
            "significant": res.significant,
            "terminal_winner": winner,
            "engine_factor": factor,
            "engine_mechanism": mech,
            "winner_share": res.steps[-1].winner_share if res.steps else None,
            "path": [f"{s.kind}:{s.winner}({s.winner_share:.2f})"
                     for s in res.steps],
            "scope": last_scope,
            "leaf": res.leaf,
        }
    finally:
        src.engine.dispose()


def _engine_matches_expected(summary: dict, label: dict) -> bool:
    """Does the engine's terminal decomposition agree with the label's MAIN
    driver on factor+mechanism? (Dimension naming is the narrator's job.)"""
    exp = label["expected"]
    if exp["action"] == "abstain":
        return bool(summary["abstained"])
    if exp["action"] == "no_driver":
        return not summary["significant"]
    main = max(exp["drivers"], key=lambda d: d["share_of_move"])
    return (_factor_matches(_norm(main["factor"]), _norm(summary["engine_factor"]))
            or _factor_matches(_norm(summary["engine_factor"]), _norm(main["factor"]))) \
        and _norm(main["mechanism"]) == _norm(summary["engine_mechanism"])


def _model_faithful(answer: dict, summary: dict) -> bool:
    """Did the model's MAIN driver report the engine's terminal verdict?"""
    drivers = [d for d in (answer.get("drivers") or []) if isinstance(d, dict)]
    if not drivers:
        return False
    main = max(drivers, key=lambda d: d.get("share_of_move") or 0)
    t = _driver_tuple(main)
    if not (_factor_matches(_norm(summary["engine_factor"]), t[2])
            and _norm(summary["engine_mechanism"]) == t[3]):
        return False
    # listing the walked tree path as EXTRA drivers is a misrepresentation
    return len(drivers) <= 2


def classify_run(rec: dict, label: dict, summary: dict, scored: dict) -> str:
    if not _engine_matches_expected(summary, label):
        return "engine_error"
    if rec["parse_error"] or not _model_faithful(rec["answer"] or {}, summary):
        return "narrator_error"
    return "label_error"


def audit(records, labels, banks, pg_url, types=("T2", "T5"),
          arm="D") -> list[dict]:
    """Audit every `arm` run of the given types scoring < 1.0."""
    by_id = {l["case_id"]: l for l in labels}
    cases = sorted({r["case_id"] for r in records
                    if r["arm"] == arm and by_id[r["case_id"]]["type"] in types})
    summaries = {cid: replay_engine(pg_url, cid)[1] for cid in cases}
    rows = []
    for rec in records:
        if rec["arm"] != arm or rec["case_id"] not in summaries:
            continue
        label = by_id[rec["case_id"]]
        scored = score_run(label, rec["answer"], banks[rec["case_id"]],
                           parse_error=rec["parse_error"])
        if scored["score"] >= 1.0:
            continue
        summary = summaries[rec["case_id"]]
        rows.append({
            "case_id": rec["case_id"], "type": label["type"],
            "run_idx": rec["run_idx"], "score": scored["score"],
            "bucket": classify_run(rec, label, summary, scored),
            "engine": summary,
            "answer": rec["answer"],
            "expected": label["expected"],
            "loss": {k: scored[k] for k in
                     ("action_correct", "driver_score", "grounding_pass",
                      "sum_share_pass")},
        })
    return rows


def bucket_table(rows) -> "pd.DataFrame":
    import pandas as pd
    df = pd.DataFrame([{k: r[k] for k in ("type", "bucket")} for r in rows])
    if df.empty:
        return pd.DataFrame()
    return df.value_counts().unstack(fill_value=0)
