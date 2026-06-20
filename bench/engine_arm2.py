"""
Arm D2 (v0.2): same architecture as arm D — the LLM's only tool is the
deterministic engine, no SQL — with exactly two changes:

  Fix 1  STRUCTURED narrator interface. The tool returns JSON, not prose:
         the model maps structure to structure under explicit rules, instead
         of re-reading a rendered report (v0 finding: at temperature 0 the
         narrator re-told the same deterministic text differently run to run
         and listed the walked tree path as 'drivers').

  Fix 2  COMPLETENESS GATE (one, simple). Before the walk: if p1 lost a day
         of coverage, or a segment alive in p0 is empty/truncated in p1 while
         the rest of the data flows, the tool returns abstain(data_gap)
         without walking. Row/day counts only — no statistics. Lives here in
         the bench harness, NOT in the inferah_engine package.

Arms A/B and arm D (v0) are frozen; this file is the only D2 surface.
The prompt below was frozen before the D2 rerun.
"""
from __future__ import annotations

import json
import re

import pandas as pd
from sqlalchemy import create_engine

from inferah_engine import investigate
from inferah_engine.datasource import PostgresSource
from inferah_engine.loader import load_pack

from bench.agent import MODEL, ANSWER_PROTOCOL, ArmResult, run_tool_loop
from bench.engine_arm import PACK_PATH, PERIODS, _filters_from_scope

DAY_COVERAGE_MIN = 0.95     # p1 must keep >=95% of p0's day coverage
SEG_MIN_SHARE = 0.01        # only gate segments holding >=1% of p0 rows
GATE_DIMS = ("country", "city", "category", "order_type")

KIND_MAP = {"segment": "where", "factor": "factor", "compose": "factor",
            "mix": "ratemix"}


# --------------------------------------------------------------- Fix 2: gate
def completeness_gate(pg_url: str, schema: str) -> dict | None:
    """None if coverage is complete; else an abstain(data_gap) verdict with
    details. Pure row/day-count comparison of p0 vs p1."""
    eng = create_engine(pg_url, connect_args={
        "options": f"-csearch_path={schema} -cdefault_transaction_read_only=on"})
    try:
        per = pd.read_sql(
            "SELECT period, COUNT(*) rows, COUNT(DISTINCT ts_day) days "
            "FROM orders GROUP BY period", eng).set_index("period")
        if not {"p0", "p1"} <= set(per.index):
            return {"action": "abstain", "abstain_reason": "data_gap",
                    "details": {"reason": "a whole period has zero rows"}}
        d0, d1 = int(per.loc["p0", "days"]), int(per.loc["p1", "days"])
        if d1 < d0 * DAY_COVERAGE_MIN:
            return {"action": "abstain", "abstain_reason": "data_gap",
                    "details": {"reason": "p1 is missing day(s) of data",
                                "days_p0": d0, "days_p1": d1,
                                "rows_p0": int(per.loc["p0", "rows"]),
                                "rows_p1": int(per.loc["p1", "rows"])}}
        min_rows = per.loc["p0", "rows"] * SEG_MIN_SHARE
        for dim in GATE_DIMS:
            seg = pd.read_sql(
                f"SELECT period, {dim} s, COUNT(*) rows, "
                f"COUNT(DISTINCT ts_day) days FROM orders "
                f"WHERE {dim} IS NOT NULL GROUP BY period, {dim}", eng)
            piv = seg.pivot(index="s", columns="period")
            for s in piv.index:
                r0 = piv.loc[s].get(("rows", "p0"), 0)
                if pd.isna(r0) or r0 < min_rows:
                    continue                    # tiny/new segment: not gated
                days_seg0 = piv.loc[s].get(("days", "p0"), 0)
                if pd.isna(days_seg0) or days_seg0 < d0 * DAY_COVERAGE_MIN:
                    continue                    # wasn't daily-active in p0
                r1 = piv.loc[s].get(("rows", "p1"), 0)
                days_seg1 = piv.loc[s].get(("days", "p1"), 0)
                r1 = 0 if pd.isna(r1) else r1
                days_seg1 = 0 if pd.isna(days_seg1) else days_seg1
                if r1 == 0:
                    return {"action": "abstain", "abstain_reason": "data_gap",
                            "details": {"reason": f"{dim}={s} has ZERO rows in "
                                                  f"p1 (a lost source, not a "
                                                  f"demand drop)",
                                        "rows_p0": int(r0)}}
                if days_seg1 < d1 * DAY_COVERAGE_MIN:
                    return {"action": "abstain", "abstain_reason": "data_gap",
                            "details": {"reason": f"{dim}={s} stops mid-period "
                                                  f"in p1 (truncated load)",
                                        "days_p1_segment": int(days_seg1),
                                        "days_p1_total": d1}}
        return None
    finally:
        eng.dispose()


# ------------------------------------------------- Fix 1: structured result
def result_to_json(res, src) -> dict:
    """Thin adapter over inferah_engine's Result object (the package itself
    is untouched). Everything the narrator may cite is in here."""
    base = src.aggregate("0", "gmv")
    out = {
        "headline": {"delta_abs": round(res.parent_delta, 2),
                     "delta_pct": round(res.parent_pct, 2),
                     "gmv_p0": round(base, 2),
                     "gmv_p1": round(base + res.parent_delta, 2)},
        "significant": res.significant,
        "abstained": res.abstained,
        "confidence": res.confidence,
        "leaf": res.leaf,
        "unreconciled": res.unreconciled or None,
        "drill_path": [],
        "steps": [],
    }
    for s in res.steps:
        step = {"kind": KIND_MAP[s.kind], "measure": s.measure,
                "winner": s.winner,
                "winner_share": round(s.winner_share, 3),
                "reconcile": "OK" if s.recon.ok else "FAIL",
                "contributions": {str(k): round(v, 4)
                                  for k, v in s.parts.items()}}
        if s.kind == "segment" and s.winner_share >= 0.55:
            # a drill LOCALIZES only if the winner's share of the move clearly
            # exceeds its share of baseline volume; a segment that holds 80%
            # of the volume trivially holds ~80% of any uniform move — that
            # is proportionality, not localization, and must not become the
            # answer's dimension/segment.
            dim = s.scope.split("by")[-1].strip()
            seg0 = src.segment("0", s.measure, dim,
                               _filters_from_scope(s.scope))
            tot0 = sum(seg0.values()) or 1
            weight = seg0.get(s.winner, 0) / tot0
            if s.winner_share > weight + 0.15:
                out["drill_path"].append({"dimension": dim,
                                          "segment": s.winner,
                                          "move_share": round(s.winner_share, 3),
                                          "volume_share_p0": round(weight, 3)})
        if s.kind == "mix":
            dim = s.scope.split("rate vs mix by")[-1].strip()
            t0 = src.ratio_segment("0", s.measure, dim,
                                   _filters_from_scope(s.scope))
            t1 = src.ratio_segment("1", s.measure, dim,
                                   _filters_from_scope(s.scope))
            d0 = sum(d for d, _ in t0.values()) or 1
            d1 = sum(d for d, _ in t1.values()) or 1
            step["ratemix_dimension"] = dim
            step["segments"] = {
                str(k): {"weight_p0": round(t0.get(k, (0, 0))[0] / d0, 3),
                         "weight_p1": round(t1.get(k, (0, 0))[0] / d1, 3),
                         "rate_p0": round(t0[k][1] / t0[k][0], 2)
                                    if t0.get(k, (0, 0))[0] else None,
                         "rate_p1": round(t1[k][1] / t1[k][0], 2)
                                    if t1.get(k, (0, 0))[0] else None}
                for k in sorted(set(t0) | set(t1), key=str)}
        out["steps"].append(step)
    return out


ARM_D2_SYSTEM = f"""\
You answer "why did the metric change?" questions about an e-commerce orders
dataset. You have NO SQL. Your single tool is investigate(): a deterministic,
reconciliation-gated decomposition engine over the frozen identity tree
GMV = orders x AOV (orders = buyers x freq; AOV split rate-vs-mix by
order_type; WHERE axes: country, city, category). It returns JSON. Call it
once, then FILL the answer schema from that JSON mechanically, by these
rules — do not analyze, do not add anything the JSON does not contain:

1. If the JSON has action="abstain" with abstain_reason="data_gap" (the
   completeness gate fired): answer action="abstain",
   abstain_reason="data_gap", drivers=[]. Cite numbers from its details.
2. Else if "significant" is false: action="no_driver", drivers=[].
3. Else if "abstained" is true OR any step has reconcile="FAIL":
   action="abstain", abstain_reason="unmapped_dimension", drivers=[].
4. Else action="explain" with EXACTLY ONE driver — the terminal verdict,
   never the walked path:
   * factor + mechanism from the LAST step's winner:
     orders/buyers/freq -> that factor, mechanism "volume";
     aov_rate -> factor "aov", mechanism "rate";
     aov_mix  -> factor "aov", mechanism "mix".
   * dimension/segment: the LAST entry of drill_path, if any, else null/null.
     EXCEPTION for aov_mix: dimension = the step's ratemix_dimension and
     segment = the segment whose weight GREW (weight_p1 > weight_p0).
   * share_of_move = the last step's winner_share, capped at 1.0.
5. cited_numbers: copy values verbatim from the JSON (headline delta_abs /
   delta_pct / gmv_p0 / gmv_p1, contributions, weights). Never compute new
   numbers.
6. narrative: one sentence restating the driver and headline delta_pct.

{ANSWER_PROTOCOL}"""

INVESTIGATE2_TOOL = {
    "name": "investigate",
    "description": "Run the completeness-gated deterministic decomposition "
                   "engine (baseline p0 vs current p1). Returns structured "
                   "JSON: either a data-gap abstain verdict, or the verified "
                   "step-by-step decomposition.",
    "input_schema": {
        "type": "object",
        "properties": {
            "p0": {"type": "string", "enum": ["p0"]},
            "p1": {"type": "string", "enum": ["p1"]},
        },
    },
}


def make_investigate2(pg_url: str, schema: str):
    pack = load_pack(PACK_PATH)

    def run_investigate(p0: str = "p0", p1: str = "p1") -> str:
        gate = completeness_gate(pg_url, schema)
        if gate is not None:
            return json.dumps(gate, ensure_ascii=False)
        src = PostgresSource(pg_url, f"{schema}.orders", "ts_day", PERIODS,
                             pack.measures)
        try:
            res = investigate(src, pack.tree)
            return json.dumps(result_to_json(res, src), ensure_ascii=False)
        finally:
            src.engine.dispose()

    return run_investigate


def run_engine_arm2(question: str, pg_url: str, schema: str,
                    client=None, model=None) -> ArmResult:
    tool = make_investigate2(pg_url, schema)
    return run_tool_loop(ARM_D2_SYSTEM, question, [INVESTIGATE2_TOOL],
                         {"investigate": tool}, max_tool_calls=3,
                         client=client, model=model or MODEL)
