"""
Arm D: the LLM gets NO SQL — its single tool is `investigate()`, a wrapper
over inferah_engine.investigate() running the frozen generic order pack
(cases/order_pack.yaml) against the case's Postgres schema. The model's job
is translation only: call the tool, then phrase its verified output into the
shared answer JSON. Every number it can cite comes from the engine.

The pack is ONE tree for the whole order grain, defined before any case was
generated — not tuned to any injected cause (see README: that would be
cheating).
"""
from __future__ import annotations

import io
import re

from inferah_engine import investigate, render, narrate
from inferah_engine.datasource import PostgresSource
from inferah_engine.loader import load_pack

from bench.agent import MODEL, HERE, ANSWER_PROTOCOL, ArmResult, run_tool_loop

PACK_PATH = str(HERE.parent / "cases" / "order_pack.yaml")
PERIODS = {"0": ("2026-05-04", "2026-05-11"),
           "1": ("2026-05-11", "2026-05-18")}

ARM_D_SYSTEM = f"""\
You answer "why did the metric change?" questions about an e-commerce orders
dataset. You have NO SQL access. Your single tool is investigate(): a
deterministic, reconciliation-gated decomposition engine that walks the
frozen identity tree GMV = orders x AOV (orders = buyers x freq; AOV split
into rate vs mix by order_type; WHERE axes: country, city, category) and
verifies that every step adds up. Call it once, read its verified output,
and translate that output into the answer JSON. Do not invent anything the
engine did not verify:

* If the engine ABSTAINS, your action is "abstain" (reason
  "unmapped_dimension" if it names an unexplained/unmapped or diffuse move,
  "data_gap" if the data is incomplete).
* If the engine reports the change as not significant, action="no_driver".
* Otherwise action="explain" with drivers read off the engine's winning
  path (the drilled WHERE segments and the winning factor/rate/mix leaf).
* cited_numbers may contain ONLY numbers present in the engine output.

{ANSWER_PROTOCOL}"""

INVESTIGATE_TOOL = {
    "name": "investigate",
    "description": "Run the deterministic decomposition engine over the "
                   "orders data for baseline p0 vs current p1. Returns the "
                   "verified, reconciliation-gated investigation report.",
    "input_schema": {
        "type": "object",
        "properties": {
            "p0": {"type": "string", "description": "baseline period (p0)",
                   "enum": ["p0"]},
            "p1": {"type": "string", "description": "current period (p1)",
                   "enum": ["p1"]},
        },
    },
}


def _filters_from_scope(scope: str) -> dict:
    """Recover the engine's drill filters from a Step.scope string like
    'all ∩ country=Norvik ∩ city=Brimley · rate vs mix by order_type'."""
    scope = scope.split("·")[0]
    return dict(m.groups() for m in re.finditer(r"∩\s*(\w+)=([\w-]+)", scope))


def _mix_detail(src, res) -> str:
    """For rate/mix steps, append the verified per-segment weight/rate table
    so the narrator can name segments without inventing them."""
    out = io.StringIO()
    for s in res.steps:
        if s.kind != "mix":
            continue
        dim = s.scope.split("rate vs mix by")[-1].strip()
        filters = _filters_from_scope(s.scope)
        t0 = src.ratio_segment("0", s.measure, dim, filters)
        t1 = src.ratio_segment("1", s.measure, dim, filters)
        out.write(f"\nVERIFIED DETAIL — {s.scope}:\n")
        out.write(f"  {'segment':<12}{'w0':>8}{'w1':>8}{'rate0':>10}{'rate1':>10}\n")
        d0, d1 = sum(d for d, _ in t0.values()) or 1, sum(d for d, _ in t1.values()) or 1
        for k in sorted(set(t0) | set(t1)):
            den0, num0 = t0.get(k, (0.0, 0.0))
            den1, num1 = t1.get(k, (0.0, 0.0))
            out.write(f"  {str(k):<12}{den0/d0:>8.3f}{den1/d1:>8.3f}"
                      f"{(num0/den0 if den0 else 0):>10.2f}"
                      f"{(num1/den1 if den1 else 0):>10.2f}\n")
    return out.getvalue()


def make_investigate(pg_url: str, schema: str):
    pack = load_pack(PACK_PATH)
    src = PostgresSource(pg_url, f"{schema}.orders", "ts_day", PERIODS,
                         pack.measures)

    def run_investigate(p0: str = "p0", p1: str = "p1") -> str:
        try:
            res = investigate(src, pack.tree)    # periods "0"/"1" = p0/p1
            report = render(res, metric_label="GMV")
            return (report + "\n" + _mix_detail(src, res)
                    + "\n> " + narrate(res, metric_label="GMV"))
        finally:
            src.engine.dispose()    # don't hold idle connections between runs

    return run_investigate


def run_engine_arm(question: str, pg_url: str, schema: str,
                   client=None, model=None) -> ArmResult:
    tool = make_investigate(pg_url, schema)
    return run_tool_loop(ARM_D_SYSTEM, question, [INVESTIGATE_TOOL],
                         {"investigate": tool}, max_tool_calls=3,
                         client=client, model=model or MODEL)
