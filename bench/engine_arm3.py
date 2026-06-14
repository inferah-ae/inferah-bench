"""
Arm D3 (v0.3 Part 2): the CODE narrator — zero LLM in the answer loop.

result_to_answer() is a pure, deterministic function implementing exactly the
mapping rules that arm D2's prompt asks the LLM to follow (terminal winner
only, localizing drill path, reconcile FAIL -> abstain, grown segment for
mix). What D2 requests in natural language, D3 does in code — so its
stability is 1.00 by construction and a run costs $0.

If D3 scores below D2 anywhere, that means the LLM narrator was silently
REPAIRING the engine's output — which would be important to know; the
notebook compares them case by case.
"""
from __future__ import annotations

import json

from bench.agent import ArmResult
from bench.engine_arm2 import make_investigate2

LEAF_MAP = {
    "orders": ("orders", "volume"),
    "buyers": ("buyers", "volume"),
    "freq": ("freq", "volume"),
    "aov_rate": ("aov", "rate"),
    "aov_mix": ("aov", "mix"),
    "aov": ("aov", "rate"),
}


def result_to_answer(tool_json: dict) -> dict:
    """Deterministic mapping: engine JSON -> benchmark answer schema."""
    # Rule 1: completeness gate verdict passes through
    if tool_json.get("action") == "abstain":
        d = tool_json.get("details", {})
        return {
            "action": "abstain", "drivers": [], "abstain_reason": "data_gap",
            "cited_numbers": [{"label": k, "value": float(v)}
                              for k, v in d.items()
                              if isinstance(v, (int, float))],
            "narrative": f"Cannot attribute the change: {d.get('reason', 'incomplete data')}.",
        }

    head = tool_json["headline"]
    cite_head = [{"label": k, "value": float(v)} for k, v in head.items()]

    # Rule 2: not significant
    if not tool_json["significant"]:
        return {"action": "no_driver", "drivers": [], "abstain_reason": None,
                "cited_numbers": cite_head,
                "narrative": f"GMV moved {head['delta_pct']:+.2f}% — below "
                             f"the significance threshold; no driver to name."}

    # Rule 3: engine abstain or any reconcile FAIL
    steps = tool_json["steps"]
    if tool_json["abstained"] or any(s["reconcile"] == "FAIL" for s in steps):
        return {"action": "abstain", "drivers": [],
                "abstain_reason": "unmapped_dimension",
                "cited_numbers": cite_head,
                "narrative": f"GMV moved {head['delta_pct']:+.2f}% but the "
                             f"decomposition cannot localize it: "
                             f"{tool_json.get('unreconciled') or tool_json['leaf']}"}

    # Rule 4: explain — exactly one driver, the terminal verdict
    last = steps[-1]
    factor, mechanism = LEAF_MAP.get(last["winner"],
                                     (last["winner"], "volume"))
    dimension = segment = None
    if mechanism == "mix":
        dimension = last.get("ratemix_dimension")
        grown = [(k, v) for k, v in (last.get("segments") or {}).items()
                 if (v.get("weight_p1") or 0) > (v.get("weight_p0") or 0)]
        if grown:
            segment = max(grown,
                          key=lambda kv: kv[1]["weight_p1"] - kv[1]["weight_p0"])[0]
    elif tool_json["drill_path"]:
        deepest = tool_json["drill_path"][-1]
        dimension, segment = deepest["dimension"], deepest["segment"]

    share = round(min(last["winner_share"], 1.0), 3)
    cited = cite_head + [{"label": f"contribution_{k}", "value": float(v)}
                         for k, v in last["contributions"].items()]
    where = f" in {dimension}={segment}" if segment else ""
    return {
        "action": "explain",
        "drivers": [{"dimension": dimension, "segment": segment,
                     "factor": factor, "mechanism": mechanism,
                     "share_of_move": share}],
        "abstain_reason": None,
        "cited_numbers": cited,
        "narrative": f"GMV moved {head['delta_pct']:+.2f}% driven by "
                     f"{factor} ({mechanism}){where}, holding "
                     f"{share * 100:.0f}% of the move.",
    }


def run_engine_arm3(question: str, pg_url: str, schema: str,
                    client=None) -> ArmResult:
    """No LLM: call the gated engine tool, map in code. $0, deterministic."""
    tool_json = json.loads(make_investigate2(pg_url, schema)())
    answer = result_to_answer(tool_json)
    return ArmResult(answer=answer, raw_text=json.dumps(answer),
                     parse_error=False,
                     usage={"input_tokens": 0, "output_tokens": 0,
                            "cache_read": 0, "cache_write": 0},
                     n_tool_calls=1,
                     transcript=[{"tool": "investigate",
                                  "input": {}, "output":
                                  json.dumps(tool_json)[:2000]}])
