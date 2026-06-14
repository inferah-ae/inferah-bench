"""
Factor scoring — never a single binary:

  1. action_correct    explain/abstain/no_driver matches the label
  2. driver_score      main driver tuple (dimension, segment, factor,
                       mechanism); T5 compound: 0.5 partial credit per
                       matched driver; abstain cases score the reason
  3. mechanism_correct rate vs mix vs volume on the main driver (reported
                       separately — the heart of the Simpson test)
  4. grounding_pass    every cited number must exist in the case data within
                       2% tolerance, recomputed directly from Postgres
                       (gate-as-scorer #1)
  5. sum_share_pass    driver shares sum to [0.9, 1.1] for explains
                       (gate-as-scorer #2)
  6. stability         across the R runs of a (case, arm): modal-answer share
                       and number of unique canonical answers

Case score = 0.30*action + 0.40*driver + 0.20*grounding + 0.10*sum_share.
Mechanism and stability are reported as separate columns, not in the score.
A parse_error scores 0 on every component.

Driver matching details (documented, frozen):
  * strings are lowercased; ''/None/'null'/'overall'/'all' all mean "broad,
    no specific segment" and match each other;
  * factor matches exactly OR the answer names a CHILD of the expected
    factor in the fixed identity (orders -> buyers/freq): saying "buyers"
    when the label says "orders" is more specific along the same branch and
    counts; the reverse (too vague) does not;
  * for expected abstain: full credit needs action=abstain AND the right
    reason; abstain with the wrong reason earns 0.5.
"""
from __future__ import annotations

import math
from collections import Counter

GROUND_REL_TOL = 0.02
GROUND_ABS_TOL = 0.05
FACTOR_CHILDREN = {"orders": {"buyers", "freq"}, "gmv": {"orders", "aov"}}
_BROAD = {None, "", "null", "none", "overall", "all", "total"}


def _norm(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    return None if s in _BROAD else s


def _driver_tuple(d: dict):
    return (_norm(d.get("dimension")), _norm(d.get("segment")),
            _norm(d.get("factor")), _norm(d.get("mechanism")))


def _factor_matches(expected, answered):
    if expected == answered:
        return True
    return answered in FACTOR_CHILDREN.get(expected, set())


def _driver_matches(exp: dict, ans: dict) -> bool:
    e, a = _driver_tuple(exp), _driver_tuple(ans)
    return (e[0] == a[0] and e[1] == a[1]
            and _factor_matches(e[2], a[2]) and e[3] == a[3])


def _main_driver(drivers):
    if not drivers:
        return None
    return max(drivers, key=lambda d: d.get("share_of_move") or 0)


def driver_score(label: dict, answer: dict) -> float:
    exp = label["expected"]
    if exp["action"] == "no_driver":
        return 1.0 if answer.get("action") == "no_driver" else 0.0
    if exp["action"] == "abstain":
        if answer.get("action") != "abstain":
            return 0.0
        return 1.0 if (_norm(answer.get("abstain_reason"))
                       == _norm(exp["abstain_reason"])) else 0.5
    # expected explain
    if answer.get("action") != "explain":
        return 0.0
    ans_drivers = answer.get("drivers") or []
    if not isinstance(ans_drivers, list):
        return 0.0
    matched = sum(
        1 for e in exp["drivers"]
        if any(_driver_matches(e, a) for a in ans_drivers if isinstance(a, dict)))
    return matched / len(exp["drivers"])


def mechanism_correct(label: dict, answer: dict):
    """rate-vs-mix-vs-volume on the main driver; None when not applicable."""
    exp_main = _main_driver(label["expected"]["drivers"])
    if exp_main is None or answer.get("action") != "explain":
        return None
    ans_main = _main_driver([d for d in (answer.get("drivers") or [])
                             if isinstance(d, dict)])
    if ans_main is None:
        return False
    return _norm(ans_main.get("mechanism")) == _norm(exp_main.get("mechanism"))


def grounding_pass(answer: dict, facts: list[float]):
    """Every cited number must match SOME true value of the case within 2%
    relative tolerance (0.05 absolute floor for near-zero values)."""
    cited = answer.get("cited_numbers") or []
    if not isinstance(cited, list):
        return False
    for item in cited:
        try:
            v = float(item["value"])
        except (TypeError, KeyError, ValueError):
            return False
        if math.isnan(v):
            return False
        ok = any(abs(v - f) <= max(GROUND_ABS_TOL, GROUND_REL_TOL * abs(f))
                 for f in facts)
        if not ok:
            return False
    return True


def sum_share_pass(answer: dict):
    if answer.get("action") != "explain":
        return True
    try:
        total = sum(float(d.get("share_of_move", 0))
                    for d in (answer.get("drivers") or []))
    except (TypeError, ValueError):
        return False
    return 0.9 <= total <= 1.1


def action_correct(label: dict, answer: dict) -> bool:
    return answer.get("action") == label["expected"]["action"]


def score_run(label: dict, answer: dict | None, facts: list[float],
              parse_error: bool = False) -> dict:
    """All components for one (case, arm, run). parse_error zeroes the score."""
    if parse_error or not isinstance(answer, dict):
        return {"parse_error": True, "action_correct": False,
                "driver_score": 0.0, "mechanism_correct": None,
                "grounding_pass": False, "sum_share_pass": False,
                "score": 0.0}
    a = action_correct(label, answer)
    d = driver_score(label, answer)
    g = grounding_pass(answer, facts)
    s = sum_share_pass(answer)
    return {
        "parse_error": False,
        "action_correct": a,
        "driver_score": d,
        "mechanism_correct": mechanism_correct(label, answer),
        "grounding_pass": g,
        "sum_share_pass": s,
        "score": round(0.30 * a + 0.40 * d + 0.20 * g + 0.10 * s, 4),
    }


# ------------------------------------------------------------- stability
def canonical_key(answer: dict | None, parse_error: bool = False) -> str:
    """The 'same answer' identity used for stability: action + reason +
    the set of driver tuples (shares excluded — wobbling shares with the
    same drivers still count as the same verdict)."""
    if parse_error or not isinstance(answer, dict):
        return "parse_error"
    drivers = sorted(str(_driver_tuple(d))
                     for d in (answer.get("drivers") or [])
                     if isinstance(d, dict))
    return f"{answer.get('action')}|{_norm(answer.get('abstain_reason'))}|" \
           + ";".join(drivers)


def stability(keys: list[str]) -> dict:
    """keys = canonical keys of the R runs of one (case, arm)."""
    counts = Counter(keys)
    modal = counts.most_common(1)[0][1] if keys else 0
    return {"modal_share": modal / len(keys) if keys else 0.0,
            "n_unique": len(counts)}


# ------------------------------------------------------------- fact bank
DIMS = ("country", "city", "category", "order_type")


def build_fact_bank(pg_url: str, schema: str) -> list[float]:
    """Recompute, straight from Postgres, every value an honest answer could
    cite: per-period aggregates (overall and per segment of each dimension),
    deltas, percent deltas (both as percent and fraction, both signs),
    per-segment shares of the move, and day/row counts."""
    import pandas as pd
    from sqlalchemy import create_engine

    eng = create_engine(pg_url, connect_args={
        "options": f"-csearch_path={schema} -cdefault_transaction_read_only=on"})
    facts: list[float] = []

    def add(*vals):
        for v in vals:
            v = float(v)
            if math.isfinite(v):
                facts.append(round(v, 6))

    def add_period_pair(g0, g1):
        add(g0, g1, g1 - g0, g0 - g1)
        if g0:
            pct = (g1 - g0) / g0
            add(pct, -pct, pct * 100, -pct * 100)

    overall = pd.read_sql(
        "SELECT period, SUM(gmv) g, COUNT(*) o, COUNT(DISTINCT user_id) u, "
        "COUNT(DISTINCT ts_day) d FROM orders GROUP BY period", eng
    ).set_index("period")
    for col in ("g", "o", "u", "d"):
        add_period_pair(overall.loc["p0", col], overall.loc["p1", col])
    for p in ("p0", "p1"):
        days = overall.loc[p, "d"] or 1
        if overall.loc[p, "o"]:
            add(overall.loc[p, "g"] / overall.loc[p, "o"])          # AOV
        if overall.loc[p, "u"]:
            add(overall.loc[p, "o"] / overall.loc[p, "u"])          # freq
        add(overall.loc[p, "o"] / days, overall.loc[p, "g"] / days)  # per-day
    aov0 = overall.loc["p0", "g"] / overall.loc["p0", "o"]
    aov1 = overall.loc["p1", "g"] / overall.loc["p1", "o"]
    add_period_pair(aov0, aov1)
    total_move = overall.loc["p1", "g"] - overall.loc["p0", "g"]

    for dim in DIMS:
        seg = pd.read_sql(
            f"SELECT period, {dim} s, SUM(gmv) g, COUNT(*) o, AVG(gmv) a, "
            f"COUNT(DISTINCT user_id) u, COUNT(DISTINCT ts_day) d "
            f"FROM orders GROUP BY period, {dim}", eng)
        piv = seg.pivot(index="s", columns="period")

        def val(s, col, p):
            v = piv.loc[s].get((col, p), 0)
            return 0 if pd.isna(v) else (v or 0)

        # non-NULL ("mapped") totals + their complement of the overall
        mapped = seg[seg.s.notna()].groupby("period")[["g", "o"]].sum()
        for col in ("g", "o"):
            if {"p0", "p1"} <= set(mapped.index):
                add_period_pair(mapped.loc["p0", col], mapped.loc["p1", col])

        for s in piv.index:
            g0, g1 = val(s, "g", "p0"), val(s, "g", "p1")
            o0, o1 = val(s, "o", "p0"), val(s, "o", "p1")
            add_period_pair(g0, g1)
            add_period_pair(o0, o1)
            add_period_pair(val(s, "a", "p0"), val(s, "a", "p1"))
            add_period_pair(val(s, "u", "p0"), val(s, "u", "p1"))   # buyers
            for p, o, g in (("p0", o0, g0), ("p1", o1, g1)):
                d = val(s, "d", p) or 1
                u = val(s, "u", p)
                add(o / d, g / d)                                   # per-day
                if u:
                    add(o / u, g / o if o else 0)                   # freq, aov
            if total_move:
                share = (g1 - g0) / total_move
                add(share, -share, share * 100, -share * 100)
            tot0 = overall.loc["p0", "o"]
            tot1 = overall.loc["p1", "o"]
            if tot0 and tot1:                       # segment order shares
                add(o0 / tot0, o1 / tot1, (o0 / tot0) * 100, (o1 / tot1) * 100,
                    o1 / tot1 - o0 / tot0, (o1 / tot1 - o0 / tot0) * 100)
    eng.dispose()
    return sorted(set(facts))
