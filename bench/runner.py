"""
Orchestration: cases x arms x runs, with a raw-answer cache so re-running the
notebook never re-burns tokens.

Cache: results/raw.jsonl, one record per (case_id, arm, run_idx, model). A
record found in the cache is returned as-is; only missing combinations hit
the API. Delete a line (or the file) to force a re-run.
"""
from __future__ import annotations

import json
import pathlib
import time

from tqdm.auto import tqdm

from bench.agent import MODEL, ArmResult, cost_usd, run_sql_arm
from bench.engine_arm import run_engine_arm
from bench.scoring import build_fact_bank, canonical_key, score_run, stability

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "results" / "raw.jsonl"
ARMS = ("A", "B", "D")
N_RUNS = 5


def _key(case_id, arm, run_idx, model=MODEL):
    return f"{case_id}|{arm}|{run_idx}|{model}"


def load_cache(path=RAW_PATH) -> dict:
    cache = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                cache[_key(rec["case_id"], rec["arm"], rec["run_idx"],
                           rec.get("model", MODEL))] = rec
    return cache


def _run_one(case_id, arm, question, pg_url, model=MODEL) -> ArmResult:
    if arm in ("A", "B"):
        from bench.llm import provider_of
        # Anthropic models (incl. non-default like Opus) use the native loop;
        # other providers go through the multi-provider transport.
        if provider_of(model) == "anthropic":
            return run_sql_arm(arm, question, pg_url, schema=case_id,
                               model=model)
        from bench.llm import run_sql_arm_multi
        return run_sql_arm_multi(arm, question, pg_url, schema=case_id,
                                 model=model)
    if arm == "D":
        return run_engine_arm(question, pg_url, schema=case_id)
    if arm == "D2":
        from bench.engine_arm2 import run_engine_arm2
        return run_engine_arm2(question, pg_url, schema=case_id)
    if arm == "D3":
        from bench.engine_arm3 import run_engine_arm3
        return run_engine_arm3(question, pg_url, schema=case_id)
    raise ValueError(f"unknown arm {arm!r} (arm C is a v1 TODO)")


def run_benchmark(labels, pg_url, arms=ARMS, n_runs=N_RUNS, cases=None,
                  path=RAW_PATH, progress=True, model=MODEL):
    """Run (or resume) the grid for ONE model tag. Returns the raw records,
    cached and fresh alike. Appends fresh records to results/raw.jsonl as
    they land, so an interrupt loses at most one in-flight call."""
    labels = [l for l in labels if cases is None or l["case_id"] in cases]
    cache = load_cache(path)
    todo = [(l, arm, r) for l in labels for arm in arms for r in range(n_runs)
            if _key(l["case_id"], arm, r, model) not in cache]
    records = [cache[k] for k in cache
               if any(k == _key(l["case_id"], a, r, model) for l in labels
                      for a in arms for r in range(n_runs))]

    path.parent.mkdir(exist_ok=True)
    bar = tqdm(todo, disable=not progress, desc="bench")
    with open(path, "a") as fh:
        for label, arm, run_idx in bar:
            case_id = label["case_id"]
            bar.set_postfix_str(f"{case_id} {arm} run{run_idx}")
            t0 = time.time()
            try:
                res = _run_one(case_id, arm, label["question"], pg_url,
                               model=model)
                rec = {
                    "case_id": case_id, "arm": arm, "run_idx": run_idx,
                    "model": model,
                    "answer": res.answer, "parse_error": res.parse_error,
                    "raw_text": res.raw_text, "usage": res.usage,
                    "n_tool_calls": res.n_tool_calls,
                    "transcript": res.transcript,
                    "elapsed_s": round(time.time() - t0, 1),
                }
            except Exception as e:                      # record, don't die
                rec = {
                    "case_id": case_id, "arm": arm, "run_idx": run_idx,
                    "model": model, "answer": None, "parse_error": True,
                    "raw_text": f"RUNNER ERROR: {type(e).__name__}: {e}",
                    "usage": {"input_tokens": 0, "output_tokens": 0,
                              "cache_read": 0, "cache_write": 0},
                    "n_tool_calls": 0, "transcript": [],
                    "elapsed_s": round(time.time() - t0, 1),
                }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            records.append(rec)
    return records


# ----------------------------------------------------------------- scoring
def fact_banks(labels, pg_url, cases=None) -> dict:
    out = {}
    for l in labels:
        if cases is None or l["case_id"] in cases:
            out[l["case_id"]] = build_fact_bank(pg_url, l["case_id"])
    return out


def score_records(records, labels, banks) -> "pd.DataFrame":
    """One scored row per raw record + stability columns merged per
    (case, arm). Returns a tidy DataFrame ready for pivoting."""
    import pandas as pd

    from bench.llm import cost_for

    by_id = {l["case_id"]: l for l in labels}
    rows = []
    for rec in records:
        label = by_id[rec["case_id"]]
        s = score_run(label, rec["answer"], banks[rec["case_id"]],
                      parse_error=rec["parse_error"])
        model = rec.get("model", MODEL)
        rows.append({
            "case_id": rec["case_id"], "type": label["type"],
            "arm": rec["arm"], "model": model, "run_idx": rec["run_idx"],
            "canonical": canonical_key(rec["answer"], rec["parse_error"]),
            "n_tool_calls": rec["n_tool_calls"],
            "cost_usd": round(cost_for(model, rec["usage"]), 4),
            "input_tokens": rec["usage"]["input_tokens"],
            "output_tokens": rec["usage"]["output_tokens"],
            **s,
        })
    df = pd.DataFrame(rows)
    stab = (df.groupby(["case_id", "arm", "model"])["canonical"]
              .apply(lambda ks: pd.Series(stability(list(ks))))
              .unstack().reset_index())
    return df.merge(stab, on=["case_id", "arm", "model"])


def usage_summary(records) -> dict:
    tot = {"input_tokens": 0, "output_tokens": 0, "cache_read": 0,
           "cache_write": 0}
    for r in records:
        for k in tot:
            tot[k] += r["usage"].get(k, 0)
    tot["cost_usd"] = round(cost_usd(tot), 2)
    tot["n_calls"] = len(records)
    return tot
