"""
Recover rate-limit casualties for one model: merge all shard files for that
model into raw.jsonl (dropping RUNNER ERROR records), then re-run ONLY the
missing (case, arm, run) cells SERIALLY — the cache keeps every success, so a
single-threaded pass with SDK retries fills the gaps without tripping
tier-1 RPM limits.

    python -m bench.recover gpt-5.5
"""
from __future__ import annotations

import json
import pathlib
import sys

from bench.config import pg_url
from bench.runner import RAW_PATH, ROOT, run_benchmark

model = sys.argv[1]
labels = json.loads((ROOT / "cases" / "labels.json").read_text())

# 1) merge raw.jsonl + every shard, dedup by (case,arm,run,model), drop errors
seen, recs = {}, []
files = [RAW_PATH, *sorted((ROOT / "results").glob("shard_*.jsonl"))]
for p in files:
    if not p.exists():
        continue
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "RUNNER ERROR" in (r.get("raw_text") or ""):
            continue                       # never let an error shadow a retry
        k = (r["case_id"], r["arm"], r["run_idx"], r.get("model"))
        seen[k] = r
recs = list(seen.values())
RAW_PATH.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                              for r in sorted(recs, key=lambda r: (
                                  r["case_id"], r["arm"], r["run_idx"],
                                  r.get("model", "")))) + "\n")
have = sum(1 for r in recs if r.get("model") == model)
print(f"merged clean cache: {len(recs)} records ({have} for {model})")

# 2) serial re-run of whatever is still missing for this model (arms A/B)
run_benchmark(labels, pg_url(), arms=("A", "B"), n_runs=5, model=model,
              progress=True)
done = sum(1 for line in RAW_PATH.read_text().splitlines()
           if line.strip() and json.loads(line).get("model") == model
           and "RUNNER ERROR" not in line)
print(f"{model}: {done}/280 cells present after recovery")
