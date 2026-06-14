"""One shard of the full grid: an arm + a case-id modulo slice.

    python -m bench.shard_run A 0 3   # arm A, cases where (idx % 3) == 0
"""
import json
import pathlib
import shutil
import sys

from bench.runner import ROOT, RAW_PATH, run_benchmark

URL = "postgresql+psycopg2://inferah:inferah@localhost:5433/inferah"

arm, residue, mod = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
model = sys.argv[4] if len(sys.argv) > 4 else "claude-sonnet-4-6"
tag = model.replace("/", "_").replace(".", "")
labels = json.loads((ROOT / "cases" / "labels.json").read_text())
cases = [l["case_id"] for i, l in enumerate(labels) if i % mod == residue]
shard_path = ROOT / "results" / f"shard_{arm}{residue}_{tag}.jsonl"
if not shard_path.exists() and RAW_PATH.exists():
    shutil.copy(RAW_PATH, shard_path)        # reuse cached records
run_benchmark(labels, URL, arms=(arm,), n_runs=5, cases=cases,
              path=shard_path, progress=False, model=model)
print(f"shard {arm}/{residue} [{model}] done: {len(cases)} cases")
