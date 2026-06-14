"""
Plug your own agent into inferah-bench.

Implement ONE method — `answer(question, conn) -> dict` — returning the
benchmark answer schema, and the scorer treats your agent exactly like the
built-in arms. `conn` is a read-only SQLAlchemy connection scoped to one
case's schema; `question` is always "Why did GMV change between p0 and p1?".

Run this file directly to score your agent on all 28 cases:

    python -m examples.custom_agent

The answer schema (see cases/labels.schema.json for the full contract):

    {
      "action": "explain" | "abstain" | "no_driver",
      "drivers": [{"dimension","segment","factor","mechanism","share_of_move"}],
      "abstain_reason": "unmapped_dimension" | "data_gap" | null,
      "cited_numbers": [{"label": "...", "value": <float>}],
      "narrative": "<one sentence>"
    }
"""
from __future__ import annotations

import json

import pandas as pd
from sqlalchemy import create_engine, text

from bench.config import pg_url
from bench.runner import ROOT
from bench.scoring import build_fact_bank, score_run


class Agent:
    """Replace the body of answer() with your logic. This baseline just reads
    the headline delta and always abstains — a deliberately weak example."""

    def answer(self, question: str, conn) -> dict:
        df = pd.read_sql(text("SELECT period, SUM(gmv) g FROM orders "
                              "GROUP BY period"), conn).set_index("period")
        g0, g1 = float(df.loc["p0", "g"]), float(df.loc["p1", "g"])
        return {
            "action": "abstain",
            "drivers": [],
            "abstain_reason": "unmapped_dimension",
            "cited_numbers": [{"label": "gmv_p0", "value": g0},
                              {"label": "gmv_p1", "value": g1}],
            "narrative": f"GMV moved {(g1-g0)/g0*100:+.1f}%; baseline agent "
                         f"does not attempt to localize.",
        }


def evaluate(agent: Agent):
    labels = json.loads((ROOT / "cases" / "labels.json").read_text())
    url = pg_url()
    total = 0.0
    for lbl in labels:
        cid = lbl["case_id"]
        eng = create_engine(url, connect_args={
            "options": f"-csearch_path={cid} -cdefault_transaction_read_only=on"})
        with eng.connect() as conn:
            ans = agent.answer(lbl["question"], conn)
        facts = build_fact_bank(url, cid)
        s = score_run(lbl, ans, facts)
        total += s["score"]
        print(f"{cid} [{lbl['type']}] {s['score']:.2f}  {ans['action']}")
        eng.dispose()
    print(f"\nMEAN SCORE: {total/len(labels):.3f}  (over {len(labels)} cases)")


if __name__ == "__main__":
    evaluate(Agent())
