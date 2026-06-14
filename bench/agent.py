"""
Arms A and B: an LLM data-analyst agent with a single read-only `run_sql`
tool against the case's Postgres schema.

  * Arm A ("bare"): context = table + column names only.
  * Arm B ("docs"): same + the full semantic layer doc (cases/docs/).

Both arms get the identical question, the identical answer protocol, and a
hard budget of MAX_SQL_CALLS queries. Temperature 0. The prompts are FROZEN —
do not tune them after looking at results (that's p-hacking; see README).
"""
from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import create_engine, text

# `anthropic` / `httpx` are imported lazily inside run_tool_loop so the
# engine-only arms (D3, pure code, $0) and the scorer run with no LLM SDK
# installed — `make seed && make report` needs no API client at all.

HERE = pathlib.Path(__file__).resolve().parent
DOCS = (HERE.parent / "cases" / "docs" / "semantic_layer.md").read_text()

MODEL = "claude-sonnet-4-6"
# The spec pinned claude-sonnet-4-20250514, which Anthropic retires on
# 2026-06-15 — days after this benchmark was built. v0 pins sonnet-4-6 so the
# runs stay reproducible (decision recorded in README).
TEMPERATURE = 0.0
MAX_TOKENS = 2000
MAX_SQL_CALLS = 15
PRICE_PER_MTOK = {"input": 3.00, "output": 15.00}   # claude-sonnet-4-6, USD

SCHEMA_COLUMNS = [
    ("order_id", "integer"), ("ts_day", "date"), ("period", "text"),
    ("country", "text"), ("city", "text"), ("order_type", "text"),
    ("category", "text"), ("gmv", "numeric"), ("user_id", "text"),
]

ANSWER_PROTOCOL = """\
When you are done, respond with ONLY a JSON object (no prose, no markdown
fences) in exactly this shape:

{
  "action": "explain" | "abstain" | "no_driver",
  "drivers": [
    {"dimension": "country|city|category|order_type" or null,
     "segment": "<value>" or null,
     "factor": "orders|buyers|freq|aov" or null,
     "mechanism": "rate" | "mix" | "volume",
     "share_of_move": <float 0..1>}
  ],
  "abstain_reason": "unmapped_dimension" | "data_gap" | null,
  "cited_numbers": [{"label": "<what it is>", "value": <float>}],
  "narrative": "<one sentence>"
}

Semantics:
* action: "explain" if you can name the driver(s); "abstain" if the data
  cannot honestly localize the cause (unmapped/NULL segment rows, incomplete
  data load); "no_driver" if the metric move is not significant / within
  noise.
* drivers: only for action="explain". Name the MOST SPECIFIC segment where
  the move concentrates (dimension+segment), the factor of
  GMV = orders x AOV (orders = buyers x freq) that carries it, and the
  mechanism: "volume" (count of orders/buyers moved), "rate" (per-segment
  value genuinely changed), "mix" (composition shifted between segments
  whose own rates are unchanged). For a mix driver, segment = the segment
  whose SHARE GREW. dimension/segment are null when the move is broad-based
  rather than concentrated. share_of_move values must sum to ~1.0.
* abstain_reason: "unmapped_dimension" when the move sits in rows no modeled
  column accounts for (or is uniformly everywhere, i.e. external);
  "data_gap" when rows are missing (lost day / lost source).
* cited_numbers: every number you state, exactly as computed from the data.
  Do not round beyond one decimal. Do not invent numbers.
"""

ARM_A_SYSTEM = f"""\
You are a careful senior data analyst. You answer "why did the metric
change?" questions about an e-commerce orders table in Postgres.

You have one tool: run_sql(query) — read-only SQL against the table
`orders`. Columns: {", ".join(f"{c} ({t})" for c, t in SCHEMA_COLUMNS)}.
You may run at most {MAX_SQL_CALLS} queries; make them count.

{ANSWER_PROTOCOL}"""

ARM_B_SYSTEM = f"""\
You are a careful senior data analyst. You answer "why did the metric
change?" questions about an e-commerce orders table in Postgres.

You have one tool: run_sql(query) — read-only SQL against the table
`orders`. You may run at most {MAX_SQL_CALLS} queries; make them count.

Below is the complete, authoritative documentation of the dataset. Follow
its cautions — especially the mix-effect check and the honest-answer rules.

<documentation>
{DOCS}
</documentation>

{ANSWER_PROTOCOL}"""

RUN_SQL_TOOL = {
    "name": "run_sql",
    "description": "Run one read-only SQL SELECT against the orders table. "
                   "Returns up to 50 rows as CSV. No DDL/DML.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string",
                                 "description": "A single SELECT statement."}},
        "required": ["query"],
    },
}

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|grant|revoke|truncate|copy|"
    r"vacuum|analyze|call|do|set|reset|listen|notify)\b", re.I)


@dataclass
class ArmResult:
    answer: dict | None
    raw_text: str
    parse_error: bool
    usage: dict
    n_tool_calls: int
    transcript: list = field(default_factory=list)


def make_run_sql(pg_url: str, schema: str):
    """A read-only executor pinned to one case schema. Enforced three ways:
    statement allow-list, Postgres read-only transaction, and search_path."""
    # NullPool: no idle connections held between queries — dozens of runs per
    # process otherwise exhaust Postgres max_connections.
    from sqlalchemy.pool import NullPool
    eng = create_engine(pg_url, poolclass=NullPool, connect_args={
        "options": f"-csearch_path={schema} -cdefault_transaction_read_only=on"
    })

    def run_sql(query: str) -> str:
        q = re.sub(r"--.*$", "", query, flags=re.M).strip().rstrip(";").strip()
        if not re.match(r"^(select|with)\b", q, re.I) or ";" in q:
            return "ERROR: only a single SELECT statement is allowed."
        if _FORBIDDEN.search(q):
            return "ERROR: read-only SELECT statements only."
        try:
            with eng.connect() as c:
                c.execute(text("SET statement_timeout = 10000"))
                df = pd.read_sql(text(q), c)
        except Exception as e:  # surface the DB error to the model verbatim
            return f"ERROR: {type(e).__name__}: {str(e).splitlines()[0][:300]}"
        if len(df) > 50:
            return (df.head(50).to_csv(index=False)
                    + f"... ({len(df) - 50} more rows truncated)")
        return df.to_csv(index=False) if len(df) else "(0 rows)"

    return run_sql


def _extract_json(txt: str):
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", txt, re.S)
    cand = m.group(1) if m else None
    if cand is None:
        start = txt.find("{")
        if start >= 0:
            depth = 0
            for i, ch in enumerate(txt[start:], start):
                depth += ch == "{"
                depth -= ch == "}"
                if depth == 0:
                    cand = txt[start:i + 1]
                    break
    if cand is None:
        raise ValueError("no JSON object found")
    return json.loads(cand)


def _add_usage(total: dict, usage) -> None:
    total["input_tokens"] += usage.input_tokens
    total["output_tokens"] += usage.output_tokens
    total["cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0
    total["cache_write"] += getattr(usage, "cache_creation_input_tokens", 0) or 0


def cost_usd(usage: dict) -> float:
    """Token cost. The API's input_tokens already EXCLUDES cached tokens:
    cache reads bill at 0.1x input price, cache writes at 1.25x."""
    return (usage["input_tokens"] / 1e6 * PRICE_PER_MTOK["input"]
            + usage["cache_read"] / 1e6 * PRICE_PER_MTOK["input"] * 0.1
            + usage["cache_write"] / 1e6 * PRICE_PER_MTOK["input"] * 1.25
            + usage["output_tokens"] / 1e6 * PRICE_PER_MTOK["output"])


def run_tool_loop(system: str, question: str, tools: list, executors: dict,
                  max_tool_calls: int, client=None) -> ArmResult:
    """Generic manual agentic loop: run tools until the model stops or the
    budget runs out, then force a final JSON answer. One re-ask on bad JSON."""
    # Hard timeouts: a wedged TCP connection must fail fast and retry, not
    # hang a shard for the SDK's default 10 minutes (observed in the wild).
    if client is None:
        import anthropic
        import httpx
        client = anthropic.Anthropic(
            timeout=httpx.Timeout(120.0, connect=10.0), max_retries=4)
    sys_block = [{"type": "text", "text": system,
                  "cache_control": {"type": "ephemeral"}}]
    messages = [{"role": "user", "content": question}]
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read": 0,
             "cache_write": 0}
    transcript, n_calls = [], 0

    def call(**kw):
        resp = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            system=sys_block, messages=messages, **kw)
        _add_usage(usage, resp.usage)
        return resp

    while True:
        resp = call(tools=tools)
        if resp.stop_reason != "tool_use":
            break
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            n_calls += 1
            over = n_calls > max_tool_calls
            out = ("ERROR: tool budget exhausted — produce your final JSON "
                   "answer now." if over
                   else executors[block.name](**block.input))
            transcript.append({"tool": block.name, "input": block.input,
                               "output": out[:2000]})
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": out})
        messages.append({"role": "user", "content": results})
        if n_calls > max_tool_calls:
            messages.append({"role": "user",
                             "content": "Tool budget exhausted. Respond with "
                                        "the final JSON object now."})
            resp = call(tools=tools, tool_choice={"type": "none"})
            break

    raw = "".join(b.text for b in resp.content if b.type == "text")
    answer, parse_error = None, False
    try:
        answer = _extract_json(raw)
    except Exception:
        messages.append({"role": "assistant", "content": raw or "(empty)"})
        messages.append({"role": "user",
                         "content": "Return ONLY the JSON object matching the "
                                    "required schema. No other text."})
        resp = call(tools=tools, tool_choice={"type": "none"})
        raw = "".join(b.text for b in resp.content if b.type == "text")
        try:
            answer = _extract_json(raw)
        except Exception:
            parse_error = True
    return ArmResult(answer=answer, raw_text=raw, parse_error=parse_error,
                     usage=usage, n_tool_calls=n_calls, transcript=transcript)


def run_sql_arm(arm: str, question: str, pg_url: str, schema: str,
                client=None) -> ArmResult:
    """arm 'A' (bare) or 'B' (docs)."""
    system = {"A": ARM_A_SYSTEM, "B": ARM_B_SYSTEM}[arm]
    run_sql = make_run_sql(pg_url, schema)
    return run_tool_loop(system, question, [RUN_SQL_TOOL],
                         {"run_sql": run_sql}, MAX_SQL_CALLS, client=client)
