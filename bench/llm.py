"""
v0.3 Part 1 — provider-neutral transport for the arms A/B SQL loop.

The agentic loop SEMANTICS are frozen from v0 (see bench/agent.py):
at most MAX_SQL_CALLS tool calls, temperature 0, forced final JSON answer,
one "return only JSON" retry, then parse_error. This module changes ONLY the
transport underneath: the same frozen prompts and the same run_sql tool run
against Anthropic, OpenAI, or Google models. Tool schemas are converted per
provider; behavior is identical. Verification status: OpenAI (gpt-5.5) and
Google (gemini-3.1-pro-preview) each answered one case end to end; no full
non-Anthropic grid has been run.

Anthropic models keep going through bench/agent.run_tool_loop (the exact v0
code path) so prior results remain byte-comparable; this module routes only
non-Anthropic models.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from bench.agent import (ANSWER_PROTOCOL, ARM_A_SYSTEM, ARM_B_SYSTEM,
                         MAX_SQL_CALLS, MAX_TOKENS, RUN_SQL_TOOL, TEMPERATURE,
                         ArmResult, _extract_json, make_run_sql)

# USD per MTok (input, output, cached-input). Filled for the models actually
# run; unknown models cost 0 with a loud warning in the report.
PRICES = {
    "claude-sonnet-4-6": (3.00, 15.00, 0.30),
    "claude-opus-4-8": (5.00, 25.00, 0.50),
    "gpt-5.2": (1.25, 10.00, 0.125),
    "gpt-5.2-pro": (15.00, 120.00, 1.50),
    "gpt-5-mini": (0.25, 2.00, 0.025),
    # APPROXIMATE — exact list prices for these were not on hand at run time;
    # cost figures for them are estimates (flagged in the report / notebook).
    "gpt-5.5": (2.00, 16.00, 0.20),
    "gemini-3.1-pro-preview": (2.00, 12.00, 0.20),
}
APPROX_PRICED = {"gpt-5.5", "gemini-3.1-pro-preview"}


def cost_for(model: str, usage: dict) -> float:
    p = PRICES.get(model)
    if p is None:
        p = next((v for k, v in PRICES.items() if model.startswith(k)), None)
    if p is None:
        return 0.0
    inp, out, cached = p
    return (usage["input_tokens"] / 1e6 * inp
            + usage["output_tokens"] / 1e6 * out
            + usage.get("cache_read", 0) / 1e6 * cached
            + usage.get("cache_write", 0) / 1e6 * inp * 1.25)


def provider_of(model: str) -> str:
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    if model.startswith("gemini"):
        return "google"
    raise ValueError(f"can't infer provider for model {model!r}")


@dataclass
class Reply:
    kind: str                  # "tool" | "text"
    text: str
    tool_calls: list           # [(id, name, input_dict)]


class OpenAITransport:
    """Chat Completions with function calling; one instance = one run."""

    def __init__(self, model: str, system: str, tools: list):
        import openai
        self.client = openai.OpenAI(timeout=120.0, max_retries=4)
        self.model = model
        self.messages = [{"role": "system", "content": system}]
        self.tools = [{"type": "function",
                       "function": {"name": t["name"],
                                    "description": t["description"],
                                    "parameters": t["input_schema"]}}
                      for t in tools]
        self.usage = {"input_tokens": 0, "output_tokens": 0,
                      "cache_read": 0, "cache_write": 0}
        self._temperature_ok = True

    def add_user(self, text: str):
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results):
        for call_id, output in results:
            self.messages.append({"role": "tool", "tool_call_id": call_id,
                                  "content": output})

    def reply(self, force_text: bool = False) -> Reply:
        kwargs = dict(model=self.model, messages=self.messages,
                      max_completion_tokens=MAX_TOKENS,
                      tools=self.tools,
                      tool_choice="none" if force_text else "auto")
        if self._temperature_ok:
            kwargs["temperature"] = TEMPERATURE
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            # reasoning-tier models reject temperature; drop it once, note it
            if self._temperature_ok and "temperature" in str(e):
                self._temperature_ok = False
                kwargs.pop("temperature", None)
                resp = self.client.chat.completions.create(**kwargs)
            else:
                raise
        u = resp.usage
        self.usage["input_tokens"] += u.prompt_tokens
        self.usage["output_tokens"] += u.completion_tokens
        cached = getattr(getattr(u, "prompt_tokens_details", None),
                         "cached_tokens", 0) or 0
        self.usage["cache_read"] += cached
        self.usage["input_tokens"] -= cached      # keep "uncached input" shape
        msg = resp.choices[0].message
        self.messages.append(msg.model_dump(exclude_none=True))
        if msg.tool_calls:
            calls = [(c.id, c.function.name,
                      json.loads(c.function.arguments or "{}"))
                     for c in msg.tool_calls]
            return Reply("tool", msg.content or "", calls)
        return Reply("text", msg.content or "", [])

    @property
    def notes(self):
        return [] if self._temperature_ok else ["temperature unsupported, ran at provider default"]


class GoogleTransport:
    """Gemini via google-genai; same loop interface as OpenAITransport.
    System prompt -> system_instruction; tools -> function_declarations;
    history kept as Content parts with function_call / function_response."""

    def __init__(self, model: str, system: str, tools: list):
        import os
        from google import genai
        from google.genai import types
        self._types = types
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model
        self.system = system
        self.decls = [types.FunctionDeclaration(
            name=t["name"], description=t["description"],
            parameters_json_schema=t["input_schema"]) for t in tools]
        self.contents = []
        self.usage = {"input_tokens": 0, "output_tokens": 0,
                      "cache_read": 0, "cache_write": 0}

    def add_user(self, text: str):
        t = self._types
        self.contents.append(t.Content(role="user",
                                       parts=[t.Part(text=text)]))

    def add_tool_results(self, results):
        t = self._types
        parts = [t.Part.from_function_response(name=name,
                                               response={"result": output})
                 for name, output in results]
        self.contents.append(t.Content(role="user", parts=parts))

    def reply(self, force_text: bool = False) -> Reply:
        t = self._types
        mode = "NONE" if force_text else "AUTO"
        cfg = t.GenerateContentConfig(
            system_instruction=self.system,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
            tools=[t.Tool(function_declarations=self.decls)],
            tool_config=t.ToolConfig(
                function_calling_config=t.FunctionCallingConfig(mode=mode)),
        )
        resp = self.client.models.generate_content(
            model=self.model, contents=self.contents, config=cfg)
        um = resp.usage_metadata
        cached = getattr(um, "cached_content_token_count", 0) or 0
        self.usage["input_tokens"] += (um.prompt_token_count or 0) - cached
        self.usage["output_tokens"] += (um.candidates_token_count or 0)
        self.usage["cache_read"] += cached
        cand = resp.candidates[0]
        self.contents.append(cand.content)
        calls, text = [], ""
        for part in (cand.content.parts or []):
            if getattr(part, "function_call", None):
                fc = part.function_call
                calls.append((fc.name, fc.name, dict(fc.args or {})))
            elif getattr(part, "text", None):
                text += part.text
        return Reply("tool" if calls else "text", text, calls)

    # Gemini tool-result parts key on name, not an id; the loop passes
    # (id, output) where id == name for this transport.
    @property
    def notes(self):
        return []


def make_transport(model: str, system: str, tools: list):
    prov = provider_of(model)
    if prov == "openai":
        return OpenAITransport(model, system, tools)
    if prov == "google":
        return GoogleTransport(model, system, tools)
    raise ValueError("anthropic models go through bench.agent.run_tool_loop")


def run_tool_loop_multi(model: str, system: str, question: str, tools: list,
                        executors: dict, max_tool_calls: int) -> ArmResult:
    """Same loop semantics as bench.agent.run_tool_loop, provider-neutral."""
    t = make_transport(model, system, tools)
    t.add_user(question)
    transcript, n_calls = [], 0

    reply = t.reply()
    while reply.kind == "tool":
        results = []
        for call_id, name, args in reply.tool_calls:
            n_calls += 1
            over = n_calls > max_tool_calls
            out = ("ERROR: tool budget exhausted — produce your final JSON "
                   "answer now." if over else executors[name](**args))
            transcript.append({"tool": name, "input": args,
                               "output": out[:2000]})
            results.append((call_id, out))
        t.add_tool_results(results)
        if n_calls > max_tool_calls:
            t.add_user("Tool budget exhausted. Respond with the final JSON "
                       "object now.")
            reply = t.reply(force_text=True)
            break
        reply = t.reply()

    raw = reply.text
    answer, parse_error = None, False
    try:
        answer = _extract_json(raw)
    except Exception:
        t.add_user("Return ONLY the JSON object matching the required "
                   "schema. No other text.")
        raw = t.reply(force_text=True).text
        try:
            answer = _extract_json(raw)
        except Exception:
            parse_error = True
    res = ArmResult(answer=answer, raw_text=raw, parse_error=parse_error,
                    usage=t.usage, n_tool_calls=n_calls,
                    transcript=transcript)
    if t.notes:
        res.transcript.append({"notes": t.notes})
    return res


def run_sql_arm_multi(arm: str, question: str, pg_url: str, schema: str,
                      model: str) -> ArmResult:
    """Arms A/B on a non-Anthropic model: frozen prompts, frozen tool, frozen
    loop semantics — only the transport differs."""
    system = {"A": ARM_A_SYSTEM, "B": ARM_B_SYSTEM}[arm]
    run_sql = make_run_sql(pg_url, schema)
    return run_tool_loop_multi(model, system, question, [RUN_SQL_TOOL],
                               {"run_sql": run_sql}, MAX_SQL_CALLS)
