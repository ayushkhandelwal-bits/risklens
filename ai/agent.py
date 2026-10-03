"""
AI Risk Analyst — tool-using agent.

  question + dashboard filters
        │
        ▼
  LLM (Claude / OpenAI) ──decides──▶ tool calls ──▶ RiskLens tools (DB, model, SHAP, EWS, SQL guard)
        ▲                                   │
        └───────────── tool results ◀───────┘           every step written to audit_log
        │
        ▼
  answer (decision support) + tool trace

Without an API key (or if the LLM fails) the deterministic planner in
ai/planner.py runs the same tools and assembles a templated answer.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ai import llm
from ai.planner import run as planner_run
from ai.tools import REGISTRY
from ai.tools.context import set_filters
from backend.services.audit_service import log_event

SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system.md").read_text(encoding="utf-8")
MAX_STEPS = 10
MAX_RESULT_CHARS = 14000


@dataclass
class ToolTrace:
    tool: str
    args: dict
    status: str
    duration_ms: int
    progress: str
    summary: str
    result: object = None


@dataclass
class AgentResult:
    answer: str
    mode: str
    model: str | None
    session_id: str
    tool_calls: list[ToolTrace] = field(default_factory=list)
    notice: str | None = None
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        for t in d["tool_calls"]:
            t["result"] = _truncate(t["result"], 3000)
        return d


def _truncate(obj, limit: int):
    s = json.dumps(obj, default=str)
    return obj if len(s) <= limit else s[:limit] + "…(truncated)"


def _summarise(name: str, result) -> str:
    if isinstance(result, dict):
        if "error" in result:
            return f"error: {result['error']}"
        keys = [k for k in ("customer_id", "pd", "risk_tier", "ews_score", "anomaly_status", "row_count",
                            "metric_before", "metric_after", "overall_status", "total_alerts", "selection") if k in result]
        if keys:
            return ", ".join(f"{k}={result[k]}" for k in keys)[:300]
        return f"{len(result)} fields"
    return str(result)[:200]


TOOL_TRANSPORT = os.getenv("RISKLENS_TOOL_TRANSPORT", "direct").lower()   # direct | mcp


def _call_via_mcp(name: str, args: dict):
    """Invoke a tool through the RiskLens MCP server (MCP protocol, in-memory transport)."""
    import anyio
    from mcp.client.client import Client
    from mcp_server.server import mcp as server

    async def go():
        async with Client(server) as c:
            r = await c.call_tool(name, args)
            return json.loads(r.content[0].text) if r.content else {"error": "empty MCP response"}
    return anyio.run(go)


class ToolExecutor:
    """Executes registered tools with timing, error capture and audit logging.
    transport='mcp' routes every call through the MCP server instead of a direct Python call."""

    def __init__(self, session_id: str, user: str, transport: str | None = None):
        self.session_id, self.user = session_id, user
        self.transport = transport or TOOL_TRANSPORT
        self.trace: list[ToolTrace] = []

    def __call__(self, name: str, **args):
        tool = REGISTRY.get(name)
        t0 = time.time()
        if tool is None:
            result, status = {"error": f"Unknown tool '{name}'"}, "error"
        else:
            try:
                result = _call_via_mcp(name, args) if self.transport == "mcp" else tool.func(**args)
                status = "error" if isinstance(result, dict) and "error" in result else "ok"
            except Exception as exc:   # surfaced to the model / planner, never crashes the agent
                result, status = {"error": f"{type(exc).__name__}: {exc}"}, "error"
        dur = int((time.time() - t0) * 1000)
        summary = _summarise(name, result)
        self.trace.append(ToolTrace(name, args, status, dur, tool.progress if tool else name, summary, result))
        log_event("tool_call", tool=name if self.transport != "mcp" else f"agent→mcp:{name}", customer_id=args.get("customer_id"), parameters=args,
                  result_summary=summary, user=self.user, session_id=self.session_id, status=status)
        return result


def investigate(question: str, filters: dict | None = None, session_id: str | None = None,
                history: list[dict] | None = None, user: str = "analyst", force_offline: bool = False) -> AgentResult:
    session_id = session_id or uuid.uuid4().hex[:12]
    t0 = time.time()
    set_filters(filters)
    log_event("ai_query", parameters={"question": question, "filters": filters or {}}, user=user,
              session_id=session_id, result_summary=question[:500])
    execu = ToolExecutor(session_id, user)
    tools = list(REGISTRY.values())
    answer, mode, model, notice = None, "llm", None, None

    if force_offline or not llm.llm_configured():
        mode, notice = "planner", "Offline analyst mode — no LLM configured; tools ran on live data and the answer was assembled from their outputs."
    else:
        try:
            ctx = f"Dashboard selection (use as default context): {filters or 'entire booked portfolio'}"
            client = llm.make_client(SYSTEM_PROMPT + "\n\n" + ctx, tools)
            model = client.model
            for h in (history or [])[-6:]:
                (client.add_user if h["role"] == "user" else client.add_assistant_text)(h["content"])
            client.add_user(question)
            for _ in range(MAX_STEPS):
                turn = client.step()
                if not turn.tool_calls:
                    answer = turn.text.strip()
                    break
                results = []
                for call in turn.tool_calls:
                    res = execu(call.name, **call.args)
                    out = json.dumps(res, default=str)
                    if len(out) > MAX_RESULT_CHARS:
                        out = out[:MAX_RESULT_CHARS] + "…(truncated)"
                    results.append((call, out, isinstance(res, dict) and "error" in res))
                client.add_tool_results(results)
            if not answer:
                answer = turn.text.strip() or "I reached the tool-call limit before finishing; please narrow the question."
        except Exception as exc:
            mode = "planner"
            notice = f"The LLM service was unavailable ({type(exc).__name__}); fell back to the offline analyst planner."
            log_event("ai_error", parameters={"error": str(exc)[:500]}, user=user, session_id=session_id, status="error")

    if mode == "planner":
        try:
            answer = planner_run(question, execu)
        except Exception as exc:
            answer = f"I could not complete this investigation: {type(exc).__name__}: {exc}"

    if execu.transport == "mcp":
        mode += "+mcp"
    res = AgentResult(answer=answer, mode=mode, model=model, session_id=session_id, tool_calls=execu.trace,
                      notice=notice, elapsed_ms=int((time.time() - t0) * 1000))
    log_event("ai_response", parameters={"mode": mode, "model": model, "tools": [t.tool for t in execu.trace]},
              result_summary=answer[:4000], user=user, session_id=session_id)
    return res
