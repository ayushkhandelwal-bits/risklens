"""AI Risk Analyst tests: SQL guard, tools, planner and the LLM tool-calling loop (with mocked providers)."""
import copy
import json

import pytest

from ai.planner import detect_intent
from ai.tools.sql_tool import UnsafeQuery, validate
from tests.conftest import requires_db

UNSAFE = [
    "DROP TABLE customer_360",
    "DELETE FROM risk_scores",
    "UPDATE risk_scores SET pd = 0",
    "INSERT INTO audit_log(action) VALUES ('x')",
    "ALTER TABLE risk_scores ADD COLUMN x int",
    "TRUNCATE risk_scores",
    "select 1; drop table risk_scores",
    "WITH d AS (DELETE FROM risk_scores RETURNING *) SELECT * FROM d",
    "select * into tmp_copy from risk_scores",
    "select * from audit_log",
    "select * from pg_catalog.pg_roles",
    "select * from information_schema.tables",
    "select pg_sleep(30)",
    "select * from raw_application",
    "COPY risk_scores TO '/tmp/x'",
    "",
]


@pytest.mark.parametrize("sql", UNSAFE)
def test_unsafe_sql_is_rejected(sql):
    with pytest.raises(UnsafeQuery):
        validate(sql)


def test_safe_sql_is_accepted():
    assert validate("SELECT risk_tier, COUNT(*) FROM v_portfolio GROUP BY 1;")
    assert validate("with t as (select * from risk_scores) select count(*) from t")
    assert validate("select 'drop table' as label from v_portfolio")      # keyword inside a string literal


def test_intent_routing():
    assert detect_intent("Why is customer C10291 high risk?") == ("customer", "C10291") or \
        detect_intent("Why is customer C102910 high risk?")[0] == "customer"
    assert detect_intent("Why is customer C100002 high risk?") == ("customer", "C100002")
    assert detect_intent("Why has portfolio risk increased?")[0] == "rca"
    assert detect_intent("Which accounts require investigation?")[0] == "priority"
    assert detect_intent("Are there signs of behavioural anomalies?")[0] == "anomaly"
    assert detect_intent("Is the model showing feature drift?")[0] == "drift"
    assert detect_intent("Which customer segments are becoming risky?")[0] == "segments"


@requires_db
def test_safe_sql_executes_read_only():
    from ai.tools.sql_tool import execute_safe
    r = execute_safe("select risk_tier, count(*) n from v_portfolio where population='portfolio' group by 1")
    assert r["row_count"] == 4 and set(r["columns"]) == {"risk_tier", "n"}
    big = execute_safe("select customer_id from v_portfolio")
    assert big["row_count"] == 200 and big["truncated"]


@requires_db
def test_tools_return_live_data_and_handle_errors():
    from ai.agent import ToolExecutor
    ex = ToolExecutor("test-session", "pytest")
    risk = ex("calculate_customer_risk", customer_id="C100002")
    assert risk["customer_id"] == "C100002" and risk["matches_stored_score"]
    bad = ex("get_customer_profile", customer_id="C1")
    assert "error" in bad and "not found" in bad["error"]
    unsafe = ex("run_safe_sql_query", sql="delete from risk_scores")
    assert "error" in unsafe
    assert ex("no_such_tool")["error"].startswith("Unknown tool")
    assert [t.status for t in ex.trace] == ["ok", "error", "error", "error"]


@requires_db
def test_filter_context_reaches_tools():
    from ai.tools.context import set_filters
    from ai.tools.risk_tool import get_portfolio_metrics
    set_filters({"product": ["Revolving loans"]})
    rev = get_portfolio_metrics()
    set_filters({})
    allp = get_portfolio_metrics()
    assert rev["summary"]["customers"] < allp["summary"]["customers"]
    assert "Revolving" in rev["selection"]


@requires_db
def test_planner_answers_are_grounded():
    from ai.agent import investigate
    r = investigate("Why is customer C100002 high risk?", force_offline=True)
    assert r.mode == "planner" and len(r.tool_calls) == 5
    pd_value = next(t.result for t in r.tool_calls if t.tool == "calculate_customer_risk")["pd"]
    assert f"{pd_value * 100:.1f}%" in r.answer               # number in the answer comes from the tool
    assert "decision support" in r.answer.lower()
    q = investigate("Which accounts require investigation?", force_offline=True)
    top = next(t.result for t in q.tool_calls if t.tool == "get_investigation_priorities")["customers"][0]
    assert top["customer_id"] in q.answer


# --------------------------------------------------------------------------- mocked LLM providers
class _FakeAnthropicMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        from anthropic.types import Message
        self.calls.append(copy.deepcopy(kw))
        if len(self.calls) == 1:
            content = [{"type": "text", "text": "Let me look this customer up."},
                       {"type": "tool_use", "id": "toolu_1", "name": "calculate_customer_risk",
                        "input": {"customer_id": "C100002"}},
                       {"type": "tool_use", "id": "toolu_2", "name": "explain_prediction",
                        "input": {"customer_id": "C100002", "top_n": 3}}]
            stop = "tool_use"
        else:
            results = kw["messages"][-1]["content"]
            pd_ = json.loads(results[0]["content"])["pd"]
            content = [{"type": "text", "text": f"**Risk summary** PD is {pd_:.4f}."}]
            stop = "end_turn"
        return Message.model_validate({"id": f"msg_{len(self.calls)}", "type": "message", "role": "assistant",
                                       "model": kw["model"], "content": content, "stop_reason": stop,
                                       "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}})


@requires_db
def test_llm_agent_loop_anthropic(monkeypatch):
    import anthropic
    from ai import agent, llm
    fake = _FakeAnthropicMessages()

    class FakeClient:
        def __init__(self, **kw):
            self.messages = fake
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    monkeypatch.setattr(llm, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(llm, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(llm, "llm_configured", lambda: True)
    monkeypatch.setattr(llm, "make_client", lambda system, tools: llm.AnthropicClient(system, tools, api_key="k"))
    import uuid
    sid = "pytest-" + uuid.uuid4().hex[:8]
    r = agent.investigate("Why is customer C100002 high risk?", session_id=sid)
    assert r.mode == "llm" and r.model == "claude-sonnet-5-5"
    assert [t.tool for t in r.tool_calls] == ["calculate_customer_risk", "explain_prediction"]
    # tool results were sent back in Anthropic's tool_result format, matched to the tool_use ids
    second = fake.calls[1]["messages"][-1]["content"]
    assert [b["tool_use_id"] for b in second] == ["toolu_1", "toolu_2"] and second[0]["type"] == "tool_result"
    assert {t["name"] for t in fake.calls[0]["tools"]} >= {"run_safe_sql_query", "generate_root_cause_analysis"}
    assert "PD is 0." in r.answer
    from backend.services.audit_service import recent
    actions = [a["action"] for a in recent(20, session_id=sid)]
    assert actions.count("tool_call") == 2 and "ai_query" in actions and "ai_response" in actions


@requires_db
def test_llm_agent_loop_openai(monkeypatch):
    import openai
    from openai.types.chat import ChatCompletion
    from ai import agent, llm
    calls = []

    class FakeCompletions:
        def create(self, **kw):
            calls.append(copy.deepcopy(kw))
            if len(calls) == 1:
                msg = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get_portfolio_metrics", "arguments": json.dumps({"group_by": "risk_tier"})}}]}
                fr = "tool_calls"
            else:
                n = json.loads(kw["messages"][-1]["content"])["summary"]["customers"]
                msg, fr = {"role": "assistant", "content": f"The selection has {n} customers."}, "stop"
            return ChatCompletion.model_validate({"id": "x", "object": "chat.completion", "created": 0, "model": kw["model"],
                                                 "choices": [{"index": 0, "message": msg, "finish_reason": fr}]})

    class FakeOpenAI:
        def __init__(self, **kw):
            self.chat = type("C", (), {"completions": FakeCompletions()})()
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(llm, "llm_configured", lambda: True)
    monkeypatch.setattr(llm, "make_client", lambda system, tools: llm.OpenAIClient(system, tools, api_key="k"))
    r = agent.investigate("How big is the portfolio?")
    assert r.mode == "llm" and r.tool_calls[0].tool == "get_portfolio_metrics"
    assert calls[1]["messages"][-1]["role"] == "tool" and calls[1]["messages"][-1]["tool_call_id"] == "call_1"
    assert "50000 customers" in r.answer


@requires_db
def test_llm_failure_falls_back_to_planner(monkeypatch):
    from ai import agent, llm

    def boom(system, tools):
        raise ConnectionError("service down")
    monkeypatch.setattr(llm, "llm_configured", lambda: True)
    monkeypatch.setattr(llm, "make_client", boom)
    r = agent.investigate("Which accounts require investigation?")
    assert r.mode == "planner" and "unavailable" in r.notice and "Customers to investigate first" in r.answer


@requires_db
def test_ai_api_endpoints():
    from fastapi.testclient import TestClient
    from backend.main import app
    c = TestClient(app)
    s = c.get("/ai/status").json()
    assert len(s["tools"]) == 10
    r = c.post("/ai/query", json={"question": "Are there signs of behavioural anomalies?"}).json()
    assert r["tool_calls"][0]["tool"] == "get_anomaly_signals" and "fraud determination" in r["answer"]
    assert c.post("/ai/investigate", json={"customer_id": "C1"}).status_code == 404
    assert c.post("/ai/query", json={"question": ""}).status_code == 422
