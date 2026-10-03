"""AI Risk Analyst — chat interface over controlled RiskLens tools."""
from __future__ import annotations

import json

import streamlit as st

from frontend.components import filters, ui
from frontend.utils import api

ui.page_header("Investigate", "AI Risk Analyst",
               "Ask risk questions in plain language. The analyst plans its own investigation, calls controlled "
               "RiskLens tools on live data, and answers with evidence — decision support, not decisions.")

try:
    status = api.get("/ai/status", cache=False)
except api.APIError as e:
    ui.api_error(e)
    st.stop()

mode_txt = (f"LLM agent · {status['provider']} · {status['model']}" if status["llm_configured"]
            else "Offline analyst mode · deterministic tool planner (set LLM_API_KEY for the LLM agent)")
st.markdown(f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px'>"
            f"{ui.pill(mode_txt, 'good' if status['llm_configured'] else 'neutral')}"
            f"{ui.pill('Context: ' + filters.describe(), 'neutral')}"
            f"{ui.pill(str(len(status['tools'])) + ' tools · every call audited', 'neutral')}</div>",
            unsafe_allow_html=True)

st.session_state.setdefault("ai_messages", [])
st.session_state.setdefault("ai_session", None)

SUGGESTED = ["Why has portfolio risk increased?", "Which customer segments are becoming risky?",
             "Why is customer C426709 high risk?", "Which accounts require investigation?",
             "Are there signs of behavioural anomalies?", "Is the model showing feature drift?"]


def render_trace(trace: list[dict]) -> None:
    if not trace:
        return
    total = sum(t["duration_ms"] for t in trace)
    with st.expander(f"Tool execution · {len(trace)} step(s) · {total / 1000:.1f}s", expanded=False):
        for i, t in enumerate(trace, 1):
            icon = "✓" if t["status"] == "ok" else "✕"
            st.markdown(f"**{i}. {t['progress']}** {icon} &nbsp; <code>{t['tool']}</code> "
                        f"<span style='color:#898781'>· {t['duration_ms']} ms</span>", unsafe_allow_html=True)
            st.caption(f"args: {json.dumps(t['args'])} → {t['summary']}")
        st.caption("Raw tool results are recorded in the audit log (Data & Audit page).")


def ask(question: str) -> None:
    st.session_state.ai_messages.append({"role": "user", "content": question})
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.ai_messages[:-1]][-6:]
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        box = st.status("Investigating…", expanded=True)
        box.write("Planning which tools to use for this question…")
        try:
            res = api.post("/ai/query", {"question": question, "filters": filters.current(),
                                         "session_id": st.session_state.ai_session, "history": history})
        except api.APIError as e:
            box.update(label="Investigation failed", state="error")
            st.error(str(e))
            return
        for t in res["tool_calls"]:
            box.write(f"{'✓' if t['status'] == 'ok' else '✕'} {t['progress']}")
        box.update(label=f"Investigation complete · {len(res['tool_calls'])} tool call(s) · "
                         f"{res['elapsed_ms'] / 1000:.1f}s", state="complete", expanded=False)
        st.session_state.ai_session = res["session_id"]
        st.markdown(res["answer"])
        if res.get("notice"):
            st.caption(res["notice"])
        render_trace(res["tool_calls"])
    st.session_state.ai_messages.append({"role": "assistant", "content": res["answer"], "trace": res["tool_calls"],
                                         "notice": res.get("notice")})


# ---------------------------------------------------------------- history
for m in st.session_state.ai_messages:
    with st.chat_message(m["role"], avatar=":material/smart_toy:" if m["role"] == "assistant" else None):
        st.markdown(m["content"])
        if m.get("notice"):
            st.caption(m["notice"])
        render_trace(m.get("trace", []))

if not st.session_state.ai_messages:
    st.markdown("<div class='rl-section'>Suggested questions</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, q in enumerate(SUGGESTED):
        if cols[i % 3].button(q, key=f"sugg_{i}", use_container_width=True):
            st.session_state["ai_prefill"] = q
            st.session_state["ai_autorun"] = True
            st.rerun()

prefill = st.session_state.pop("ai_prefill", None)
autorun = st.session_state.pop("ai_autorun", False)
typed = st.chat_input("Ask about a customer, a segment, the portfolio, early warnings or the model…")
if typed:
    ask(typed)
elif prefill and autorun:
    ask(prefill)
elif prefill:
    st.info(f"Suggested from your previous page: **{prefill}** — press the button to run it.")
    if st.button(f"Ask: {prefill}", type="primary"):
        ask(prefill)

if st.session_state.ai_messages:
    if st.button("New conversation"):
        st.session_state.ai_messages, st.session_state.ai_session = [], None
        st.rerun()
