"""Early Warnings — behavioural alerts with the triggers that fired them, plus behavioural anomalies."""
from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.components import filters, ui
from frontend.utils import api

ui.page_header("Monitor", "Early Warnings",
               "Customers whose recent behaviour signals rising credit risk — before it shows up as default. "
               "Every alert lists the exact triggers and evidence behind it.")

params = filters.as_params()
if any(filters.current().values()):
    st.caption(f"Portfolio filters apply: **{filters.describe()}**")
try:
    s = api.get("/early-warnings/summary", params)
    val = api.get("/early-warnings/validation")
except api.APIError as e:
    ui.api_error(e)
    st.stop()

k = st.columns(5)
with k[0]:
    ui.kpi("Total Alerts", ui.fmt_int(s["total_alerts"]), f"{ui.fmt_pct(s['total_alerts'] / max(s['customers'], 1))} of customers", "derived")
with k[1]:
    ui.kpi("Critical", ui.fmt_int(s["critical"]), "score 81–100", "derived")
with k[2]:
    ui.kpi("High", ui.fmt_int(s["high"]), "score 61–80", "derived")
with k[3]:
    ui.kpi("Medium", ui.fmt_int(s["medium"]), "score 31–60", "derived")
with k[4]:
    ui.kpi("Exposure under alert", ui.fmt_money(s["alert_exposure"]), "current loan amount", "observed")

c1, c2 = st.columns([1.4, 1])
with c1:
    trig = pd.DataFrame(s["triggers"])
    if not trig.empty:
        piv = trig.pivot_table(index="description", columns="severity", values="customers", aggfunc="sum").fillna(0)
        piv = piv.loc[piv.sum(axis=1).sort_values().index]
        fig = go.Figure()
        for sev in ("MEDIUM", "HIGH", "CRITICAL"):
            if sev in piv:
                fig.add_trace(go.Bar(y=piv.index, x=piv[sev], name=sev.title(), orientation="h",
                                     marker=dict(color=ui.TIER_COLOR[sev], line=dict(color="#ffffff", width=2)),
                                     hovertemplate="%{y}<br>" + sev.title() + ": %{x:,} customers<extra></extra>"))
        fig.update_layout(title="Alerts by trigger and severity", barmode="stack", xaxis_title="Customers")
        ui.chart(fig, 380)
with c2:
    v = [r for r in val if r["signal"] == "ews"]
    order = ["Low", "Medium", "High", "Critical"]
    v = sorted(v, key=lambda r: order.index(r["band"]))
    fig = go.Figure(go.Bar(x=[r["band"] for r in v], y=[r["default_rate"] for r in v],
                           marker=dict(color=[ui.TIER_COLOR[r["band"]] for r in v], cornerradius=4),
                           text=[ui.fmt_pct(r["default_rate"]) for r in v], textposition="outside",
                           customdata=[r["n"] for r in v],
                           hovertemplate="%{x}<br>Observed default %{y:.1%}<br>%{customdata:,} customers<extra></extra>"))
    fig.update_layout(title="Does the score work? Observed default rate by band", yaxis_tickformat=".0%",
                      xaxis_title="Early Warning band (booked portfolio)")
    ui.chart(fig, 380)
    st.caption("The EWS is rule-based (not trained on outcomes); rising observed default by band shows the triggers carry real risk signal.")

# ---------------------------------------------------------------- alerts table + detail
ui.section("Alert queue", "Select a customer to see why the alert fired")
f1, f2, _ = st.columns([2, 3, 3])
with f1:
    bands = st.multiselect("Band", ["Critical", "High", "Medium"], default=["Critical", "High"])
with f2:
    codes = {r["description"]: r["trigger_code"] for r in s["triggers"]}
    trig_sel = st.selectbox("Trigger", ["Any trigger"] + sorted(codes))
try:
    rows = api.get("/early-warnings/alerts", {**params, "band": bands or ["Critical", "High", "Medium"],
                                               "trigger": codes.get(trig_sel), "limit": 300})
except api.APIError as e:
    ui.api_error(e)
    st.stop()
df = pd.DataFrame(rows)
if df.empty:
    ui.empty_state("No alerts match this selection.")
    st.stop()
df["pd"] = df["pd"] * 100
event = st.dataframe(
    df[["customer_id", "ews_score", "ews_band", "risk_tier", "top_trigger", "n_triggers", "exposure", "pd", "anomaly_status"]],
    hide_index=True, use_container_width=True, height=340, on_select="rerun", selection_mode="single-row",
    column_config={"customer_id": "Customer", "ews_score": st.column_config.ProgressColumn("Warning score", min_value=0, max_value=100, format="%d"),
                   "ews_band": "Band", "risk_tier": "Risk tier", "top_trigger": "Primary trigger",
                   "n_triggers": "Triggers", "exposure": st.column_config.NumberColumn("Exposure", format="%,.0f"),
                   "pd": st.column_config.NumberColumn("PD", format="%.1f%%"), "anomaly_status": "Anomaly"})
sel = event.selection.rows if event and event.selection else []
cid = df.iloc[sel[0]]["customer_id"] if sel else df.iloc[0]["customer_id"]

w = api.get(f"/customers/{cid}/warnings")
r = api.get(f"/customers/{cid}/risk")
ui.section(f"Why {cid} triggered an alert",
           f"Early Warning Score {w['ews_score']} ({w['ews_band']}) · PD {ui.fmt_pct(r['pd'])} ({r['risk_tier']}) · "
           f"exposure {ui.fmt_money(r['current_exposure'])}" + ("" if sel else " · showing the top alert — select a row to change"))
rows_html = "".join(
    f"<div class='rl-trigger'><div style='min-width:96px'>{ui.pill(t['severity'].title(), ui.TIER_STATUS[t['severity']])}</div>"
    f"<div><div class='rl-trigger-text'>{html.escape(t['description'])}</div>"
    f"<div class='rl-trigger-evidence'>{html.escape(t['evidence'])}</div></div></div>" for t in w["triggers"])
st.markdown(f"<div class='rl-card'>{rows_html}</div>", unsafe_allow_html=True)
b1, b2, _ = st.columns([2, 2, 4])
if b1.button(f"Open {cid} in Investigation Center", type="primary", use_container_width=True):
    st.session_state["investigate_customer"] = cid
    st.switch_page("frontend/pages/investigations.py")
if b2.button("Investigate with AI", use_container_width=True):
    st.session_state["ai_prefill"] = f"Why is customer {cid} high risk?"
    st.switch_page("frontend/pages/ai_analyst.py")

# ---------------------------------------------------------------- behavioural anomalies
ui.section("Behavioural anomalies", "Unusual behaviour for analyst review — Isolation Forest + rules. "
                                     "Not a confirmed-fraud determination: the source data has no fraud labels.")
an = api.get("/anomalies/summary", params)
a1, a2 = st.columns([1, 1.6])
with a1:
    stt = {r["anomaly_status"]: r for r in an["statuses"]}
    for name in ("Suspicious", "Watch", "Normal"):
        r0 = stt.get(name, {"customers": 0, "exposure": 0})
        st.markdown(f"<div class='rl-card' style='margin-bottom:8px;display:flex;justify-content:space-between'>"
                    f"<span>{ui.pill(name)}</span><span style='font-weight:650'>{ui.fmt_int(r0['customers'])} customers · "
                    f"{ui.fmt_money(r0['exposure'])}</span></div>", unsafe_allow_html=True)
    av = sorted([r for r in val if r["signal"] == "anomaly"], key=lambda r: ["Normal", "Watch", "Suspicious"].index(r["band"]))
    st.caption("Observed default rate — " + " · ".join(f"{r['band']} {ui.fmt_pct(r['default_rate'])}" for r in av)
               + ". Anomaly ≠ credit risk: it flags unusual behaviour, which is why it is reviewed separately.")
with a2:
    rr = list(reversed(an["rules"]))
    fig = go.Figure(go.Bar(y=[r["description"] for r in rr], x=[r["customers"] for r in rr], orientation="h",
                           marker=dict(color=ui.SERIES[6], cornerradius=4),
                           hovertemplate="%{y}<br>%{x:,} customers<extra></extra>"))
    fig.update_layout(title="Anomaly rule triggers", xaxis_title="Customers")
    ui.chart(fig, 320)
