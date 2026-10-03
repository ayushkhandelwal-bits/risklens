"""Overview — Portfolio Health."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from frontend.components import filters, ui
from frontend.utils import api

ui.page_header("Overview", "Portfolio Health",
               "Booked lending portfolio at a glance — observed outcomes, model risk, early warnings and what is driving them.")

params = filters.as_params()
if filters.current() and any(filters.current().values()):
    st.caption(f"Filters from the Portfolio page apply here: **{filters.describe()}**")

try:
    s = api.get("/portfolio/summary", params)
    trend = api.get("/portfolio/trend", params)
except api.APIError as e:
    ui.api_error(e)
    st.stop()

scored = s.get("model_scored")

# ---------------------------------------------------------------- KPI row
k = st.columns(6)
with k[0]:
    ui.kpi("Portfolio Exposure", ui.fmt_money(s["exposure"]), f"{ui.fmt_int(s['customers'])} customers", "observed")
with k[1]:
    ui.kpi("Default Rate", ui.fmt_pct(s["default_rate"], 2), "repayment difficulty (TARGET)", "observed")
with k[2]:
    ui.kpi("Average PD", ui.fmt_pct(s["avg_pd"], 2) if scored else "—",
           "champion model" if scored else "model not scored yet", "model")
with k[3]:
    ui.kpi("Expected Loss", ui.fmt_money(s["expected_loss"]) if scored else "—",
           f"{ui.fmt_pct(s['el_rate'], 2)} of EAD" if scored else "PD × LGD × EAD", "model")
with k[4]:
    ui.kpi("High-Risk Customers", ui.fmt_int(s["high_risk_customers"]) if scored else "—",
           f"{ui.fmt_pct(s['high_risk_share'])} · {ui.fmt_money(s['high_risk_exposure'])} exposure" if scored else "",
           "model")
with k[5]:
    has_ews = s.get("early_warning_rate") is not None
    ui.kpi("Active Early Warnings", ui.fmt_int(s["active_early_warnings"]) if has_ews else "—",
           f"{ui.fmt_int(s['critical_warnings'])} critical · {ui.fmt_pct(s['early_warning_rate'])} of book" if has_ews else "",
           "derived")

# ---------------------------------------------------------------- AI insight
try:
    ins = api.get("/portfolio/insight", params)
    body = "".join(f"<p>{html.escape(l)}</p>" for l in ins["lines"])
    ev = "".join(f"<div class='rl-evidence'>{html.escape(e)}</div>" for e in ins["evidence"])
    st.markdown(f"<div class='rl-insight' style='margin-top:16px'><div class='rl-insight-head'>◈ Insight · generated from live data</div>"
                f"{body}<div style='margin-top:8px'>{ev}</div></div>", unsafe_allow_html=True)
    with st.expander("How this insight was produced"):
        st.caption(ins["method"])
        if ins.get("segment_contributions"):
            st.dataframe([{**r, "change_pp": round(r["change"] * 100, 2), "recent_share": round(r["recent_share"] * 100, 1)}
                          for r in ins["segment_contributions"]], hide_index=True, use_container_width=True,
                         column_order=["dimension", "segment", "metric", "change_pp", "recent_share", "customers"])
except api.APIError as e:
    ui.api_error(e)

# ---------------------------------------------------------------- charts row 1
c1, c2 = st.columns([1.35, 1])
with c1:
    vint = [r for r in trend if r["vintage"] != "2026-Q1"]
    fig = go.Figure()
    x = [r["vintage"] for r in vint]
    fig.add_trace(go.Scatter(x=x, y=[r["default_rate"] for r in vint], name="Observed default rate",
                             mode="lines+markers", line=dict(width=2, color=ui.SERIES[1]), marker=dict(size=8),
                             hovertemplate="Observed %{y:.2%}<extra></extra>"))
    if scored:
        fig.add_trace(go.Scatter(x=x, y=[r["avg_pd"] for r in vint], name="Average PD (model)",
                                 mode="lines+markers", line=dict(width=2, color=ui.SERIES[0]), marker=dict(size=8),
                                 hovertemplate="Avg PD %{y:.2%}<extra></extra>"))
    fig.update_layout(title="Portfolio Risk Trend by Vintage", yaxis_tickformat=".1%", hovermode="x unified")
    ui.chart(fig, 320)
    st.caption("Vintages are simulated booking quarters (Home Credit has no calendar dates) — see methodology.")
with c2:
    if scored:
        dist = api.get("/portfolio/risk-distribution", params)
        tiers = dist["tiers"]
        fig = go.Figure(go.Bar(
            y=[t["risk_tier"] for t in tiers], x=[t["exposure"] for t in tiers], orientation="h",
            marker=dict(color=[ui.TIER_COLOR[t["risk_tier"]] for t in tiers], cornerradius=4),
            text=[f"{ui.fmt_money(t['exposure'])} · EL {ui.fmt_money(t['expected_loss'])}" for t in tiers],
            textposition="outside", cliponaxis=False,
            customdata=[[t["customers"], t["avg_pd"], t["default_rate"]] for t in tiers],
            hovertemplate="%{y}<br>Exposure %{x:,.0f}<br>Customers %{customdata[0]:,}<br>Avg PD %{customdata[1]:.2%}"
                          "<br>Observed default %{customdata[2]:.2%}<extra></extra>"))
        fig.update_layout(title="Exposure by Risk Tier", xaxis=dict(showticklabels=False, showgrid=False),
                          yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"), margin=dict(r=120))
        ui.chart(fig, 320)
    else:
        ui.empty_state("Exposure by risk tier appears once the risk model has scored the portfolio.")

# ---------------------------------------------------------------- charts row 2
c3, c4 = st.columns([1, 1.35])
with c3:
    if scored:
        hist = dist["pd_histogram"]
        fig = go.Figure(go.Bar(x=[h["label"] for h in hist], y=[h["customers"] for h in hist],
                               marker=dict(color=ui.ACCENT, cornerradius=4),
                               customdata=[h["observed_default_rate"] for h in hist],
                               hovertemplate="PD %{x}<br>%{y:,} customers<br>Observed default %{customdata:.1%}<extra></extra>"))
        fig.update_layout(title="Risk Distribution (PD)", xaxis_title="Predicted probability of default",
                          yaxis_title="Customers")
        ui.chart(fig, 300)
    else:
        ui.empty_state("PD distribution appears once the risk model has scored the portfolio.")
with c4:
    beh = api.get("/portfolio/behaviour-trend", params)
    if beh:
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Card utilisation", "Late-payment rate", "POS delinquency rate"),
                            horizontal_spacing=0.09)
        m = [r["month"] for r in beh]
        for i, (key, col) in enumerate((("utilisation", ui.SERIES[0]), ("late_rate", ui.SERIES[1]),
                                        ("pos_dpd_rate", ui.SERIES[2])), start=1):
            fig.add_trace(go.Scatter(x=m, y=[r[key] for r in beh], mode="lines", line=dict(width=2, color=col),
                                     showlegend=False, hovertemplate="Month %{x}<br>%{y:.2%}<extra></extra>"), 1, i)
            top = max((r[key] or 0) for r in beh)
            fig.update_yaxes(tickformat=".1%" if top < 0.05 else ".0%", row=1, col=i, gridcolor=ui.GRID)
            fig.update_xaxes(title_text="months before application", row=1, col=i, title_font=dict(size=10))
        fig.update_annotations(font=dict(size=12, color=ui.INK2))
        fig.update_layout(title="Early Warning Trends · behaviour in the 24 months before application",
                          margin=dict(t=70))
        ui.chart(fig, 300)

# ---------------------------------------------------------------- drivers
if scored:
    drivers = api.get("/portfolio/top-drivers", {**params, "limit": 8})
    if drivers:
        ui.section("Top Risk Drivers", "Features most often pushing PD up across the selection (aggregated SHAP, champion model)")
        d = list(reversed(drivers))
        fig = go.Figure(go.Bar(y=[r["label"] for r in d], x=[r["customers_affected"] for r in d], orientation="h",
                               marker=dict(color=ui.ACCENT, cornerradius=4),
                               customdata=[r["avg_contribution"] for r in d],
                               hovertemplate="%{y}<br>Top driver for %{x:,} customers<br>Avg SHAP +%{customdata:.3f} log-odds<extra></extra>"))
        fig.update_layout(xaxis_title="Customers for whom this is a top-5 risk-increasing driver", margin=dict(t=10))
        ui.chart(fig, 300)

# ---------------------------------------------------------------- data quality strip
try:
    dq = api.get("/portfolio/data-quality")
    ui.section("Data Quality", "Checks run before analytics on every ETL load")
    q = st.columns([1, 1, 1, 1, 2])
    with q[0]:
        st.markdown(ui.pill(dq["status"]), unsafe_allow_html=True)
        st.caption(f"Last load {dq['finished_at'][:16]} UTC")
    q[1].metric("Records loaded", ui.fmt_int(dq["total_records"]))
    q[2].metric("Checks passed", f"{dq['pass']}/{dq['pass'] + dq['warn'] + dq['fail']}")
    q[3].metric("Warnings / failures", f"{dq['warn']} / {dq['fail']}")
    with q[4]:
        cov = dq["coverage"]
        st.caption(f"Coverage — external score {ui.fmt_pct(cov['score_coverage'])} · payment history "
                   f"{ui.fmt_pct(cov['payment_history_coverage'])} · bureau {ui.fmt_pct(cov['bureau_coverage'])} · "
                   f"credit card {ui.fmt_pct(cov['card_coverage'])}")
except api.APIError as e:
    ui.api_error(e)
