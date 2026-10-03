"""Portfolio — filterable portfolio analytics. Every number re-queries the API with the filters."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.components import filters, ui
from frontend.utils import api

ui.page_header("Monitor", "Portfolio",
               "Slice the booked book by vintage, product, income, region, segment, risk tier and credit score band. "
               "All metrics, charts and the customer list recalculate from the database on every change.")

try:
    filters.filter_bar()
    params = filters.as_params()
    s = api.get("/portfolio/summary", params)
except api.APIError as e:
    ui.api_error(e)
    st.stop()

if not s["customers"]:
    ui.empty_state("No customers match this combination of filters. Remove a filter to widen the selection.")
    st.stop()

scored = s["model_scored"]
k = st.columns(6)
with k[0]:
    ui.kpi("Exposure", ui.fmt_money(s["exposure"]), "current loan amount", "observed")
with k[1]:
    ui.kpi("Default Rate", ui.fmt_pct(s["default_rate"], 2), f"{ui.fmt_int(s['customers_with_outcome'])} with outcome",
           "observed")
with k[2]:
    ui.kpi("Average PD", ui.fmt_pct(s["avg_pd"], 2), "champion model", "model")
with k[3]:
    ui.kpi("Expected Loss", ui.fmt_money(s["expected_loss"]), f"{ui.fmt_pct(s['el_rate'], 2)} of EAD", "model")
with k[4]:
    ui.kpi("Customers", ui.fmt_int(s["customers"]), s["filters"][:40], "observed")
with k[5]:
    ui.kpi("High-Risk %", ui.fmt_pct(s["high_risk_share"]), f"{ui.fmt_int(s['high_risk_customers'])} High / Very High",
           "model")

metric = "avg_pd" if scored else "default_rate"
metric_label = "Average PD" if scored else "Default rate"


def risk_bars(rows, title, x_key="segment"):
    fig = go.Figure()
    xs = [str(r[x_key]) for r in rows]
    fig.add_trace(go.Bar(x=xs, y=[r["default_rate"] for r in rows], name="Observed default rate",
                         marker=dict(color=ui.SERIES[1], cornerradius=4),
                         hovertemplate="%{x}<br>Observed default %{y:.2%}<extra></extra>"))
    if scored:
        fig.add_trace(go.Bar(x=xs, y=[r["avg_pd"] for r in rows], name="Average PD",
                             marker=dict(color=ui.SERIES[0], cornerradius=4),
                             customdata=[[r["customers"], r["exposure"]] for r in rows],
                             hovertemplate="%{x}<br>Average PD %{y:.2%}<br>%{customdata[0]:,} customers"
                                           "<br>Exposure %{customdata[1]:,.0f}<extra></extra>"))
    fig.update_layout(title=title, yaxis_tickformat=".1%", barmode="group", bargroupgap=0.08)
    return fig


try:
    trend = [r for r in api.get("/portfolio/trend", params) if r["vintage"] != "2026-Q1"]
    by_seg = api.get("/portfolio/breakdown/customer_segment", params)
    by_prod = api.get("/portfolio/breakdown/product", params)
except api.APIError as e:
    ui.api_error(e)
    st.stop()

c1, c2 = st.columns(2)
with c1:
    ui.chart(risk_bars(trend, "Risk by Vintage", "vintage"), 300)
with c2:
    ui.chart(risk_bars([r for r in by_seg if r["customers"] >= 30], "Risk by Customer Segment"), 300)

c3, c4 = st.columns(2)
with c3:
    ui.chart(risk_bars(by_prod, "Risk by Product"), 300)
with c4:
    if scored:
        conc = api.get("/portfolio/concentration", params)
        fig = go.Figure()
        x = [f"D{r['decile']}" for r in conc]
        fig.add_trace(go.Bar(x=x, y=[r["exposure_share"] for r in conc], name="Share of exposure",
                             marker=dict(color=ui.MUTED, cornerradius=4),
                             hovertemplate="%{x}<br>%{y:.1%} of exposure<extra></extra>"))
        fig.add_trace(go.Bar(x=x, y=[r["el_share"] for r in conc], name="Share of expected loss",
                             marker=dict(color=ui.SERIES[0], cornerradius=4),
                             customdata=[[r["pd_min"], r["pd_max"], r["default_rate"]] for r in conc],
                             hovertemplate="%{x} · PD %{customdata[0]:.1%}–%{customdata[1]:.1%}<br>%{y:.1%} of expected loss"
                                           "<br>Observed default %{customdata[2]:.1%}<extra></extra>"))
        top = conc[-1]
        fig.update_layout(title=f"Exposure Concentration · riskiest decile holds {ui.fmt_pct(top['el_share'])} of EL",
                          yaxis_tickformat=".0%", barmode="group", xaxis_title="PD decile (D10 = riskiest)")
        ui.chart(fig, 300)
    else:
        ui.empty_state("Exposure concentration by PD decile appears once the model has scored the portfolio.")

if scored:
    dist = api.get("/portfolio/risk-distribution", params)
    hist = dist["pd_histogram"]
    fig = go.Figure(go.Bar(x=[h["label"] for h in hist], y=[h["customers"] for h in hist],
                           marker=dict(color=ui.ACCENT, cornerradius=4),
                           customdata=[h["observed_default_rate"] for h in hist],
                           hovertemplate="PD %{x}<br>%{y:,} customers<br>Observed default %{customdata:.1%}<extra></extra>"))
    fig.update_layout(title="PD Distribution", xaxis_title="Predicted probability of default", yaxis_title="Customers")
    ui.chart(fig, 260)

# ---------------------------------------------------------------- customer table
ui.section("Customers", "Select a row to open the customer in the Investigation Center")
cc = st.columns([2, 2, 4])
with cc[0]:
    sort = st.selectbox("Sort by", ["expected_loss", "pd", "exposure", "ews", "anomaly"],
                        format_func=lambda v: {"expected_loss": "Expected loss", "pd": "PD", "exposure": "Exposure",
                                               "ews": "Early warning score", "anomaly": "Anomaly score"}[v])
with cc[1]:
    search = st.text_input("Find customer ID", placeholder="e.g. C100002")
try:
    res = api.get("/portfolio/customers", {**params, "sort": sort, "limit": 200, "search": search or None})
except api.APIError as e:
    ui.api_error(e)
    st.stop()
df = pd.DataFrame(res["rows"])
st.caption(f"Top {len(df):,} of {res['total']:,} customers in the selection")
if not df.empty:
    show = df[["customer_id", "exposure", "pd", "risk_tier", "expected_loss", "ews_score", "anomaly_score",
               "vintage", "product", "customer_segment", "default_flag"]].copy()
    show["pd"] = show["pd"] * 100
    event = st.dataframe(
        show, hide_index=True, use_container_width=True, height=380, on_select="rerun", selection_mode="single-row",
        column_config={
            "customer_id": "Customer",
            "exposure": st.column_config.NumberColumn("Exposure", format="%,.0f"),
            "pd": st.column_config.ProgressColumn("PD", format="%.1f%%", min_value=0, max_value=100),
            "risk_tier": "Risk tier",
            "expected_loss": st.column_config.NumberColumn("Expected loss", format="%,.0f"),
            "ews_score": st.column_config.NumberColumn("Early warning", format="%d"),
            "anomaly_score": st.column_config.NumberColumn("Anomaly", format="%d"),
            "vintage": "Vintage", "product": "Product", "customer_segment": "Segment",
            "default_flag": st.column_config.NumberColumn("Defaulted", help="Observed outcome (TARGET)"),
        })
    sel = event.selection.rows if event and event.selection else []
    if sel:
        cid = show.iloc[sel[0]]["customer_id"]
        st.session_state["investigate_customer"] = cid
        b1, b2, _ = st.columns([2, 2, 5])
        if b1.button(f"Open {cid} in Investigation Center", type="primary", use_container_width=True):
            st.switch_page("frontend/pages/investigations.py")
        if b2.button(f"Investigate {cid} with AI", use_container_width=True):
            st.session_state["ai_prefill"] = f"Why is customer {cid} high risk?"
            st.switch_page("frontend/pages/ai_analyst.py")
