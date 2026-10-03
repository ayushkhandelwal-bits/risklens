"""Model Monitor — performance, population stability (PSI), segment drift, explain-with-AI."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.components import ui
from frontend.utils import api

ui.page_header("Govern", "Model Monitor",
               "Is the champion model still fit for purpose? Performance on held-out data, and whether the population "
               "it now scores has drifted from the one it was trained on.")

try:
    d = api.get("/model/drift", {"top_n": 60, "include_bins": True})
except api.APIError as e:
    ui.api_error(e)
    st.stop()
if d.get("status") == "not_run":
    ui.empty_state("No monitoring run yet. Run <code>python -m ml.monitoring</code> or use the button below.")
    if st.button("Run monitoring now"):
        api.post("/model/monitoring/run", {})
        st.rerun()
    st.stop()

perf = d["performance"]
st.markdown(f"<div style='display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap'>"
            f"{ui.pill('Overall: ' + d['overall_status'], ui.TIER_STATUS[d['overall_status']])}"
            f"{ui.pill(d['model'], 'neutral')}{ui.pill('Run ' + d['run_at'][:16] + ' UTC', 'neutral')}"
            f"<span style='font-size:12.5px;color:#52514e'>{d['comparison']} · {d['thresholds']['rule']}</span></div>",
            unsafe_allow_html=True)

k = st.columns(6)
with k[0]:
    ui.kpi("AUC", f"{perf['auc']:.3f}", "held-out test split", "model")
with k[1]:
    ui.kpi("Gini", f"{perf['gini']:.3f}", "2·AUC − 1", "model")
with k[2]:
    ui.kpi("KS", f"{perf['ks']:.3f}", "max TPR − FPR", "model")
with k[3]:
    ui.kpi("Brier", f"{perf['brier']:.4f}", "lower is better", "model")
with k[4]:
    ui.kpi("Calibration", f"{perf['calibration_ratio']:.2f}", f"max decile gap {perf['max_calibration_gap'] * 100:.1f} pts", "model")
with k[5]:
    sp = d["score_psi"]
    ui.kpi("Score PSI", f"{sp['value']:.3f}", f"{sp['status']} · mean PD {sp['baseline_mean_pd']:.1%} → {sp['current_mean_pd']:.1%}",
           "derived")

# ---------------------------------------------------------------- AI explanation
b1, b2, _ = st.columns([2, 2, 5])
with b1:
    explain = st.button("✦ Explain Model Drift", type="primary", use_container_width=True)
with b2:
    if st.button("Re-run monitoring", use_container_width=True):
        with st.spinner("Recomputing PSI and performance…"):
            api.post("/model/monitoring/run", {})
        st.cache_data.clear()
        st.rerun()
if explain:
    with st.spinner("AI Risk Analyst is analysing drift with the analyze_model_drift tool…"):
        try:
            res = api.post("/ai/query", {"question": "Explain the model drift. Is the model showing feature drift and "
                                                     "what should the model-risk team do?"})
            st.session_state["drift_ai"] = res
        except api.APIError as e:
            ui.api_error(e)
if st.session_state.get("drift_ai"):
    res = st.session_state["drift_ai"]
    with st.container(border=True):
        st.markdown(res["answer"])
        st.caption(f"{res['mode']} · tools: {', '.join(t['tool'] for t in res['tool_calls'])}"
                   + (f" · {res['notice']}" if res.get("notice") else ""))

# ---------------------------------------------------------------- feature drift
ui.section("Feature drift (PSI)", "Development sample vs recent intake — every model input")
feats = pd.DataFrame(d["features"])
c1, c2 = st.columns([1.2, 1])
with c1:
    top = feats.head(15).iloc[::-1]
    fig = go.Figure(go.Bar(y=top["label"], x=top["psi"], orientation="h",
                           marker=dict(color=[ui.TIER_COLOR[s] for s in top["status"]], cornerradius=3),
                           customdata=top[["status", "importance"]].values,
                           hovertemplate="%{y}<br>PSI %{x:.3f} · %{customdata[0]}<br>model importance "
                                         "%{customdata[1]:.3f}<extra></extra>"))
    for x, lab, pos in ((d["thresholds"]["warning"], "warning 0.10", "bottom left"),
                        (d["thresholds"]["critical"], "critical 0.25", "bottom right")):
        fig.add_vline(x=x, line_width=1, line_color=ui.AXIS, annotation_text=lab, annotation_position=pos,
                      annotation_font=dict(size=10, color=ui.MUTED_INK))
    fig.update_layout(title="Top 15 features by PSI", xaxis_title="Population Stability Index",
                      yaxis=dict(gridcolor="rgba(0,0,0,0)"))
    ui.chart(fig, 460)
with c2:
    counts = d["counts"]
    st.markdown("<div class='rl-card'>" + "".join(
        f"<div style='display:flex;justify-content:space-between;padding:6px 0'>{ui.pill(s)}"
        f"<b>{counts[s]} features</b></div>" for s in ("Critical", "Warning", "Stable")) + "</div>",
        unsafe_allow_html=True)
    sel = st.selectbox("Compare distributions for", feats["label"].tolist())
    f = feats[feats.label == sel].iloc[0]
    bins = pd.DataFrame(f["bins"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=bins["bin"], y=bins["baseline"], name="Development", marker=dict(color=ui.MUTED, cornerradius=3)))
    fig.add_trace(go.Bar(x=bins["bin"], y=bins["current"], name="Recent intake", marker=dict(color=ui.SERIES[0], cornerradius=3)))
    fig.update_layout(title=f"{sel} · PSI {f['psi']:.3f} ({f['status']})", yaxis_tickformat=".0%", barmode="group",
                      xaxis_title="baseline decile bins")
    ui.chart(fig, 300)

show = feats[["label", "group", "psi", "status", "baseline_mean", "current_mean", "baseline_missing", "current_missing",
              "importance"]].copy()
st.dataframe(show, hide_index=True, use_container_width=True, height=260,
             column_config={"label": "Feature", "group": "Source", "psi": st.column_config.NumberColumn("PSI", format="%.3f"),
                            "status": "Status", "baseline_mean": st.column_config.NumberColumn("Dev mean", format="%.3g"),
                            "current_mean": st.column_config.NumberColumn("Intake mean", format="%.3g"),
                            "baseline_missing": st.column_config.NumberColumn("Dev missing", format="%.2f"),
                            "current_missing": st.column_config.NumberColumn("Intake missing", format="%.2f"),
                            "importance": st.column_config.NumberColumn("Importance (mean |SHAP|)", format="%.3f")})

# ---------------------------------------------------------------- population + performance
c3, c4 = st.columns(2)
with c3:
    seg = pd.DataFrame(d["segment_drift"])
    fig = go.Figure()
    lab = seg["dimension"].str.replace("_", " ") + ": " + seg["segment"].astype(str)
    fig.add_trace(go.Bar(y=lab, x=seg["baseline_share"], name="Development", orientation="h",
                         marker=dict(color=ui.MUTED, cornerradius=3)))
    fig.add_trace(go.Bar(y=lab, x=seg["current_share"], name="Recent intake", orientation="h",
                         marker=dict(color=ui.SERIES[0], cornerradius=3),
                         customdata=seg[["psi", "status"]].values,
                         hovertemplate="%{y}<br>%{x:.1%} of intake<br>dimension PSI %{customdata[0]:.3f} "
                                       "(%{customdata[1]})<extra></extra>"))
    fig.update_layout(title="Population changes — largest segment shift per dimension", barmode="group",
                      xaxis_tickformat=".0%", yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"))
    ui.chart(fig, 340)
with c4:
    ps = pd.DataFrame(api.get("/model/performance/segments"))
    if not ps.empty:
        ps = ps.sort_values(["dimension", "segment"])
        fig = go.Figure(go.Bar(x=ps["segment"], y=ps["auc"],
                               marker=dict(color=[ui.SERIES[0] if s == "vintage" else ui.SERIES[2] for s in ps["dimension"]],
                                           cornerradius=3),
                               customdata=ps[["n", "observed_default_rate", "mean_pd"]].values,
                               hovertemplate="%{x}<br>AUC %{y:.3f}<br>n=%{customdata[0]:,} · observed default "
                                             "%{customdata[1]:.1%} · mean PD %{customdata[2]:.1%}<extra></extra>"))
        fig.add_hline(y=perf["auc"], line_color=ui.INK2, line_width=1,
                      annotation_text=f"overall {perf['auc']:.3f}", annotation_font=dict(size=10, color=ui.INK2))
        fig.update_layout(title="Segment drift — AUC by vintage and product (test split)",
                          yaxis=dict(range=[0.5, 0.9], rangemode="normal"))
        ui.chart(fig, 340)
        st.caption("Small segments (~1,200 customers per vintage) carry ±0.03 AUC sampling noise.")

st.markdown(f"<div class='rl-insight' style='margin-top:12px'><div class='rl-insight-head'>Monitoring interpretation</div>"
            f"<p>{d['interpretation']}</p></div>", unsafe_allow_html=True)
