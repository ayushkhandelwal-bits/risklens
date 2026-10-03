"""Credit Risk — champion/challenger models, performance, explainability, Expected Loss."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.components import filters, ui
from frontend.utils import api

ui.page_header("Monitor", "Credit Risk",
               "Probability-of-default models, how well they separate and calibrate on unseen customers, what drives "
               "them, and the resulting Expected Loss.")

try:
    models = api.get("/model/metrics")
except api.APIError as e:
    ui.api_error(e)
    st.stop()
if not models:
    ui.empty_state("No registered models. Run <code>python -m ml.train</code>.")
    st.stop()

champ = next(m for m in models if m["role"] == "champion" and m["is_active"])
chall = next((m for m in models if m["role"] == "challenger" and m["is_active"]), None)

# ---------------------------------------------------------------- model cards
c1, c2 = st.columns(2)
for col, m, role in ((c1, champ, "Champion"), (c2, chall, "Challenger")):
    if not m:
        continue
    mt = m["metrics"]
    with col:
        st.markdown(
            f"<div class='rl-card'><div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<div><div class='rl-eyebrow'>{role}</div><div style='font-size:18px;font-weight:650'>{m['model_name']} "
            f"<span style='color:#898781;font-weight:500;font-size:14px'>v{m['model_version']}</span></div></div>"
            f"{ui.pill('Active', 'good')}</div>"
            f"<div style='font-size:12.5px;color:#52514e;margin:6px 0 10px'>{m['algorithm']} · trained "
            f"{m['training_date'][:16]} UTC · {m['train_rows']:,} train / {m['test_rows']:,} test · "
            f"{len(m['features'])} features</div>"
            f"<dl class='rl-dl'>"
            f"<dt>AUC (test)</dt><dd>{mt['auc']:.4f}</dd>"
            f"<dt>Gini</dt><dd>{mt['gini']:.4f}</dd>"
            f"<dt>KS</dt><dd>{mt['ks']:.4f}</dd>"
            f"<dt>Brier score (baseline {mt['brier_baseline']:.4f})</dt><dd>{mt['brier']:.4f}</dd>"
            f"<dt>Calibration (mean PD ÷ observed default)</dt><dd>{mt['calibration_ratio']:.3f}</dd>"
            f"<dt>Precision / Recall / F1 @ PD ≥ {mt['threshold']:.3f}</dt>"
            f"<dd>{mt['precision']:.3f} / {mt['recall']:.3f} / {mt['f1']:.3f}</dd>"
            f"<dt>Train AUC (overfit check)</dt><dd>{mt['train_auc']:.4f}</dd>"
            f"</dl></div>", unsafe_allow_html=True)
st.caption("All metrics are measured once on a held-out 20% test split never used for training, early stopping or "
           "threshold selection. Accuracy is not reported: with an ~8% default rate it is uninformative. "
           "The decision threshold maximises F1 on the validation split.")

tab_perf, tab_exp, tab_el, tab_seg = st.tabs(["Performance", "Explainability", "Expected Loss", "Risk by segment"])

# ---------------------------------------------------------------- performance
with tab_perf:
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for m, color in ((champ, ui.SERIES[0]), (chall, ui.SERIES[1])):
            if m:
                roc = m["curves"]["roc"]
                fig.add_trace(go.Scatter(x=[p["fpr"] for p in roc], y=[p["tpr"] for p in roc], mode="lines",
                                         name=f"{m['model_name']} (AUC {m['metrics']['auc']:.3f})",
                                         line=dict(width=2, color=color),
                                         hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=ui.AXIS, width=1),
                                 showlegend=False, hoverinfo="skip"))
        fig.update_layout(title="ROC Curve (test split)", xaxis_title="False positive rate",
                          yaxis_title="True positive rate")
        ui.chart(fig, 360)
    with b:
        fig = go.Figure()
        for m, color in ((champ, ui.SERIES[0]), (chall, ui.SERIES[1])):
            if m:
                cal = m["curves"]["calibration"]
                fig.add_trace(go.Scatter(x=[p["predicted"] for p in cal], y=[p["observed"] for p in cal],
                                         mode="lines+markers", name=m["model_name"], marker=dict(size=8),
                                         line=dict(width=2, color=color),
                                         customdata=[p["n"] for p in cal],
                                         hovertemplate="Predicted %{x:.1%}<br>Observed %{y:.1%}<br>n=%{customdata:,}<extra></extra>"))
        mx = max(p["predicted"] for p in champ["curves"]["calibration"]) * 1.05
        fig.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color=ui.AXIS, width=1),
                                 showlegend=False, hoverinfo="skip"))
        fig.update_layout(title="Calibration (test split, PD deciles)", xaxis_title="Mean predicted PD",
                          yaxis_title="Observed default rate", xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        ui.chart(fig, 360)

    params = filters.as_params()
    dist = api.get("/portfolio/risk-distribution", params)
    hist = dist["pd_histogram"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[h["label"] for h in hist], y=[h["customers"] for h in hist], name="Customers",
                         marker=dict(color=ui.ACCENT, cornerradius=4),
                         customdata=[h["observed_default_rate"] for h in hist],
                         hovertemplate="PD %{x}<br>%{y:,} customers<br>Observed default %{customdata:.1%}<extra></extra>"))
    fig.update_layout(title=f"PD Distribution · {filters.describe()}", yaxis_title="Customers",
                      xaxis_title="Predicted probability of default")
    ui.chart(fig, 280)

# ---------------------------------------------------------------- explainability
with tab_exp:
    imp = api.get("/model/importance", {"limit": 15})
    a, b = st.columns([1.3, 1])
    with a:
        d = list(reversed(imp))
        fig = go.Figure(go.Bar(y=[r["label"] for r in d], x=[r["mean_abs_shap"] for r in d], orientation="h",
                               marker=dict(color=ui.ACCENT, cornerradius=4),
                               customdata=[r["group"] for r in d],
                               hovertemplate="%{y}<br>%{customdata}<br>mean |SHAP| %{x:.3f}<extra></extra>"))
        fig.update_layout(title="Global feature importance (mean |SHAP|, log-odds)", margin=dict(l=8))
        ui.chart(fig, 460)
    with b:
        grp = pd.DataFrame(api.get("/model/importance", {"limit": 60})).groupby("group").mean_abs_shap.sum()
        grp = (grp / grp.sum()).sort_values()
        fig = go.Figure(go.Bar(y=grp.index, x=grp.values, orientation="h", marker=dict(color=ui.SERIES[2], cornerradius=4),
                               hovertemplate="%{y}<br>%{x:.1%} of total attribution<extra></extra>"))
        fig.update_layout(title="Attribution by data source", xaxis_tickformat=".0%")
        ui.chart(fig, 300)
        st.markdown("<div class='rl-card' style='font-size:13px;color:#52514e'>"
                    "<b>Why SHAP?</b> Each PD is decomposed into additive feature contributions that sum exactly to the "
                    "model's log-odds output, so every customer-level explanation is faithful to the model — not a "
                    "separate surrogate. Contributions are verified to reconstruct the PD on every explanation.</div>",
                    unsafe_allow_html=True)

    ui.section("Risk Driver Analysis", f"How often each feature is a top-5 risk-increasing driver · {filters.describe()}")
    drv = pd.DataFrame(api.get("/risk/drivers", filters.as_params()))
    if not drv.empty:
        top = drv.sort_values("share_up", ascending=False).head(12)
        st.dataframe(top[["label", "group", "share_up", "customers_up", "avg_shap"]], hide_index=True,
                     use_container_width=True,
                     column_config={"label": "Driver", "group": "Source",
                                    "share_up": st.column_config.ProgressColumn("Share of customers", format="%.1f%%",
                                                                                min_value=0, max_value=1),
                                    "customers_up": st.column_config.NumberColumn("Customers", format="%,d"),
                                    "avg_shap": st.column_config.NumberColumn("Avg SHAP (when top driver)", format="%.3f")})

# ---------------------------------------------------------------- expected loss
with tab_el:
    asm = api.get("/risk/assumptions")
    params = filters.as_params()
    s = api.get("/portfolio/summary", params)
    k = st.columns(4)
    with k[0]:
        ui.kpi("Expected Loss", ui.fmt_money(s["expected_loss"]), filters.describe()[:40], "model")
    with k[1]:
        ui.kpi("Exposure (EAD basis)", ui.fmt_money(s["exposure"]), "loan amount", "observed")
    with k[2]:
        ui.kpi("EL rate", ui.fmt_pct(s["el_rate"], 2), "EL ÷ EAD", "model")
    with k[3]:
        ui.kpi("LGD", " / ".join(f"{v:.0%}" for v in asm["lgd"].values()), "cash / revolving", "assumption")
    a, b = st.columns(2)
    for col, dim, title in ((a, "risk_tier", "Expected Loss by Risk Tier"), (b, "customer_segment", "Expected Loss by Segment")):
        rows = api.get(f"/risk/expected-loss/{dim}", params)
        if dim == "risk_tier":
            order = ["Low", "Medium", "High", "Very High"]
            rows = sorted(rows, key=lambda r: order.index(r["segment"]) if r["segment"] in order else 9)
            colors = [ui.TIER_COLOR.get(r["segment"], ui.ACCENT) for r in rows]
        else:
            rows = [r for r in rows if r["customers"] >= 30]
            colors = ui.ACCENT
        with col:
            fig = go.Figure(go.Bar(x=[r["segment"] for r in rows], y=[r["expected_loss"] for r in rows],
                                   marker=dict(color=colors, cornerradius=4),
                                   customdata=[[r["el_rate"], r["exposure"], r["customers"]] for r in rows],
                                   hovertemplate="%{x}<br>EL %{y:,.0f}<br>EL rate %{customdata[0]:.2%}<br>"
                                                 "Exposure %{customdata[1]:,.0f}<br>%{customdata[2]:,} customers<extra></extra>"))
            fig.update_layout(title=title, yaxis_title="Expected loss")
            ui.chart(fig, 300)
    st.markdown(
        f"<div class='rl-card' style='font-size:13px;line-height:1.6'><b>Methodology & assumptions</b><br>"
        f"{ui.tag('model')} PD — {asm['pd']}<br>"
        f"{ui.tag('assumption')} LGD — {', '.join(f'{k}: {v:.0%}' for k, v in asm['lgd'].items())}. {asm['lgd_note']}<br>"
        f"{ui.tag('assumption')} EAD — {asm['ead']}<br>"
        f"These assumptions are documented configuration, not observed Home Credit data.</div>",
        unsafe_allow_html=True)

# ---------------------------------------------------------------- risk by segment
with tab_seg:
    params = filters.as_params()
    st.caption(f"Selection: {filters.describe()} — change filters on the Portfolio page.")
    a, b = st.columns(2)
    for col, dim, title in ((a, "vintage", "Risk by Vintage"), (b, "credit_score_band", "Risk by Credit Score Band")):
        rows = [r for r in api.get(f"/portfolio/breakdown/{dim}", params) if r["segment"] != "2026-Q1"]
        with col:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[r["segment"] for r in rows], y=[r["default_rate"] for r in rows],
                                 name="Observed default rate", marker=dict(color=ui.SERIES[1], cornerradius=4)))
            fig.add_trace(go.Bar(x=[r["segment"] for r in rows], y=[r["avg_pd"] for r in rows], name="Average PD",
                                 marker=dict(color=ui.SERIES[0], cornerradius=4)))
            fig.update_layout(title=title, yaxis_tickformat=".0%", barmode="group")
            ui.chart(fig, 300)
