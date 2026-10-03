"""Investigation Center — the analyst's single-customer workspace."""
from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.components import filters, ui
from frontend.utils import api

ui.page_header("Investigate", "Investigation Center",
               "Everything known about one customer — profile, risk, payment and credit behaviour, bureau history, "
               "early-warning and anomaly signals, and the model's explanation — in one workspace.")

# ---------------------------------------------------------------- choose customer
try:
    queue = api.get("/investigations/queue", {**filters.as_params(), "limit": 25})
except api.APIError as e:
    ui.api_error(e)
    st.stop()

default = st.session_state.get("investigate_customer") or (queue["rows"][0]["customer_id"] if queue["rows"] else "")
c1, c2 = st.columns([1, 2])
with c1:
    typed = st.text_input("Customer ID", value=default, placeholder="e.g. C100002", key="inv_search")
with c2:
    opts = [f"{r['customer_id']} · priority {r['priority_score']:.0f} · {r['risk_tier']} · EWS {r['ews_score']}"
            for r in queue["rows"]]
    pick = st.selectbox("…or pick from the investigation queue (highest priority first)", ["—"] + opts)
cid = pick.split(" · ")[0] if pick != "—" else typed.strip()
if not cid:
    st.stop()

try:
    prof = api.get(f"/customers/{cid}")
except api.APIError as e:
    if e.status == 404:
        st.warning(f"{e}", icon="🔎")
    else:
        ui.api_error(e)
    st.stop()
cid = prof["identity"]["customer_id"]
st.session_state["investigate_customer"] = cid
if st.session_state.get("_audited") != cid:     # one audit event per opened customer
    try:
        api.get(f"/customers/{cid}", {"log": True}, cache=False)
        st.session_state["_audited"] = cid
    except api.APIError:
        pass

risk = prof.get("risk") or {}
ews = prof.get("early_warning") or {}
anom = prof.get("anomaly") or {}
ident = prof["identity"]

# ---------------------------------------------------------------- header
outcome = prof["observed_outcome"]["default_flag"]
outcome_txt = ("Observed outcome: <b>defaulted</b>" if outcome == 1 else "Observed outcome: repaid" if outcome == 0
               else "New application — outcome not yet observed")
st.markdown(
    f"<div class='rl-card' style='display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap'>"
    f"<div><div style='font-size:22px;font-weight:700'>{cid}</div>"
    f"<div style='font-size:13px;color:#52514e'>{ident['product']} · {ident['customer_segment']} · {ident['region']} · "
    f"income {ident['income_band']} · vintage {ident['vintage']} · score band {ident['credit_score_band']}</div>"
    f"<div style='font-size:12px;color:#898781;margin-top:2px'>{outcome_txt}</div></div>"
    f"<div style='display:flex;gap:8px;flex-wrap:wrap'>"
    f"{ui.pill('Risk: ' + str(risk.get('risk_tier', '—')), ui.TIER_STATUS.get(risk.get('risk_tier'), 'neutral'))}"
    f"{ui.pill('EWS: ' + str(ews.get('ews_band', '—')), ui.TIER_STATUS.get(ews.get('ews_band'), 'neutral'))}"
    f"{ui.pill('Anomaly: ' + str(anom.get('anomaly_status', '—')), ui.TIER_STATUS.get(anom.get('anomaly_status'), 'neutral'))}"
    f"</div></div>", unsafe_allow_html=True)

k = st.columns(6)
with k[0]:
    ui.kpi("Risk Score", f"{risk.get('risk_score', '—')}<span style='font-size:14px;color:#898781'>/100</span>",
           "PD percentile", "model")
with k[1]:
    ui.kpi("Probability of Default", ui.fmt_pct(risk.get("pd")), f"challenger {ui.fmt_pct(risk.get('pd_challenger'))}", "model")
with k[2]:
    ui.kpi("Exposure", ui.fmt_money(prof["exposure"]["current_exposure"]),
           f"total incl. other lenders {ui.fmt_money(prof['exposure']['total_exposure'])}", "observed")
with k[3]:
    ui.kpi("Expected Loss", ui.fmt_money(risk.get("expected_loss")),
           f"LGD {risk.get('lgd', 0):.0%} × EAD {ui.fmt_money(risk.get('ead'))}", "model")
with k[4]:
    ui.kpi("Early Warning", f"{ews.get('ews_score', '—')}", f"{ews.get('n_triggers', 0)} trigger(s)", "derived")
with k[5]:
    ui.kpi("Anomaly Score", f"{anom.get('anomaly_score', '—')}", f"{anom.get('n_rules', 0)} rule(s)", "derived")

b1, b2, _ = st.columns([2, 2, 5])
with b1:
    if st.button("✦ Investigate with AI", type="primary", use_container_width=True):
        st.session_state["ai_prefill"] = f"Why is customer {cid} high risk?"
        st.session_state["ai_autorun"] = True
        st.switch_page("frontend/pages/ai_analyst.py")

# ---------------------------------------------------------------- tabs
t_exp, t_pay, t_util, t_prev, t_bur, t_sig, t_prof = st.tabs(
    ["SHAP explanation", "Payment behaviour", "Credit utilisation", "Previous applications", "Bureau history",
     "Signals", "Customer profile"])

with t_exp:
    try:
        ex = api.get(f"/customers/{cid}/explanation", {"top_n": 8})
    except api.APIError as e:
        ui.api_error(e)
        ex = None
    if ex:
        items = ex["top_risk_increasing"] + ex["top_risk_decreasing"]
        items = sorted(items, key=lambda d: d["shap_value"])
        fig = go.Figure(go.Bar(
            y=[f"{d['label']} = {d['display_value']}" for d in items], x=[d["shap_value"] for d in items],
            orientation="h", marker=dict(color=["#d03b3b" if d["shap_value"] > 0 else "#2a78d6" for d in items],
                                         cornerradius=3),
            customdata=[d["group"] for d in items],
            hovertemplate="%{y}<br>%{customdata}<br>SHAP %{x:+.3f} log-odds<extra></extra>"))
        fig.update_layout(title=f"What moves this customer's PD from the portfolio average "
                                f"{ex['base_pd']:.1%} to {ex['pd']:.1%}",
                          xaxis_title="SHAP contribution (log-odds) — red raises risk, blue lowers it",
                          yaxis=dict(gridcolor="rgba(0,0,0,0)"))
        fig.add_vline(x=0, line_color=ui.AXIS, line_width=1)
        ui.chart(fig, 420)
        a, b = st.columns(2)
        with a:
            st.markdown("**Top risk drivers**")
            for d in ex["top_risk_increasing"][:5]:
                st.markdown(f"- {html.escape(d['description'])} <span style='color:#898781'>(+{d['shap_value']:.3f})</span>",
                            unsafe_allow_html=True)
        with b:
            st.markdown("**Top mitigating factors**")
            for d in ex["top_risk_decreasing"][:5]:
                st.markdown(f"- {html.escape(d['description'])} <span style='color:#898781'>({d['shap_value']:.3f})</span>",
                            unsafe_allow_html=True)
        st.caption(f"{ex['method']}. Reconstruction check |PD − sigmoid(base + ΣSHAP)| = {ex['reconstruction_error']:.1e}.")

hist = api.get(f"/customers/{cid}/history")
with t_pay:
    pb = prof["payment_behaviour"]
    k2 = st.columns(4)
    k2[0].metric("Instalments on record", ui.fmt_int(pb["inst_count"]))
    k2[1].metric("Paid late (all time)", ui.fmt_pct(pb["inst_late_rate"]))
    k2[2].metric("Paid late (last 12m)", ui.fmt_pct(pb["inst_late_rate_12m"]),
                 delta=None if pb["inst_late_rate_change"] is None else f"{pb['inst_late_rate_change'] * 100:+.1f} pts vs prior 12m",
                 delta_color="inverse")
    k2[3].metric("Max days late", ui.fmt_int(pb["inst_max_days_late"]))
    pay = pd.DataFrame(hist["payments"])
    if pay.empty:
        ui.empty_state("No instalment history in the Payment System for this customer.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=pay.month, y=pay.instalments - pay.late, name="Paid on time",
                             marker=dict(color=ui.MUTED, line=dict(color="#fff", width=1))))
        fig.add_trace(go.Bar(x=pay.month, y=pay.late, name="Paid late",
                             marker=dict(color=ui.STATUS["critical"], line=dict(color="#fff", width=1)),
                             customdata=pay.max_days_late,
                             hovertemplate="Month %{x}<br>%{y} late · max %{customdata} days<extra></extra>"))
        fig.update_layout(title="Instalments by month (months before application)", barmode="stack",
                          xaxis_title="months before application", yaxis_title="Instalments")
        ui.chart(fig, 300)

with t_util:
    cu = prof["credit_utilisation"]
    k3 = st.columns(4)
    k3[0].metric("Utilisation (last 3m)", ui.fmt_pct(cu["cc_util_3m"]))
    k3[1].metric("Utilisation (prior 9m)", ui.fmt_pct(cu["cc_util_prior"]))
    k3[2].metric("Max utilisation", ui.fmt_pct(cu["cc_util_max"]))
    k3[3].metric("Card months past due", ui.fmt_int(cu["cc_dpd_months"]))
    ut = pd.DataFrame(hist["utilisation"])
    if ut.empty:
        ui.empty_state("No credit card history in the Credit Card System for this customer.")
    else:
        fig = go.Figure(go.Scatter(x=ut.month, y=ut.utilisation, mode="lines+markers", line=dict(width=2, color=ui.ACCENT),
                                   marker=dict(size=8), customdata=ut[["balance", "credit_limit"]].values,
                                   hovertemplate="Month %{x}<br>Utilisation %{y:.0%}<br>Balance %{customdata[0]:,.0f} / "
                                                 "limit %{customdata[1]:,.0f}<extra></extra>"))
        fig.add_hline(y=1.0, line_color=ui.STATUS["critical"], line_width=1,
                      annotation_text="credit limit", annotation_font_color=ui.INK2)
        fig.update_layout(title="Card utilisation by month", yaxis_tickformat=".0%", xaxis_title="months before application")
        ui.chart(fig, 300)

with t_prev:
    pa = prof["previous_applications"]
    k4 = st.columns(4)
    k4[0].metric("Previous applications", ui.fmt_int(pa["prev_app_count"]))
    k4[1].metric("Approval rate", ui.fmt_pct(pa["prev_approval_rate"]))
    k4[2].metric("Applications (12m)", ui.fmt_int(pa["prev_apps_365d"]))
    k4[3].metric("Refusals (12m)", ui.fmt_int(pa["prev_refused_365d"]))
    prev = pd.DataFrame(hist["previous_applications"])
    if not prev.empty:
        st.dataframe(prev.drop(columns=["sk_id_prev"]), hide_index=True, use_container_width=True,
                     column_config={"amt_application": st.column_config.NumberColumn("Applied", format="%,.0f"),
                                    "amt_credit": st.column_config.NumberColumn("Granted", format="%,.0f"),
                                    "days_ago": "Days before application"})

with t_bur:
    bu = prof["bureau"]
    k5 = st.columns(4)
    k5[0].metric("Bureau accounts", ui.fmt_int(bu["bureau_accounts"]))
    k5[1].metric("Active", ui.fmt_int(bu["bureau_active"]))
    k5[2].metric("Currently overdue", ui.fmt_int(bu["bureau_delinquent"]))
    k5[3].metric("Credit history", f"{bu['bureau_history_years']:.1f} yrs" if bu["bureau_history_years"] else "—")
    bdf = pd.DataFrame(hist["bureau"])
    if bdf.empty:
        ui.empty_state("No records from the Credit Bureau for this customer.")
    else:
        st.dataframe(bdf, hide_index=True, use_container_width=True,
                     column_config={"amt_credit_sum": st.column_config.NumberColumn("Credit", format="%,.0f"),
                                    "amt_credit_sum_debt": st.column_config.NumberColumn("Debt", format="%,.0f"),
                                    "amt_credit_sum_overdue": st.column_config.NumberColumn("Overdue", format="%,.0f")})

with t_sig:
    a, b = st.columns(2)
    with a:
        w = api.get(f"/customers/{cid}/warnings")
        st.markdown(f"**Early Warning Signals** · score {w['ews_score']} ({w['ews_band']})")
        if w["triggers"]:
            st.markdown("<div class='rl-card'>" + "".join(
                f"<div class='rl-trigger'><div style='min-width:92px'>{ui.pill(t['severity'].title(), ui.TIER_STATUS[t['severity']])}</div>"
                f"<div><div class='rl-trigger-text'>{html.escape(t['description'])}</div>"
                f"<div class='rl-trigger-evidence'>{html.escape(t['evidence'])}</div></div></div>" for t in w["triggers"])
                + "</div>", unsafe_allow_html=True)
        else:
            ui.empty_state("No early-warning triggers fired.")
    with b:
        an = api.get(f"/customers/{cid}/anomaly")
        st.markdown(f"**Anomaly Signals** · score {an['anomaly_score']} ({an['anomaly_status']}) · isolation "
                    f"percentile {an['isolation_pct']:.0f}")
        if an["rules"]:
            st.markdown("<div class='rl-card'>" + "".join(
                f"<div class='rl-trigger'><div style='min-width:92px'>{ui.pill('Rule', 'warning')}</div>"
                f"<div><div class='rl-trigger-text'>{html.escape(r['description'])}</div>"
                f"<div class='rl-trigger-evidence'>{html.escape(r['evidence'])}</div></div></div>" for r in an["rules"])
                + "</div>", unsafe_allow_html=True)
        else:
            ui.empty_state("No anomaly rules fired.")
        st.caption(an.get("disclaimer", ""))

with t_prof:
    def dl(d: dict) -> str:
        return "<dl class='rl-dl'>" + "".join(
            f"<dt>{html.escape(k.replace('_', ' ').capitalize())}</dt><dd>{html.escape(_fmt(v))}</dd>" for k, v in d.items()) + "</dl>"

    def _fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:,.2f}" if abs(v) < 1000 else f"{v:,.0f}"
        return str(v)

    a, b, c = st.columns(3)
    a.markdown("**Demographics**<div class='rl-card'>" + dl(prof["demographics"]) + "</div>", unsafe_allow_html=True)
    b.markdown("**Application**<div class='rl-card'>" + dl(prof["application"]) + "</div>", unsafe_allow_html=True)
    c.markdown("**Exposure & flags**<div class='rl-card'>" + dl({**prof["exposure"], **prof["behavioural_flags"]})
               + "</div>", unsafe_allow_html=True)
