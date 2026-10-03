"""
Offline analyst mode — used when no LLM is configured or the LLM service fails.

A deterministic planner routes the question to the same controlled tools the
LLM uses, then writes the answer from templates filled ONLY with tool outputs.
It is less flexible than the LLM, but it never invents numbers and keeps the
product fully demonstrable without an API key.
"""
from __future__ import annotations

import re
from typing import Callable

INTENTS = [
    ("customer", re.compile(r"\bc?\d{6}\b", re.I)),
    ("drift", re.compile(r"drift|psi|stabil|model (health|performance|monitor)|degrad", re.I)),
    ("anomaly", re.compile(r"anomal|suspicious|fraud|unusual|outlier", re.I)),
    ("priority", re.compile(r"investigat|priorit|first|which (accounts|customers)|work ?queue|review first", re.I)),
    ("segments", re.compile(r"segment|which (groups|products|regions)|becoming risk", re.I)),
    ("rca", re.compile(r"why .*(risk|loss|default|pd).*(increas|ris|grow|up|chang|decreas|fall)|root cause|rca|"
                       r"what.*driv.*(portfolio|risk)|(increas|ris|chang).*portfolio risk", re.I)),
    ("warnings", re.compile(r"early warning|alert|trigger|warning", re.I)),
    ("metrics", re.compile(r"portfolio|exposure|expected loss|default rate|average pd|kpi|how (big|risky)", re.I)),
]


def detect_intent(q: str) -> tuple[str, str | None]:
    m = re.search(r"\b(c?\d{6})\b", q, re.I)
    cid = ("C" + m.group(1).lstrip("cC")) if m else None
    for name, rx in INTENTS:
        if rx.search(q):
            if name == "customer" and not cid:
                continue
            return name, cid
    return "help", cid


def pct(v, d=1):
    return "n/a" if v is None else f"{v * 100:.{d}f}%"


def money(v):
    if v is None:
        return "n/a"
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"{v / div:,.2f}{suf}"
    return f"{v:,.0f}"


def fmt_measure(b: dict, key: str) -> str:
    v = b[key]
    u = b.get("unit")
    return pct(v) if u == "pct" else f"{v:.2f}x" if u == "x" else f"{v:.3f}" if u == "score" else f"{v:.2f}"


def pp(v):
    return "n/a" if v is None else f"{v * 100:+.2f} pp"


# --------------------------------------------------------------------------- plans
def plan_customer(call: Callable, cid: str) -> str:
    prof = call("get_customer_profile", customer_id=cid)
    risk = call("calculate_customer_risk", customer_id=cid)
    exp = call("explain_prediction", customer_id=cid, top_n=5)
    ews = call("get_early_warning_signals", customer_id=cid)
    an = call("get_anomaly_signals", customer_id=cid)
    if any(isinstance(x, dict) and x.get("error") for x in (prof, risk)):
        return f"I could not retrieve customer **{cid}**: {(prof or {}).get('error') or (risk or {}).get('error')}"
    pb, cu, bu, pa = prof["payment_behaviour"], prof["credit_utilisation"], prof["bureau"], prof["previous_applications"]
    tier = risk["risk_tier"]
    high = tier in ("High", "Very High")
    lines = [f"### Risk summary — {cid}",
             f"{cid} has a **PD of {pct(risk['pd'])}** (model), placing them in the **{tier}** tier with a risk score of "
             f"**{risk['risk_score']}/100** (riskier than {pct(risk['portfolio_pd_percentile'], 2 if risk['portfolio_pd_percentile'] > 0.99 else 0)} of customers). "
             f"Exposure is **{money(risk['current_exposure'])}** and Expected Loss **{money(risk['expected_loss'])}** "
             f"(LGD {risk['lgd']:.0%} assumption). Portfolio average PD is {pct(risk['portfolio_average_pd'])}."
             + ("" if high else f" Note: the model does **not** classify this customer as high risk."),
             "", "**Primary risk drivers** (SHAP, champion model)"]
    lines += [f"- {d['label']} = {d['display_value']} (+{d['shap_value']:.2f} log-odds)" for d in exp["top_risk_increasing"]]
    if exp["top_risk_decreasing"]:
        lines.append("- Mitigating: " + "; ".join(f"{d['label']} = {d['display_value']}" for d in exp["top_risk_decreasing"][:3]))
    lines += ["", "**Evidence**",
              f"- Payments: {pct(pb['inst_late_rate_12m'])} of instalments late in the last 12m "
              f"(prior 12m {pct(pb['inst_late_rate_prior'])}); {pb['inst_missed_12m']} missed/under-paid; max {pb['inst_max_days_late'] or 0:.0f} days late.",
              f"- Credit card: utilisation {pct(cu['cc_util_3m'])} over the last 3m vs {pct(cu['cc_util_prior'])} before."
              if cu["cc_util_3m"] is not None else "- Credit card: no card history on record.",
              f"- Bureau: {bu['bureau_active']} active of {bu['bureau_accounts']} accounts, {bu['bureau_delinquent']} currently overdue, "
              f"{bu['bureau_opened_12m']} opened in the last 12m.",
              f"- Applications: {pa['prev_apps_365d']} in the last 12m, {pa['prev_refused_365d']} refused; lifetime approval rate {pct(pa['prev_approval_rate'], 0)}.",
              f"- Affordability: loan-to-income {prof['application']['credit_to_income']:.2f}x, instalment-to-income {pct(prof['application']['annuity_to_income'])}.",
              "", f"**Early warning & anomaly signals** — EWS {ews.get('ews_score')} ({ews.get('ews_band')}); "
              f"anomaly {an.get('anomaly_score')} ({an.get('anomaly_status')})"]
    lines += [f"- {t['severity']}: {t['description']} — {t['evidence']}" for t in ews.get("triggers", [])] or ["- No early-warning triggers fired."]
    lines += [f"- Anomaly rule: {r['description']} — {r['evidence']}" for r in an.get("rules", [])]
    actions = []
    if any(t["trigger_code"] in ("PAYMENT_DETERIORATION", "MISSED_PAYMENTS") for t in ews.get("triggers", [])):
        actions.append("Review recent repayment history and contact the customer to understand the deterioration.")
    if any(t["trigger_code"] in ("HIGH_UTILISATION", "UTILISATION_INCREASE") for t in ews.get("triggers", [])):
        actions.append("Assess revolving-line utilisation; consider a limit review (analyst decision).")
    if an.get("anomaly_status") in ("Watch", "Suspicious"):
        actions.append("Route to fraud-risk review to verify identity / contact details (anomaly is not a fraud finding).")
    if high:
        actions.append("Place on enhanced monitoring and verify income/affordability before any new credit.")
    if not actions:
        actions.append("No immediate action indicated by the signals; continue standard monitoring.")
    lines += ["", "**Recommended analyst action** (decision support — final decision rests with the analyst)"]
    lines += [f"- {a}" for a in actions]
    return "\n".join(lines)


def plan_rca(call: Callable, question: str) -> str:
    m = call("get_portfolio_metrics", group_by="vintage")
    r = call("generate_root_cause_analysis", metric="avg_pd", comparison="vintage")
    ri = call("generate_root_cause_analysis", metric="avg_pd", comparison="intake")
    ew = call("get_early_warning_signals")
    s = m["summary"]
    lines = ["### Portfolio Risk RCA", f"Selection: {m['selection']} · {s['customers']:,} customers · exposure "
             f"{money(s['exposure'])} · observed default rate {pct(s['default_rate'], 2)} · average PD {pct(s['avg_pd'], 2)}.", ""]
    asked_increase = bool(re.search(r"increas|ris|grow|up", question, re.I))
    if r.get("material"):
        lines.append(f"Average PD moved from **{pct(r['metric_before'], 2)}** ({r['baseline']}) to **{pct(r['metric_after'], 2)}** "
                     f"({r['comparison']}), {pp(r['absolute_change'])} ({r['relative_change']:+.1%} relative).")
    else:
        lines.append(f"**Booked-portfolio risk has not materially increased.** Average PD is {pct(r['metric_before'], 2)} for 2024 "
                     f"vintages vs {pct(r['metric_after'], 2)} for 2025 ({pp(r['absolute_change'])}, {r['relative_change']:+.1%} "
                     f"relative — below the 3% materiality threshold)."
                     + (" The premise of the question is not supported by the data." if asked_increase else ""))
    if ri.get("relative_change") is not None:
        direction = "lower" if ri["relative_change"] < 0 else "higher"
        lines.append(f"Recent **intake** (new applications) has average PD {pct(ri['metric_after'], 2)} vs {pct(ri['metric_before'], 2)} "
                     f"for the booked book — {abs(ri['relative_change']):.1%} {direction}.")
    lines += ["", "**Main contributing segments** (2025 vs 2024; mix = shift in book composition, rate = segment got riskier)"]
    for sgm in r["top_segment_effects"][:4]:
        lines.append(f"- {sgm['segment']} ({sgm['dimension'].replace('_', ' ')}): PD {pct(sgm['metric_before'])} → "
                     f"{pct(sgm['metric_after'])}, share {pct(sgm['share_before'], 0)} → {pct(sgm['share_after'], 0)}; "
                     f"mix {pp(sgm['mix_effect'])}, rate {pp(sgm['rate_effect'])}")
    lines += ["", "**Main drivers** (change in mean SHAP contribution, 2025 vs 2024)"]
    lines += [f"- {d['label']}: {d['mean_shap_change_logodds']:+.4f} log-odds ({d['direction']})" for d in r["shap_driver_shift"][:4]]
    lines += ["", "**Observed behavioural changes**"]
    lines += [f"- {b['measure']}: {fmt_measure(b, 'before')} → {fmt_measure(b, 'after')} ({(b['relative_change'] or 0):+.1%})"
              for b in r["behavioural_shifts"][:4]]
    lines.append(f"- Early-warning alert rate: {pct(r['early_warning_rate']['before'])} → {pct(r['early_warning_rate']['after'])}; "
                 f"portfolio alerts now {ew['total_alerts']:,} ({ew['critical']} critical).")
    if ri.get("shap_driver_shift"):
        lines += ["", "**New-applicant mix vs booked book** (largest SHAP shifts)"]
        lines += [f"- {d['label']}: {d['mean_shap_change_logodds']:+.4f} ({d['direction']})" for d in ri["shap_driver_shift"][:3]]
    lines += ["", "**Potential business implication**",
              "- Headline risk is stable, but offsetting segment movements mean averages can hide pockets of deterioration "
              "(see the segment rows with positive rate effects).",
              "", "**Suggested investigation**",
              "- Drill into segments with positive rate effects on the Portfolio page.",
              "- Review customers with Critical early warnings (Early Warnings page).",
              "- Check feature drift for the intake population (Model Monitor) before relying on PDs for new applicants.",
              "", "_Vintages are simulated booking quarters — the source data has no calendar dates._"]
    return "\n".join(lines)


def plan_segments(call: Callable, question: str) -> str:
    r = call("generate_root_cause_analysis", metric="avg_pd", comparison="vintage")
    m = call("get_portfolio_metrics", group_by="customer_segment")
    mat = r["materiality_threshold_relative"]
    moved = [s for s in r["top_segment_effects"] if s["metric_before"]]
    for s in moved:
        s["rel"] = (s["metric_after"] - s["metric_before"]) / s["metric_before"]
    worse = [s for s in moved if s["rel"] >= mat]
    lines = ["### Segments becoming riskier (2025 vs 2024 vintages)",
             f"Overall average PD {pct(r['metric_before'], 2)} → {pct(r['metric_after'], 2)} ({pp(r['absolute_change'])})."]
    if worse:
        lines += ["", f"**Segments whose own risk rose materially (≥{mat:.0%} relative)**"]
        lines += [f"- {s['segment']} ({s['dimension'].replace('_', ' ')}): PD {pct(s['metric_before'])} → {pct(s['metric_after'])} "
                  f"({s['rel']:+.1%}; rate effect {pp(s['rate_effect'])})" for s in worse[:5]]
    else:
        lines.append(f"**No segment shows a material (≥{mat:.0%} relative) increase in its own risk.** Largest movements:")
        lines += [f"- {s['segment']} ({s['dimension'].replace('_', ' ')}): PD {pct(s['metric_before'])} → {pct(s['metric_after'])} "
                  f"({s['rel']:+.1%})" for s in sorted(moved, key=lambda s: -s["rel"])[:3]]
    lines += ["", "**Current risk by customer segment**"]
    lines += [f"- {b['segment']}: {b['customers']:,} customers, observed default {pct(b['default_rate'])}, avg PD {pct(b['avg_pd'])}, "
              f"high-risk share {pct(b['high_risk_share'])}" for b in m["breakdown"] if b["customers"] >= 30]
    return "\n".join(lines)


def plan_priority(call: Callable, question: str) -> str:
    p = call("get_investigation_priorities", limit=10)
    lines = ["### Customers to investigate first", f"Selection: {p['selection']}", "",
             "| # | Customer | Priority | PD | Tier | Exposure | Exp. loss | EWS | Anomaly | Key reason |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(p["customers"], 1):
        lines.append(f"| {i} | {c['customer_id']} | {c['priority_score']:.0f} | {pct(c['pd'])} | {c['risk_tier']} | "
                     f"{money(c['current_exposure'])} | {money(c['expected_loss'])} | {c['ews_score']} ({c['ews_band']}) | "
                     f"{c['anomaly_status']} | {c.get('ews_top_trigger') or 'model risk'} |")
    lines += ["", f"_Method: {p['method']['formula']}. Ranking is computed live; it prioritises review, it does not decide outcomes._"]
    return "\n".join(lines)


def plan_anomaly(call: Callable, question: str) -> str:
    a = call("get_anomaly_signals", limit=8)
    st = {s["anomaly_status"]: s for s in a["statuses"]}
    lines = ["### Behavioural anomalies", f"Selection: {a['selection']}",
             f"- Suspicious: {st.get('Suspicious', {}).get('customers', 0):,} · Watch: {st.get('Watch', {}).get('customers', 0):,} · "
             f"Normal: {st.get('Normal', {}).get('customers', 0):,}", "", "**Most common anomaly rules**"]
    lines += [f"- {r['description']}: {r['customers']:,} customers" for r in a["rules"][:6]]
    lines += ["", "**Most anomalous customers**"]
    lines += [f"- {c['customer_id']}: anomaly {c['anomaly_score']} ({c['anomaly_status']}), {c['n_rules']} rule(s), "
              f"PD {pct(c['pd'])}, exposure {money(c['current_exposure'])}" for c in a["most_anomalous"]]
    v = {r["band"]: r["default_rate"] for r in a["validation"]}
    lines += ["", f"Observed default rate — Normal {pct(v.get('Normal'))}, Watch {pct(v.get('Watch'))}, Suspicious {pct(v.get('Suspicious'))}.",
              f"_{a['disclaimer']}_ Recommended: route Suspicious cases to fraud-risk review for verification."]
    return "\n".join(lines)


def plan_drift(call: Callable, question: str) -> str:
    d = call("analyze_model_drift", top_n=8)
    if d.get("status") == "not_run":
        return "Model monitoring has not been run yet. Run `python -m ml.monitoring`."
    perf = d["performance"]
    lines = ["### Model drift & health", f"Overall status: **{d['overall_status']}** ({d['comparison']}).",
             f"- Performance (held-out test): AUC {perf['auc']:.3f}, Gini {perf['gini']:.3f}, KS {perf['ks']:.3f}, "
             f"Brier {perf['brier']:.4f}, calibration ratio {perf['calibration_ratio']:.2f}.",
             f"- Score PSI: {d['score_psi']['value']:.3f} ({d['score_psi']['status']}).",
             f"- Features: {d['counts']['Critical']} critical, {d['counts']['Warning']} warning, {d['counts']['Stable']} stable "
             f"(PSI thresholds {d['thresholds']['warning']} / {d['thresholds']['critical']}).", "", "**Most drifted features**"]
    lines += [f"- {f['label']}: PSI {f['psi']:.3f} ({f['status']}); mean {f['baseline_mean']:.3g} → {f['current_mean']:.3g}"
              for f in d["features"][:6]]
    if d.get("segment_drift"):
        lines += ["", "**Population mix shifts**"]
        lines += [f"- {s['dimension']}: {s['segment']} {pct(s['baseline_share'])} → {pct(s['current_share'])}"
                  for s in d["segment_drift"][:4]]
    lines += ["", "**Implication**: " + d.get("interpretation", "")]
    return "\n".join(lines)


def plan_warnings(call: Callable, question: str) -> str:
    e = call("get_early_warning_signals")
    lines = ["### Early warning overview", f"Selection: {e['selection']}",
             f"- Alerts: {e['total_alerts']:,} ({e['critical']} critical, {e['high']} high, {e['medium']:,} medium); "
             f"exposure under alert {money(e['alert_exposure'])}.", "", "**Most frequent triggers**"]
    lines += [f"- {t['severity']}: {t['description']} — {t['customers']:,} customers, observed default {pct(t['default_rate'])}"
              for t in e["triggers"][:6]]
    v = {r["band"]: r["default_rate"] for r in e["band_validation"]}
    lines.append(f"\nValidation — observed default by band: Low {pct(v.get('Low'))}, Medium {pct(v.get('Medium'))}, "
                 f"High {pct(v.get('High'))}, Critical {pct(v.get('Critical'))}.")
    return "\n".join(lines)


def plan_metrics(call: Callable, question: str) -> str:
    m = call("get_portfolio_metrics", group_by="risk_tier")
    s = m["summary"]
    lines = ["### Portfolio metrics", f"Selection: {m['selection']}",
             f"- Customers {s['customers']:,} · exposure {money(s['exposure'])} · observed default rate {pct(s['default_rate'], 2)}",
             f"- Average PD {pct(s['avg_pd'], 2)} · Expected Loss {money(s['expected_loss'])} ({pct(s['el_rate'], 2)} of EAD)",
             f"- High-risk customers {s['high_risk_customers']:,} ({pct(s['high_risk_share'])}) · active early warnings "
             f"{s['active_early_warnings']:,} · anomalies {s['anomalies']:,}", "", "**By risk tier**"]
    lines += [f"- {b['segment']}: {b['customers']:,} customers, exposure {money(b['exposure'])}, EL {money(b['expected_loss'])}, "
              f"observed default {pct(b['default_rate'])}" for b in m["breakdown"]]
    return "\n".join(lines)


HELP = ("I can investigate RiskLens data with controlled tools. Try:\n"
        "- *Why is customer C100002 high risk?*\n- *Why has portfolio risk increased?*\n"
        "- *Which customer segments are becoming risky?*\n- *Which accounts require investigation?*\n"
        "- *Are there signs of behavioural anomalies?*\n- *Is the model showing feature drift?*\n\n"
        "_Offline mode: answers are assembled from tool outputs by a deterministic planner. Configure LLM_API_KEY "
        "for free-form questions._")


def run(question: str, call: Callable) -> str:
    intent, cid = detect_intent(question)
    if intent == "customer":
        return plan_customer(call, cid)
    plan = {"rca": plan_rca, "segments": plan_segments, "priority": plan_priority, "anomaly": plan_anomaly,
            "drift": plan_drift, "warnings": plan_warnings, "metrics": plan_metrics}.get(intent)
    return plan(call, question) if plan else HELP
