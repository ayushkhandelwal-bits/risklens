"""Data & Audit — data quality status, model registry (versioning) and the audit trail."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from frontend.components import ui
from frontend.utils import api

ui.page_header("Govern", "Data & Audit",
               "Data is checked before analytics run, every model version is registered, and every AI query, tool "
               "call, investigation and scoring run is recorded.")

t_dq, t_models, t_audit = st.tabs(["Data quality", "Model registry", "Audit trail"])

with t_dq:
    try:
        dq = api.get("/portfolio/data-quality")
    except api.APIError as e:
        ui.api_error(e)
        st.stop()
    total = dq["pass"] + dq["warn"] + dq["fail"]
    k = st.columns(5)
    with k[0]:
        st.markdown(f"<div class='rl-kpi'><div class='rl-kpi-label'>Data Quality Status</div>"
                    f"<div style='margin-top:10px'>{ui.pill(dq['status'])}</div>"
                    f"<div class='rl-kpi-foot'>run {dq['run_id']}</div></div>", unsafe_allow_html=True)
    with k[1]:
        ui.kpi("Total records", ui.fmt_int(dq["total_records"]), "loaded into the raw layer", "observed")
    with k[2]:
        ui.kpi("Checks passed", f"{dq['pass']}/{total}", "", "observed")
    with k[3]:
        ui.kpi("Warnings", ui.fmt_int(dq["warn"]), "documented & handled", "observed")
    with k[4]:
        ui.kpi("Failures", ui.fmt_int(dq["fail"]), "block analytics if > 0", "observed")
    cov = dq["coverage"]
    st.caption("Data completeness — " + " · ".join(f"{k.replace('_', ' ')} {v:.1%}" for k, v in cov.items()))
    a, b = st.columns([1, 2])
    with a:
        rc = pd.DataFrame([{"table": k, "rows": v} for k, v in dq["row_counts"].items()]).sort_values("rows", ascending=False)
        st.markdown("**Raw layer (one table per source system)**")
        st.dataframe(rc, hide_index=True, use_container_width=True,
                     column_config={"rows": st.column_config.NumberColumn("Rows", format="%,d")})
    with b:
        checks = pd.DataFrame(dq["checks"])
        cat = st.multiselect("Status", ["FAIL", "WARN", "PASS"], default=["FAIL", "WARN"])
        view = checks[checks.status.isin(cat)] if cat else checks
        st.dataframe(view[["status", "table_name", "check_name", "category", "metric_value", "threshold", "detail"]],
                     hide_index=True, use_container_width=True, height=360,
                     column_config={"metric_value": st.column_config.NumberColumn("Value", format="%.3f"),
                                    "detail": st.column_config.TextColumn("Detail", width="large")})
    st.caption("Source-level integrity checks (prefix 'source.') are computed on the FULL Kaggle files; content checks "
               "on the loaded sample. Sentinel codes such as DAYS_EMPLOYED = 365243 are documented and set to NULL.")

with t_models:
    models = api.get("/model/metrics")
    rows = [{"model": m["model_name"], "version": m["model_version"], "role": m["role"], "active": m["is_active"],
             "algorithm": m["algorithm"], "training_date": m["training_date"][:19], "features": len(m["features"]),
             "train_rows": m["train_rows"], "test_rows": m["test_rows"], "AUC": m["metrics"]["auc"],
             "Gini": m["metrics"]["gini"], "KS": m["metrics"]["ks"], "Brier": m["metrics"]["brier"],
             "artifact": m["artifact_path"].split("/")[-1].split("\\")[-1]} for m in models]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(c, format="%.4f") for c in ("AUC", "Gini", "KS", "Brier")})
    m = st.selectbox("Inspect model", [f"{r['model']} v{r['version']}" for r in rows])
    sel = next(x for x in models if f"{x['model_name']} v{x['model_version']}" == m)
    c1, c2 = st.columns(2)
    c1.markdown("**Hyper-parameters**")
    c1.json(sel["hyperparameters"], expanded=False)
    c2.markdown("**Features**")
    c2.json(sel["features"], expanded=False)

with t_audit:
    c1, c2 = st.columns([2, 4])
    with c1:
        action = st.selectbox("Action", ["All", "ai_query", "tool_call", "ai_response", "investigation_opened",
                                         "risk_scoring", "anomaly_scoring", "model_registered", "model_monitoring",
                                         "ai_error"])
    log = api.get("/audit/log", {"limit": 300, "action": None if action == "All" else action}, cache=False)
    df = pd.DataFrame(log)
    if df.empty:
        ui.empty_state("No audit events yet.")
    else:
        df["parameters"] = df["parameters"].map(lambda p: json.dumps(p)[:160] if p else "")
        st.dataframe(df[["ts", "user_name", "action", "tool", "customer_id", "status", "parameters", "result_summary",
                         "session_id"]], hide_index=True, use_container_width=True, height=480,
                     column_config={"ts": "Timestamp (UTC)", "user_name": "User", "result_summary":
                                    st.column_config.TextColumn("Result", width="large")})
