"""
generate_root_cause_analysis — quantitative RCA of a change in portfolio risk.

Comparisons
  vintage : later vintages vs earlier vintages of the booked portfolio (default: 2025 vs 2024)
  intake  : recent applications (application_test population) vs the booked portfolio
            (PD-based only — intake has no observed outcome yet)

Outputs (all computed from the database / model):
  * headline change of the metric
  * segment decomposition per dimension: mix effect (book shifted toward riskier
    segments) vs rate effect (segments themselves got riskier)
  * SHAP driver attribution: change in mean SHAP per feature between the two
    groups — this exactly decomposes the change in mean log-odds PD
  * behavioural shifts (utilisation, late payments, application velocity...)
  * early-warning / anomaly rate change
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ai.tools.base import FILTER_SCHEMA, compact, tool
from ai.tools.context import resolve
from backend.schemas.filters import FILTER_COLUMNS
from backend.services import portfolio_service as ps
from common.config import VINTAGES
from common.db import read_sql

DIMENSIONS = ["product", "customer_segment", "income_band", "region", "credit_score_band"]
BEHAVIOUR = {"cc_util_3m": "card utilisation (3m)", "inst_late_rate_12m": "late-payment rate (12m)",
             "prev_apps_365d": "applications in last 12m", "credit_to_income": "loan-to-income",
             "ext_source_mean": "external credit score", "bureau_opened_12m": "new bureau accounts (12m)",
             "prev_refused_365d": "refusals in last 12m", "annuity_to_income": "instalment-to-income"}
BEHAVIOUR_UNIT = {"cc_util_3m": "pct", "inst_late_rate_12m": "pct", "annuity_to_income": "pct", "credit_to_income": "x",
                  "ext_source_mean": "score"}


def _group_frame(f, extra_where: str = "", params_extra: dict | None = None) -> pd.DataFrame:
    where, params = ps._where(f, alias="p")
    cond = (where + (" AND " if where else "WHERE ") + extra_where) if extra_where else where
    cols = ", ".join(f"c.{c}" for c in BEHAVIOUR)
    return read_sql(f"""SELECT p.sk_id_curr, p.default_flag, p.pd, p.expected_loss, p.ead, p.current_exposure,
                               p.risk_tier, p.ews_score, p.anomaly_status, {', '.join('p.' + d for d in DIMENSIONS)}, {cols}
                        FROM v_portfolio p JOIN customer_360 c USING (sk_id_curr) {cond}""",
                    {**params, **(params_extra or {})})


def _metric(df: pd.DataFrame, metric: str) -> float:
    if metric == "default_rate":
        return float(df.default_flag.astype(float).mean())
    if metric == "el_rate":
        return float(df.expected_loss.sum() / df.ead.sum())
    if metric == "high_risk_share":
        return float(df.risk_tier.isin(["High", "Very High"]).mean())
    return float(df.pd.mean())


def _decompose(base: pd.DataFrame, comp: pd.DataFrame, dim: str, metric: str) -> dict:
    def seg(df):
        g = df.groupby(dim)
        out = pd.DataFrame({"n": g.size()})
        out["w"] = out.n / out.n.sum()
        out["r"] = [_metric(df[df[dim] == s], metric) for s in out.index]
        return out
    b, c = seg(base), seg(comp)
    j = b.join(c, lsuffix="_0", rsuffix="_1", how="outer").fillna({"w_0": 0, "w_1": 0, "n_0": 0, "n_1": 0})
    j["r_0"] = j.r_0.fillna(j.r_1)
    j["r_1"] = j.r_1.fillna(j.r_0)
    j["mix_effect"] = (j.w_1 - j.w_0) * j.r_0
    j["rate_effect"] = j.w_1 * (j.r_1 - j.r_0)
    j["total_effect"] = j.mix_effect + j.rate_effect
    j = j[(j.n_0 >= 100) | (j.n_1 >= 100)]
    rows = [{"segment": s, "customers_before": int(r.n_0), "customers_after": int(r.n_1), "share_before": r.w_0, "share_after": r.w_1, "metric_before": r.r_0, "metric_after": r.r_1,
             "mix_effect": r.mix_effect, "rate_effect": r.rate_effect, "total_effect": r.total_effect}
            for s, r in j.sort_values("total_effect", key=abs, ascending=False).iterrows()]
    return {"dimension": dim, "mix_effect_total": float(j.mix_effect.sum()),
            "rate_effect_total": float(j.rate_effect.sum()), "segments": rows[:4]}


def _shap_shift(base_ids, comp_ids, top: int = 8) -> list[dict]:
    from ml.explainability import bulk_contributions
    from ml.feature_engineering import FEATURES, build_features
    ids = list(base_ids) + list(comp_ids)
    c360 = read_sql("SELECT * FROM customer_360 WHERE sk_id_curr = ANY(:ids)", {"ids": [int(i) for i in ids]})
    X = build_features(c360)
    sv, _ = bulk_contributions(X)
    sv = pd.DataFrame(sv, index=X.index, columns=X.columns)
    d = sv.loc[sv.index.isin(set(comp_ids))].mean() - sv.loc[sv.index.isin(set(base_ids))].mean()
    d = d.sort_values(key=abs, ascending=False).head(top)
    return [{"feature": k, "label": FEATURES[k][0], "group": FEATURES[k][1], "mean_shap_change_logodds": float(v),
             "direction": "pushed risk up" if v > 0 else "pushed risk down"} for k, v in d.items()]


@tool("generate_root_cause_analysis",
      "Root-cause analysis of a change in portfolio risk. comparison='vintage' compares later vs earlier booked "
      "vintages (default 2025 vs 2024, or pass baseline_vintages/comparison_vintages); comparison='intake' compares "
      "recent applications with the booked portfolio. Returns the metric change, segment decomposition into mix vs "
      "rate effects, SHAP driver attribution, behavioural shifts and early-warning changes — all computed live.",
      {"type": "object", "properties": {
          "metric": {"type": "string", "enum": ["avg_pd", "default_rate", "el_rate", "high_risk_share"]},
          "comparison": {"type": "string", "enum": ["vintage", "intake"]},
          "baseline_vintages": {"type": "array", "items": {"type": "string"}},
          "comparison_vintages": {"type": "array", "items": {"type": "string"}},
          "filters": FILTER_SCHEMA}},
      "Generating root-cause analysis...")
def generate_root_cause_analysis(metric: str = "avg_pd", comparison: str = "vintage",
                                 baseline_vintages: list[str] | None = None,
                                 comparison_vintages: list[str] | None = None,
                                 filters: dict | None = None) -> dict:
    f = resolve(filters).model_copy(update={"vintage": []})
    if comparison == "intake":
        if metric == "default_rate":
            metric = "avg_pd"   # intake has no observed outcome yet
        base = _group_frame(f)
        comp = _group_frame(f.model_copy(update={"population": "intake"}))
        labels = ("booked portfolio", "recent intake (new applications)")
    else:
        b_v = baseline_vintages or [v for v in VINTAGES if v < "2025"]
        c_v = comparison_vintages or [v for v in VINTAGES if v >= "2025"]
        base = _group_frame(f.model_copy(update={"vintage": b_v}))
        comp = _group_frame(f.model_copy(update={"vintage": c_v}))
        labels = (f"vintages {', '.join(b_v)}", f"vintages {', '.join(c_v)}")
    if base.empty or comp.empty:
        return {"error": "One of the comparison groups is empty for this selection."}

    m0, m1 = _metric(base, metric), _metric(comp, metric)
    decomp = [_decompose(base, comp, d, metric) for d in DIMENSIONS if not (len(getattr(f, d, [])) == 1)]
    seg_effects = sorted([dict(s, dimension=d["dimension"]) for d in decomp for s in d["segments"]],
                         key=lambda s: s["total_effect"], reverse=(m1 >= m0))
    beh = []
    for col, lab in BEHAVIOUR.items():
        a, b = base[col].astype(float).mean(), comp[col].astype(float).mean()
        if pd.notna(a) and pd.notna(b):
            beh.append({"measure": lab, "unit": BEHAVIOUR_UNIT.get(col, "count"), "before": a, "after": b, "change": b - a,
                        "relative_change": (b - a) / a if a else None})
    beh.sort(key=lambda r: abs(r["relative_change"] or 0), reverse=True)
    rng = np.random.default_rng(0)
    bi = base.sk_id_curr.values if len(base) <= 3000 else rng.choice(base.sk_id_curr.values, 3000, replace=False)
    ci = comp.sk_id_curr.values if len(comp) <= 3000 else rng.choice(comp.sk_id_curr.values, 3000, replace=False)
    out = {
        "metric": metric, "selection": f.describe(), "baseline": labels[0], "comparison": labels[1],
        "baseline_customers": len(base), "comparison_customers": len(comp),
        "metric_before": m0, "metric_after": m1, "absolute_change": m1 - m0,
        "relative_change": (m1 - m0) / m0 if m0 else None,
        "material": abs((m1 - m0) / m0) >= 0.03 if m0 else None,
        "materiality_threshold_relative": 0.03,
        "top_segment_effects": seg_effects[:6],
        "decomposition_by_dimension": [{k: v for k, v in d.items() if k != "segments"} for d in decomp],
        "shap_driver_shift": _shap_shift(bi, ci),
        "shap_sample_per_group": int(min(len(bi), 3000)),
        "behavioural_shifts": beh[:6],
        "early_warning_rate": {"before": float((base.ews_score >= 31).mean()), "after": float((comp.ews_score >= 31).mean())},
        "anomaly_rate": {"before": float((base.anomaly_status != "Normal").mean()),
                         "after": float((comp.anomaly_status != "Normal").mean())},
        "notes": ["Vintages are simulated booking quarters (no calendar dates in source)."] if comparison == "vintage"
                 else ["Intake = application_test population; outcomes not yet observed, so PD is model-based."],
    }
    return compact(out)
