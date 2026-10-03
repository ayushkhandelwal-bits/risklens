"""
Model monitoring for the champion PD model.

    python -m ml.monitoring

Baseline   = development sample (the TRAIN split the model was fitted on)
Comparison = recent application intake (application_test population — new
             applicants the model now scores; no outcomes yet)

Checks
  * Feature PSI (Population Stability Index) for every model input
      PSI = Σ (a_i − e_i) · ln(a_i / e_i)   over baseline-decile bins (+ a missing bin)
      < 0.10 Stable · 0.10–0.25 Warning · ≥ 0.25 Critical
  * Score PSI on the PD distribution
  * Segment mix drift (PSI on categorical business dimensions)
  * Performance on the held-out test split, and its stability by vintage and product
    (outcomes are only available for the booked portfolio)
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from common.config import PSI_CRITICAL, PSI_WARNING
from common.db import execute, read_sql
from ml.explainability import load_champion, load_metadata
from ml.feature_engineering import FEATURES, build_features, load_customer_frame

log = logging.getLogger("risklens.monitoring")
EPS = 1e-4
SEGMENTS = ["product", "customer_segment", "income_band", "region", "credit_score_band"]


def psi_status(v: float) -> str:
    return "Critical" if v >= PSI_CRITICAL else "Warning" if v >= PSI_WARNING else "Stable"


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> tuple[float, list[dict]]:
    """PSI with quantile bins learned on the baseline; missing values form their own bin."""
    e, a = expected.astype(float), actual.astype(float)
    e_nn = e.dropna()
    edges = np.unique(np.quantile(e_nn, np.linspace(0, 1, bins + 1))) if len(e_nn) else np.array([0.0, 1.0])
    if len(edges) < 2:
        edges = np.array([e_nn.min() - 1, e_nn.max() + 1])
    edges[0], edges[-1] = -np.inf, np.inf

    def dist(s):
        cnt = pd.cut(s.dropna(), edges, include_lowest=True, duplicates="drop").value_counts(sort=False).values
        cnt = np.append(cnt, s.isna().sum())
        return cnt / max(len(s), 1)
    pe, pa = dist(e), dist(a)
    pe, pa = np.clip(pe, EPS, None), np.clip(pa, EPS, None)
    contrib = (pa - pe) * np.log(pa / pe)
    labels = [f"bin {i + 1}" for i in range(len(pe) - 1)] + ["missing"]
    return float(contrib.sum()), [{"bin": l, "baseline": float(x), "current": float(y)} for l, x, y in zip(labels, pe, pa)]


def categorical_psi(expected: pd.Series, actual: pd.Series) -> tuple[float, dict]:
    pe = expected.fillna("missing").value_counts(normalize=True)
    pa = actual.fillna("missing").value_counts(normalize=True)
    idx = pe.index.union(pa.index)
    pe, pa = pe.reindex(idx).fillna(0).clip(lower=EPS), pa.reindex(idx).fillna(0).clip(lower=EPS)
    contrib = (pa - pe) * np.log(pa / pe)
    top = (pa - pe).abs().idxmax()
    return float(contrib.sum()), {"segment": top, "baseline_share": float(pe[top]), "current_share": float(pa[top])}


def interpret(feat: pd.DataFrame, score_psi: float, perf: dict) -> str:
    crit = feat[feat.status == "Critical"].feature.map(lambda f: FEATURES[f][0]).tolist()
    warn = feat[feat.status == "Warning"].feature.map(lambda f: FEATURES[f][0]).tolist()
    parts = [f"Score PSI is {score_psi:.3f} ({psi_status(score_psi).lower()}): the PD distribution of new applicants "
             f"{'has shifted materially' if score_psi >= PSI_WARNING else 'is close to the development sample'}."]
    if crit:
        parts.append(f"{len(crit)} input(s) show critical drift ({', '.join(crit[:4])}); these populations differ from "
                     "what the model was trained on, so PDs for new applicants should be monitored against early "
                     "outcomes and the features reviewed for data-pipeline or policy changes.")
    if warn:
        parts.append(f"{len(warn)} input(s) are in the warning band ({', '.join(warn[:4])}).")
    if not crit and not warn:
        parts.append("All model inputs are stable.")
    parts.append(f"Discrimination on the held-out test split is AUC {perf['auc']:.3f} (Gini {perf['gini']:.3f}); "
                 "performance cannot yet be measured on the intake because outcomes are not observed.")
    return " ".join(parts)


def main() -> str:
    meta = load_metadata()
    booster = load_champion()
    run_id = datetime.now(timezone.utc).strftime("mon_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:4]
    model_name, version = meta["champion"]["name"], meta["model_version"]

    c360 = load_customer_frame()
    X = build_features(c360)
    train_ids = set(meta["splits"]["train"])
    base_mask = X.index.isin(train_ids)
    cur_mask = (c360["population"] == "intake").values
    Xb, Xc = X[base_mask], X[cur_mask]
    pd_all = booster.predict(xgb.DMatrix(X, feature_names=list(X.columns)))
    rows = []

    def add(metric_type, subject, value, status, detail, baseline="development sample (train split)",
            comparison="recent intake (new applications)"):
        rows.append({"run_id": run_id, "model_name": model_name, "model_version": version, "metric_type": metric_type,
                     "subject": subject, "baseline": baseline, "comparison": comparison, "value": value,
                     "status": status, "detail": json.dumps(detail, default=float)})

    imp = read_sql("SELECT feature, mean_abs_shap FROM global_feature_importance WHERE population='portfolio'")
    imp = dict(zip(imp.feature, imp.mean_abs_shap))
    feat_rows = []
    for f in X.columns:
        v, bins = psi(Xb[f], Xc[f])
        st = psi_status(v)
        feat_rows.append({"feature": f, "psi": v, "status": st})
        add("feature_psi", f, v, st, {"label": FEATURES[f][0], "group": FEATURES[f][1],
                                      "baseline_mean": float(Xb[f].mean()) if Xb[f].notna().any() else None,
                                      "current_mean": float(Xc[f].mean()) if Xc[f].notna().any() else None,
                                      "baseline_missing": float(Xb[f].isna().mean()),
                                      "current_missing": float(Xc[f].isna().mean()),
                                      "importance": imp.get(f), "bins": bins})
    feat = pd.DataFrame(feat_rows)

    s_psi, s_bins = psi(pd.Series(pd_all[base_mask]), pd.Series(pd_all[cur_mask]))
    add("score_psi", "pd", s_psi, psi_status(s_psi),
        {"baseline_mean_pd": float(pd_all[base_mask].mean()), "current_mean_pd": float(pd_all[cur_mask].mean()),
         "bins": s_bins})

    cb, cc = c360[base_mask], c360[cur_mask]
    for d in SEGMENTS:
        v, det = categorical_psi(cb[d], cc[d])
        add("segment_mix", d, v, psi_status(v), det)

    # performance: test split overall + stability by vintage / product
    reg = read_sql("SELECT metrics FROM model_registry WHERE role='champion' AND is_active").metrics.iloc[0]
    perf = {k: reg[k] for k in ("auc", "gini", "ks", "brier", "calibration_ratio", "precision", "recall", "f1",
                                 "threshold", "max_calibration_gap")}
    add("performance", "test_split", perf["auc"], "Stable", perf, baseline="held-out test split", comparison="-")
    test_mask = X.index.isin(set(meta["splits"]["test"]))
    t = c360[test_mask].assign(pd=pd_all[test_mask])
    for dim in ("vintage", "product"):
        for seg, g in t.groupby(dim):
            if g.default_flag.nunique() == 2 and len(g) >= 300:
                auc = roc_auc_score(g.default_flag.astype(int), g.pd)
                add("performance_segment", f"{dim}={seg}", float(auc),
                    "Stable" if auc >= perf["auc"] - 0.03 else "Warning",
                    {"dimension": dim, "segment": seg, "n": int(len(g)), "gini": float(2 * auc - 1),
                     "mean_pd": float(g.pd.mean()), "observed_default_rate": float(g.default_flag.mean())},
                    baseline="held-out test split", comparison=str(seg))
    add("interpretation", "summary", None, None, {"text": interpret(feat, s_psi, perf)})

    df = pd.DataFrame(rows)
    from etl.load_postgres import load_frame
    execute("DELETE FROM model_monitoring WHERE run_id = :r", {"r": run_id})
    load_frame("model_monitoring_stage", df, if_exists="replace")
    execute("""INSERT INTO model_monitoring (run_id, model_name, model_version, metric_type, subject, baseline,
                   comparison, value, status, detail)
               SELECT run_id, model_name, model_version, metric_type, subject, baseline, comparison, value, status,
                      CAST(detail AS JSONB) FROM model_monitoring_stage; DROP TABLE model_monitoring_stage;""")
    from backend.services.audit_service import log_event
    log_event("model_monitoring", tool="ml.monitoring", user="system", parameters={"run_id": run_id},
              result_summary=f"score PSI {s_psi:.3f}; features {feat.status.value_counts().to_dict()}")
    log.info("monitoring run %s: score PSI %.3f, %s", run_id, s_psi, feat.status.value_counts().to_dict())
    return run_id


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
