"""Model monitoring service — reads the latest monitoring run (populated by ml/monitoring.py)."""
from __future__ import annotations

from common.config import PSI_CRITICAL, PSI_WARNING
from common.db import read_sql


def _rec(df):
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def latest_run() -> str | None:
    r = read_sql("SELECT run_id FROM model_monitoring ORDER BY run_at DESC LIMIT 1")
    return None if r.empty else r.run_id.iloc[0]


def drift_report(top_n: int = 10, include_bins: bool = False) -> dict:
    run = latest_run()
    if not run:
        return {"status": "not_run", "message": "Model monitoring has not been run (python -m ml.monitoring)."}
    df = read_sql("SELECT * FROM model_monitoring WHERE run_id = :r", {"r": run})
    feats = df[df.metric_type == "feature_psi"].sort_values("value", ascending=False)
    score = df[df.metric_type == "score_psi"].iloc[0]
    perf = df[df.metric_type == "performance"].iloc[0]
    seg = df[df.metric_type == "segment_mix"].sort_values("value", ascending=False)
    counts = feats.status.value_counts().to_dict()
    overall = "Critical" if score.status == "Critical" or counts.get("Critical", 0) >= 3 else \
        "Warning" if score.status != "Stable" or counts.get("Critical", 0) or counts.get("Warning", 0) else "Stable"
    def det(d):
        d = dict(d or {})
        if not include_bins:
            d.pop("bins", None)
        return d
    feat_rows = [{"feature": r.subject, "psi": r.value, "status": r.status, **det(r.detail)}
                 for r in feats.head(top_n).itertuples()]
    return {
        "run_id": run, "run_at": str(df.run_at.iloc[0]), "model": f"{perf.model_name} v{perf.model_version}",
        "comparison": f"{score.baseline} → {score.comparison}",
        "overall_status": overall,
        "thresholds": {"warning": PSI_WARNING, "critical": PSI_CRITICAL,
                       "rule": "PSI < 0.10 stable · 0.10–0.25 warning · ≥ 0.25 critical"},
        "performance": perf.detail,
        "score_psi": {"value": float(score.value), "status": score.status, **det(score.detail)},
        "counts": {k: int(counts.get(k, 0)) for k in ("Stable", "Warning", "Critical")},
        "features": feat_rows,
        "segment_drift": [{"dimension": r.subject, **(r.detail or {}), "psi": r.value, "status": r.status}
                          for r in seg.head(8).itertuples()],
        "interpretation": (df[df.metric_type == "interpretation"].detail.iloc[0] or {}).get("text", "")
        if (df.metric_type == "interpretation").any() else "",
    }


def feature_psi_all() -> list[dict]:
    run = latest_run()
    if not run:
        return []
    df = read_sql("""SELECT subject AS feature, value AS psi, status, detail FROM model_monitoring
                     WHERE run_id = :r AND metric_type = 'feature_psi' ORDER BY value DESC""", {"r": run})
    return [{"feature": r.feature, "psi": r.psi, "status": r.status, **(r.detail or {})} for r in df.itertuples()]


def performance_history() -> list[dict]:
    df = read_sql("""SELECT run_id, run_at::text AS run_at, subject, value, detail FROM model_monitoring
                     WHERE metric_type = 'performance_segment' ORDER BY run_at DESC, subject""")
    if df.empty:
        return []
    last = df.run_id.iloc[0]
    return [{"segment": r.subject, "auc": r.value, **(r.detail or {})} for r in df[df.run_id == last].itertuples()]
