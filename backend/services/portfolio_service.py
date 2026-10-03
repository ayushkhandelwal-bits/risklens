"""
Portfolio Engine — every metric is computed in PostgreSQL from v_portfolio at
request time with the caller's filters. Nothing here is precomputed or static.

Metric provenance (shown in the UI):
  OBSERVED  default_rate, exposure, customers           (source data)
  MODEL     avg_pd, expected_loss, risk tiers           (risk engine)
  DERIVED   ews / anomaly rates                         (rule + ML engines)
"""
from __future__ import annotations

import time
from typing import Any

import pandas as pd

from backend.schemas.filters import FILTER_COLUMNS, PortfolioFilters, build_where
from common.config import DIMENSION_ORDER, EWS_ALERT_THRESHOLD, HIGH_RISK_TIERS
from common.db import read_sql

_HR = ", ".join(f"'{t}'" for t in HIGH_RISK_TIERS)

_options_cache: dict[str, Any] = {"ts": 0.0, "data": None}


def filter_options() -> dict[str, list[str]]:
    """Distinct values per filter dimension (domain values only change on ETL runs)."""
    if _options_cache["data"] is not None and time.time() - _options_cache["ts"] < 300:
        return _options_cache["data"]
    cols = ", ".join(f"ARRAY_AGG(DISTINCT {c}) FILTER (WHERE {c} IS NOT NULL) AS {k}" for k, c in FILTER_COLUMNS.items())
    row = read_sql(f"SELECT {cols} FROM v_portfolio").iloc[0]
    order = DIMENSION_ORDER
    data = {}
    for k in FILTER_COLUMNS:
        vals = list(row[k]) if row[k] is not None else []
        data[k] = [v for v in order[k] if v in vals] if k in order else sorted(vals)
    _options_cache.update(ts=time.time(), data=data)
    return data


def _where(f: PortfolioFilters, alias: str = ""):
    allowed = {k: set(v) for k, v in filter_options().items()}
    return build_where(f, allowed, alias)


def _records(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def summary(f: PortfolioFilters) -> dict:
    where, params = _where(f)
    df = read_sql(f"""
        SELECT COUNT(*)                                                     AS customers,
               COALESCE(SUM(current_exposure),0)                            AS exposure,
               AVG(default_flag::float)                                     AS default_rate,
               COUNT(default_flag)                                          AS customers_with_outcome,
               AVG(pd)                                                      AS avg_pd,
               SUM(expected_loss)                                           AS expected_loss,
               SUM(expected_loss) / NULLIF(SUM(ead),0)                      AS el_rate,
               COUNT(*) FILTER (WHERE risk_tier IN ({_HR}))                 AS high_risk_customers,
               AVG((risk_tier IN ({_HR}))::int)                             AS high_risk_share,
               SUM(current_exposure) FILTER (WHERE risk_tier IN ({_HR}))    AS high_risk_exposure,
               COUNT(*) FILTER (WHERE ews_score >= {EWS_ALERT_THRESHOLD})   AS active_early_warnings,
               AVG((ews_score >= {EWS_ALERT_THRESHOLD})::int)               AS early_warning_rate,
               COUNT(*) FILTER (WHERE ews_band = 'Critical')                AS critical_warnings,
               COUNT(*) FILTER (WHERE anomaly_status <> 'Normal')           AS anomalies,
               AVG((anomaly_status <> 'Normal')::int)                       AS anomaly_rate,
               COUNT(pd)                                                    AS scored_customers
        FROM v_portfolio {where}""", params)
    out = _records(df)[0]
    out["filters"] = f.describe()
    out["model_scored"] = bool(out["scored_customers"])
    return out


def trend(f: PortfolioFilters) -> list[dict]:
    """Risk by (simulated) booking vintage for the filtered population."""
    where, params = _where(f)
    df = read_sql(f"""
        SELECT vintage,
               COUNT(*)                                       AS customers,
               SUM(current_exposure)                          AS exposure,
               AVG(default_flag::float)                       AS default_rate,
               AVG(pd)                                        AS avg_pd,
               SUM(expected_loss)                             AS expected_loss,
               AVG((risk_tier IN ({_HR}))::int)               AS high_risk_share,
               AVG((ews_score >= {EWS_ALERT_THRESHOLD})::int) AS early_warning_rate,
               AVG(cc_util_3m)                                AS avg_recent_utilisation,
               AVG(inst_late_rate_12m)                        AS avg_recent_late_rate
        FROM v_portfolio {where}
        GROUP BY vintage ORDER BY vintage""", params)
    return _records(df)


def breakdown(dimension: str, f: PortfolioFilters) -> list[dict]:
    if dimension not in FILTER_COLUMNS:
        raise ValueError(f"Unknown dimension '{dimension}'")
    col = FILTER_COLUMNS[dimension]
    where, params = _where(f)
    df = read_sql(f"""
        SELECT {col} AS segment,
               COUNT(*)                         AS customers,
               SUM(current_exposure)            AS exposure,
               AVG(default_flag::float)         AS default_rate,
               AVG(pd)                          AS avg_pd,
               SUM(expected_loss)               AS expected_loss,
               AVG((risk_tier IN ({_HR}))::int) AS high_risk_share,
               AVG(ews_score)                   AS avg_ews
        FROM v_portfolio {where}
        GROUP BY {col} ORDER BY {col}""", params)
    if dimension in DIMENSION_ORDER:
        rank = {v: i for i, v in enumerate(DIMENSION_ORDER[dimension])}
        df = df.sort_values("segment", key=lambda s: s.map(lambda v: rank.get(v, 99))).reset_index(drop=True)
    tot = df.exposure.sum()
    df["exposure_share"] = df.exposure / tot if tot else None
    return _records(df)


PD_BINS = [0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0001]


def risk_distribution(f: PortfolioFilters) -> dict:
    where, params = _where(f)
    edges = "ARRAY[" + ",".join(str(b) for b in PD_BINS) + "]::float[]"
    hist = read_sql(f"""
        SELECT width_bucket(pd, {edges}) AS bucket, COUNT(*) AS customers,
               AVG(default_flag::float) AS observed_default_rate
        FROM v_portfolio {where} {'AND' if where else 'WHERE'} pd IS NOT NULL
        GROUP BY 1 ORDER BY 1""", params)
    hist["label"] = hist.bucket.map(lambda b: f"{PD_BINS[b-1]*100:.0f}–{min(PD_BINS[b],1)*100:.0f}%")
    tiers = read_sql(f"""
        SELECT risk_tier, COUNT(*) AS customers, SUM(current_exposure) AS exposure,
               SUM(expected_loss) AS expected_loss, AVG(pd) AS avg_pd, AVG(default_flag::float) AS default_rate
        FROM v_portfolio {where} {'AND' if where else 'WHERE'} risk_tier IS NOT NULL
        GROUP BY 1""", params)
    order = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}
    tiers = tiers.sort_values("risk_tier", key=lambda s: s.map(order))
    return {"pd_histogram": _records(hist), "tiers": _records(tiers)}


SORTABLE = {"priority": "priority_score", "pd": "pd", "exposure": "current_exposure",
            "expected_loss": "expected_loss", "ews": "ews_score", "anomaly": "anomaly_score"}


def customers(f: PortfolioFilters, sort: str = "expected_loss", limit: int = 50, offset: int = 0,
              search: str | None = None) -> dict:
    where, params = _where(f)
    sort_col = SORTABLE.get(sort, "expected_loss")
    if search:
        where = (where + " AND " if where else "WHERE ") + "customer_id ILIKE :search"
        params["search"] = f"%{search.strip()}%"
    limit = max(1, min(int(limit), 500))
    total = int(read_sql(f"SELECT COUNT(*) n FROM v_portfolio {where}", params).n.iloc[0])
    order = sort_col if sort_col != "priority_score" else "expected_loss"  # priority lives in queue view
    df = read_sql(f"""
        SELECT customer_id, vintage, product, customer_segment, region,
               current_exposure AS exposure, pd, risk_score, risk_tier, expected_loss,
               ews_score, ews_band, anomaly_score, anomaly_status, default_flag
        FROM v_portfolio {where}
        ORDER BY {order} DESC NULLS LAST, customer_id
        LIMIT {limit} OFFSET {int(offset)}""", params)
    return {"total": total, "rows": _records(df)}


def behaviour_trend(f: PortfolioFilters) -> list[dict]:
    """Month-by-month behaviour (months before application) for the filtered customers.
    Uses REAL relative time from the POS / card / payment systems."""
    where, params = _where(f)
    df = read_sql(f"""
        SELECT b.month,
               AVG(b.pos_dpd::float)                                              AS pos_dpd_rate,
               AVG(b.cc_balance / NULLIF(b.cc_limit, 0)) FILTER (WHERE b.cc_limit > 0) AS utilisation,
               SUM(b.inst_late)::float / NULLIF(SUM(b.inst_n), 0)                 AS late_rate,
               COUNT(*)                                                           AS customers_observed,
               COUNT(*) FILTER (WHERE b.cc_limit > 0)                             AS card_customers_observed
        FROM customer_behaviour_monthly b
        JOIN (SELECT sk_id_curr FROM v_portfolio {where}) ids USING (sk_id_curr)
        GROUP BY b.month ORDER BY b.month""", params)
    return _records(df)


def top_drivers(f: PortfolioFilters, limit: int = 8) -> list[dict]:
    """Aggregate customer-level SHAP drivers across the filtered population."""
    where, params = _where(f, alias="p")
    df = read_sql(f"""
        SELECT e.feature,
               COUNT(*) FILTER (WHERE e.direction = 'increases_risk') AS customers_affected,
               AVG(e.shap_value) FILTER (WHERE e.direction = 'increases_risk') AS avg_contribution,
               SUM(e.shap_value) FILTER (WHERE e.direction = 'increases_risk') AS total_contribution
        FROM customer_explanations e JOIN v_portfolio p USING (sk_id_curr)
        {where}
        GROUP BY e.feature
        HAVING COUNT(*) FILTER (WHERE e.direction = 'increases_risk') > 0
        ORDER BY total_contribution DESC
        LIMIT {int(limit)}""", params)
    from ml.feature_engineering import FEATURES
    df["label"] = df.feature.map(lambda f: FEATURES.get(f, (f,))[0])
    df["group"] = df.feature.map(lambda f: FEATURES.get(f, ("", "Other"))[1])
    return _records(df)


def data_quality() -> dict:
    run = read_sql("SELECT run_id, finished_at, row_counts FROM etl_runs ORDER BY finished_at DESC LIMIT 1")
    if run.empty:
        return {"status": "Unknown", "checks": []}
    run_id = run.run_id.iloc[0]
    checks = read_sql("""SELECT table_name, check_name, category, metric_value, threshold, status, detail
                         FROM data_quality_checks WHERE run_id = :r ORDER BY status DESC, table_name""", {"r": run_id})
    status = "Critical" if (checks.status == "FAIL").any() else ("Warning" if (checks.status == "WARN").any() else "Healthy")
    counts = run.row_counts.iloc[0]
    completeness = read_sql("""
        SELECT 1 - AVG(CASE WHEN ext_source_mean IS NULL THEN 1 ELSE 0 END) AS score_coverage,
               AVG(CASE WHEN inst_count > 0 THEN 1 ELSE 0 END)            AS payment_history_coverage,
               AVG(CASE WHEN bureau_accounts > 0 THEN 1 ELSE 0 END)       AS bureau_coverage,
               AVG(CASE WHEN cc_util_avg IS NOT NULL THEN 1 ELSE 0 END)   AS card_coverage
        FROM customer_360""")
    return {"status": status, "run_id": run_id, "finished_at": str(run.finished_at.iloc[0]),
            "total_records": int(sum(counts.values())), "row_counts": counts,
            "pass": int((checks.status == "PASS").sum()), "warn": int((checks.status == "WARN").sum()),
            "fail": int((checks.status == "FAIL").sum()),
            "coverage": _records(completeness)[0], "checks": _records(checks)}


def concentration(f: PortfolioFilters) -> list[dict]:
    """Exposure / expected-loss concentration by PD decile (decile 10 = riskiest)."""
    where, params = _where(f)
    df = read_sql(f"""
        WITH d AS (
            SELECT NTILE(10) OVER (ORDER BY pd) AS decile, current_exposure, expected_loss, default_flag, pd
            FROM v_portfolio {where} {'AND' if where else 'WHERE'} pd IS NOT NULL)
        SELECT decile, COUNT(*) AS customers, MIN(pd) AS pd_min, MAX(pd) AS pd_max, AVG(pd) AS avg_pd,
               AVG(default_flag::float) AS default_rate,
               SUM(current_exposure) / SUM(SUM(current_exposure)) OVER () AS exposure_share,
               SUM(expected_loss) / NULLIF(SUM(SUM(expected_loss)) OVER (), 0) AS el_share
        FROM d GROUP BY decile ORDER BY decile""", params)
    return _records(df)
