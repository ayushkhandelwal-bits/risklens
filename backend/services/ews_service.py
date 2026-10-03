"""Early Warning + Behavioural Anomaly services and the investigation queue."""
from __future__ import annotations

from backend.schemas.filters import PortfolioFilters
from backend.services import portfolio_service as ps
from backend.services.customer_service import CustomerNotFound, normalise_id
from common.config import EWS_ALERT_THRESHOLD, PRIORITY_WEIGHTS
from common.db import read_sql

SEV_ORDER = "CASE t.severity WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 ELSE 3 END"


def _rec(df):
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def customer_warnings(customer_id: str) -> dict:
    cid, sk = normalise_id(customer_id)
    head = read_sql("SELECT ews_score, ews_band, n_triggers FROM early_warning_signals WHERE sk_id_curr = :sk", {"sk": sk})
    if head.empty:
        if read_sql("SELECT 1 FROM customer_360 WHERE sk_id_curr = :sk", {"sk": sk}).empty:
            raise CustomerNotFound(f"Customer {cid} not found")
        return {"customer_id": cid, "ews_score": None, "triggers": [], "message": "early warning not computed"}
    trig = read_sql(f"""SELECT t.trigger_code, t.severity, t.description, t.evidence
                        FROM ews_triggers t WHERE t.sk_id_curr = :sk ORDER BY {SEV_ORDER}, t.trigger_code""", {"sk": sk})
    out = {"customer_id": cid, **_rec(head)[0], "alert": bool(head.ews_score.iloc[0] >= EWS_ALERT_THRESHOLD),
           "triggers": _rec(trig),
           "method": "Rule-based Early Warning Score: CRITICAL 35 / HIGH 20 / MEDIUM 10 points per trigger, capped at 100."}
    return out


def customer_anomaly(customer_id: str) -> dict:
    cid, sk = normalise_id(customer_id)
    head = read_sql("""SELECT anomaly_score, anomaly_status, isolation_pct, rule_score, n_rules
                       FROM anomaly_scores WHERE sk_id_curr = :sk""", {"sk": sk})
    if head.empty:
        if read_sql("SELECT 1 FROM customer_360 WHERE sk_id_curr = :sk", {"sk": sk}).empty:
            raise CustomerNotFound(f"Customer {cid} not found")
        return {"customer_id": cid, "anomaly_score": None, "rules": []}
    rules = read_sql("SELECT rule_code, description, evidence FROM anomaly_rules WHERE sk_id_curr = :sk", {"sk": sk})
    return {"customer_id": cid, **_rec(head)[0], "rules": _rec(rules),
            "disclaimer": "Behavioural anomaly signal for analyst review — not a confirmed-fraud determination "
                          "(the source data has no verified fraud labels)."}


def summary(f: PortfolioFilters) -> dict:
    where, params = ps._where(f)
    bands = read_sql(f"""
        SELECT ews_band, COUNT(*) AS customers, SUM(current_exposure) AS exposure, AVG(default_flag::float) AS default_rate,
               AVG(pd) AS avg_pd
        FROM v_portfolio {where} GROUP BY ews_band""", params)
    order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    bands = bands.sort_values("ews_band", key=lambda s: s.map(order))
    pwhere, pparams = ps._where(f, alias="p")
    trig = read_sql(f"""
        SELECT t.trigger_code, t.description, t.severity, COUNT(*) AS customers,
               AVG(p.default_flag::float) AS default_rate
        FROM ews_triggers t JOIN v_portfolio p USING (sk_id_curr) {pwhere}
        GROUP BY 1, 2, 3 ORDER BY customers DESC""", pparams)
    total = int(bands.customers.sum())
    alerts = bands[bands.ews_band.isin(["Medium", "High", "Critical"])]
    return {"customers": total, "total_alerts": int(alerts.customers.sum()),
            "critical": int(bands.loc[bands.ews_band == "Critical", "customers"].sum()),
            "high": int(bands.loc[bands.ews_band == "High", "customers"].sum()),
            "medium": int(bands.loc[bands.ews_band == "Medium", "customers"].sum()),
            "alert_exposure": float(alerts.exposure.sum()) if not alerts.empty else 0.0,
            "bands": _rec(bands), "triggers": _rec(trig)}


def alerts(f: PortfolioFilters, bands: list[str] | None = None, trigger: str | None = None, limit: int = 100) -> list[dict]:
    where, params = ps._where(f, alias="p")
    clauses = [where[6:]] if where else []
    bands = bands or ["Medium", "High", "Critical"]
    names = []
    for i, b in enumerate(bands):
        params[f"band_{i}"] = b
        names.append(f":band_{i}")
    clauses.append(f"p.ews_band IN ({', '.join(names)})")
    if trigger:
        clauses.append("EXISTS (SELECT 1 FROM ews_triggers t WHERE t.sk_id_curr = p.sk_id_curr AND t.trigger_code = :trg)")
        params["trg"] = trigger
    df = read_sql(f"""
        SELECT p.customer_id, p.ews_score, p.ews_band, p.ews_triggers AS n_triggers, p.ews_top_trigger AS top_trigger,
               p.risk_tier, p.pd, p.current_exposure AS exposure, p.expected_loss, p.anomaly_status, p.product
        FROM v_portfolio p WHERE {' AND '.join(clauses)}
        ORDER BY p.ews_score DESC, p.expected_loss DESC NULLS LAST LIMIT :lim""", {**params, "lim": max(1, min(limit, 1000))})
    return _rec(df)


def validation() -> list[dict]:
    """Observed default rate by EWS band and anomaly status — evidence that the signals carry risk information."""
    ews = read_sql("""SELECT 'ews' AS signal, e.ews_band AS band, COUNT(*) n, AVG(c.default_flag::float) default_rate
                      FROM early_warning_signals e JOIN customer_360 c USING (sk_id_curr)
                      WHERE c.population = 'portfolio' GROUP BY 2""")
    an = read_sql("""SELECT 'anomaly' AS signal, a.anomaly_status AS band, COUNT(*) n, AVG(c.default_flag::float) default_rate
                     FROM anomaly_scores a JOIN customer_360 c USING (sk_id_curr)
                     WHERE c.population = 'portfolio' GROUP BY 2""")
    return _rec(ews) + _rec(an)


def anomaly_summary(f: PortfolioFilters) -> dict:
    where, params = ps._where(f, alias="p")
    st = read_sql(f"""SELECT p.anomaly_status, COUNT(*) AS customers, SUM(p.current_exposure) AS exposure
                      FROM v_portfolio p {where} GROUP BY 1""", params)
    rules = read_sql(f"""SELECT r.rule_code, r.description, COUNT(*) AS customers
                         FROM anomaly_rules r JOIN v_portfolio p USING (sk_id_curr) {where}
                         GROUP BY 1, 2 ORDER BY 3 DESC""", params)
    return {"statuses": _rec(st), "rules": _rec(rules)}


def investigation_queue(f: PortfolioFilters, limit: int = 50, status: list[str] | None = None) -> list[dict]:
    """Ranked by priority = 30% PD pct + 25% EL pct + 25% EWS + 20% anomaly (config.PRIORITY_WEIGHTS)."""
    where, params = ps._where(f)
    df = read_sql(f"""
        SELECT q.* FROM v_investigation_queue q
        WHERE q.sk_id_curr IN (SELECT sk_id_curr FROM v_portfolio {where})
        ORDER BY q.priority_score DESC LIMIT :lim""", {**params, "lim": max(1, min(limit, 500))})
    df = df.drop(columns=["sk_id_curr"])
    return _rec(df)


def priority_method() -> dict:
    return {"weights": PRIORITY_WEIGHTS,
            "formula": "priority = 100 × (0.30·PD percentile + 0.25·Expected-Loss percentile + 0.25·EWS/100 + 0.20·Anomaly/100)"}
