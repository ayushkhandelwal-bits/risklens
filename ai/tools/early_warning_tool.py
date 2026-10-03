"""get_early_warning_signals / get_anomaly_signals."""
from __future__ import annotations

from ai.tools.base import FILTER_SCHEMA, compact, tool
from ai.tools.context import resolve
from ai.tools.customer_tool import CID
from backend.services import ews_service


@tool("get_early_warning_signals",
      "Early Warning System. With customer_id: that customer's EWS score (0-100), band and every trigger with "
      "severity and evidence. Without customer_id: portfolio alert counts by band, the most frequent triggers "
      "and their observed default rates for the selection, plus validation of bands against outcomes.",
      {"type": "object", "properties": {"customer_id": CID, "filters": FILTER_SCHEMA}},
      "Checking early warning signals...")
def get_early_warning_signals(customer_id: str | None = None, filters: dict | None = None) -> dict:
    if customer_id:
        return compact(ews_service.customer_warnings(customer_id))
    f = resolve(filters)
    s = ews_service.summary(f)
    s["selection"] = f.describe()
    s["band_validation"] = [r for r in ews_service.validation() if r["signal"] == "ews"]
    s["triggers"] = s["triggers"][:12]
    return compact(s)


@tool("get_anomaly_signals",
      "Behavioural anomaly / suspicious-activity signals (Isolation Forest + rules). NOT confirmed fraud — "
      "the data has no fraud labels. With customer_id: anomaly score, status, isolation percentile and rule "
      "evidence. Without: counts by status and most common anomaly rules for the selection, plus the most "
      "anomalous customers.",
      {"type": "object", "properties": {"customer_id": CID, "filters": FILTER_SCHEMA,
                                        "limit": {"type": "integer", "minimum": 1, "maximum": 20}}},
      "Scanning for behavioural anomalies...")
def get_anomaly_signals(customer_id: str | None = None, filters: dict | None = None, limit: int = 8) -> dict:
    if customer_id:
        return compact(ews_service.customer_anomaly(customer_id))
    f = resolve(filters)
    s = ews_service.anomaly_summary(f)
    from backend.services import portfolio_service as ps
    from common.db import read_sql
    where, params = ps._where(f)
    top = read_sql(f"""SELECT customer_id, anomaly_score, anomaly_status, anomaly_rules AS n_rules, pd, risk_tier,
                              current_exposure
                       FROM v_portfolio {where} ORDER BY anomaly_score DESC, current_exposure DESC LIMIT :l""",
                   {**params, "l": int(limit or 8)})
    s["most_anomalous"] = top.astype(object).where(top.notna(), None).to_dict(orient="records")
    s["validation"] = [r for r in ews_service.validation() if r["signal"] == "anomaly"]
    s["selection"] = f.describe()
    s["disclaimer"] = "Anomaly = unusual behaviour for review, not a fraud determination."
    return compact(s)
