"""Risk tools: calculate_customer_risk, get_portfolio_metrics, get_investigation_priorities."""
from __future__ import annotations

import numpy as np
import xgboost as xgb

from ai.tools.base import FILTER_SCHEMA, compact, tool
from ai.tools.context import resolve
from ai.tools.customer_tool import CID
from backend.schemas.filters import FILTER_COLUMNS
from backend.services import ews_service, portfolio_service as ps, risk_service
from backend.services.customer_service import normalise_id
from common.db import read_sql


@tool("calculate_customer_risk",
      "Calculate a customer's credit risk with the champion model: re-scores the customer's current features "
      "live, and returns PD, challenger PD, risk score (0-100 percentile), risk tier, LGD/EAD assumptions, "
      "Expected Loss and where the customer sits in the portfolio.",
      {"type": "object", "properties": {"customer_id": CID}, "required": ["customer_id"]},
      "Calculating customer risk with the champion model...")
def calculate_customer_risk(customer_id: str) -> dict:
    stored = risk_service.customer_risk(customer_id)
    _, sk = normalise_id(customer_id)
    from ml.explainability import load_champion
    from ml.feature_engineering import build_features
    X = build_features(read_sql("SELECT * FROM customer_360 WHERE sk_id_curr = :sk", {"sk": sk}))
    live_pd = float(load_champion().predict(xgb.DMatrix(X, feature_names=list(X.columns)))[0])
    stored["live_rescored_pd"] = live_pd
    stored["matches_stored_score"] = bool(np.isclose(live_pd, stored["pd"], atol=1e-6))
    stored["portfolio_average_pd"] = ps.summary(resolve({}).model_copy(update={k: [] for k in FILTER_COLUMNS}))["avg_pd"]
    return compact(stored)


@tool("get_portfolio_metrics",
      "Portfolio KPIs for a selection: customers, exposure, observed default rate, average PD, Expected Loss, "
      "high-risk share, early-warning and anomaly rates. Optionally break down by a dimension "
      "(vintage, product, income_band, region, customer_segment, risk_tier, credit_score_band).",
      {"type": "object", "properties": {"filters": FILTER_SCHEMA,
                                        "group_by": {"type": "string", "enum": list(FILTER_COLUMNS)}}},
      "Retrieving portfolio metrics...")
def get_portfolio_metrics(filters: dict | None = None, group_by: str | None = None) -> dict:
    f = resolve(filters)
    out = {"selection": f.describe(), "summary": ps.summary(f)}
    out["summary"].pop("filters", None)
    if group_by:
        rows = ps.breakdown(group_by, f)
        out["breakdown_by"] = group_by
        out["breakdown"] = [{k: r[k] for k in ("segment", "customers", "exposure", "default_rate", "avg_pd",
                                               "expected_loss", "high_risk_share", "avg_ews")} for r in rows]
    return compact(out)


@tool("get_investigation_priorities",
      "Rank customers the risk team should investigate first, using live data: priority = 30% PD percentile + "
      "25% Expected Loss percentile + 25% Early Warning Score + 20% Anomaly Score. Returns the top customers with "
      "the evidence for each.",
      {"type": "object", "properties": {"filters": FILTER_SCHEMA,
                                        "limit": {"type": "integer", "minimum": 1, "maximum": 25}}},
      "Ranking customers for investigation...")
def get_investigation_priorities(filters: dict | None = None, limit: int = 10) -> dict:
    f = resolve(filters)
    rows = ews_service.investigation_queue(f, int(limit or 10))
    keep = ("customer_id", "priority_score", "pd", "risk_tier", "current_exposure", "expected_loss", "ews_score",
            "ews_band", "ews_top_trigger", "anomaly_score", "anomaly_status", "product", "customer_segment")
    return compact({"selection": f.describe(), "method": ews_service.priority_method(),
                    "customers": [{k: r.get(k) for k in keep} for r in rows]})
