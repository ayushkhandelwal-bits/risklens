"""
run_safe_sql_query — controlled, read-only SQL for the AI analyst.

Three independent layers:
  1. Static validation (this file): single statement, SELECT/WITH only, no DDL/DML
     or dangerous functions, only whitelisted relations, no system catalogs.
  2. Execution in a READ ONLY transaction with a statement timeout and a row cap.
  3. A least-privilege database role (sql/ai_readonly_role.sql) when available.
"""
from __future__ import annotations

import os
import re
import time
from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine, text

from ai.tools.base import compact, tool
from common.config import DATABASE_URL

ALLOWED_RELATIONS = {
    "customer_360", "customer_behaviour_monthly", "v_portfolio", "v_portfolio_by_vintage", "v_investigation_queue",
    "mv_behaviour_trend", "risk_scores", "customer_explanations", "global_feature_importance",
    "early_warning_signals", "ews_triggers", "anomaly_scores", "anomaly_rules", "model_registry",
    "model_monitoring", "data_quality_checks",
}
FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|copy|execute|call|do|vacuum|analyze|"
    r"reindex|cluster|lock|set|reset|comment|merge|refresh|listen|notify|prepare|deallocate|discard|"
    r"security|owner|into)\b|pg_sleep|pg_read|pg_ls|lo_import|lo_export|dblink|pg_catalog|information_schema|"
    r"pg_terminate|pg_cancel|set_config|current_setting|audit_log",
    re.IGNORECASE)
MAX_ROWS = 200


class UnsafeQuery(ValueError):
    pass


def _strip(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return sql.strip().rstrip(";").strip()


def _ctes(sql: str) -> set[str]:
    return {m.lower() for m in re.findall(r"(?:with|,)\s*([a-zA-Z_][\w]*)\s+as\s*\(", sql, flags=re.I)}


def validate(sql: str) -> str:
    if not sql or not sql.strip():
        raise UnsafeQuery("Empty query.")
    body = _strip(sql)
    if ";" in body:
        raise UnsafeQuery("Only a single statement is allowed.")
    if not re.match(r"^\s*(select|with)\b", body, re.I):
        raise UnsafeQuery("Only SELECT queries are allowed.")
    bad = FORBIDDEN.search(re.sub(r"'[^']*'", "''", body))   # ignore string literals
    if bad:
        raise UnsafeQuery(f"Disallowed keyword or object: '{bad.group(0)}'.")
    refs = {r.lower().split(".")[-1] for r in re.findall(r"\b(?:from|join)\s+([a-zA-Z_][\w\.]*)", body, re.I)}
    unknown = refs - ALLOWED_RELATIONS - _ctes(body)
    if unknown:
        raise UnsafeQuery(f"Relation(s) not permitted for the AI analyst: {sorted(unknown)}. "
                          f"Allowed: {sorted(ALLOWED_RELATIONS)}.")
    return body


@lru_cache(maxsize=1)
def _engine():
    """Prefer the least-privilege role; fall back to the app role (still in a READ ONLY transaction)."""
    url = os.getenv("AI_DATABASE_URL")
    if url:
        from common.config import _normalise_db_url
        url = _normalise_db_url(url)
    else:
        url = re.sub(r"//[^@]+@", "//risklens_ai:risklens_ai@", DATABASE_URL)
    eng = create_engine(url, pool_pre_ping=True)
    try:
        with eng.connect() as c:
            user = c.execute(text("SELECT current_user")).scalar()
        if user == "risklens_ai":
            return eng, "risklens_ai (least-privilege role)"
        return eng, f"{user} (read-only transaction)"
    except Exception:
        return create_engine(DATABASE_URL, pool_pre_ping=True), "application role (read-only transaction)"


def execute_safe(sql: str, max_rows: int = MAX_ROWS) -> dict:
    body = validate(sql)
    eng, role = _engine()
    t = time.time()
    with eng.connect() as conn:
        conn.execute(text("SET TRANSACTION READ ONLY"))
        conn.execute(text("SET LOCAL statement_timeout = 5000"))
        df = pd.read_sql(text(f"SELECT * FROM ({body}) AS ai_q LIMIT {int(max_rows) + 1}"), conn)
        conn.rollback()
    truncated = len(df) > max_rows
    df = df.head(max_rows)
    return {"columns": list(df.columns), "rows": compact(df.astype(object).where(df.notna(), None).values.tolist()),
            "row_count": len(df), "truncated": truncated, "elapsed_ms": int((time.time() - t) * 1000), "db_role": role}


@tool("run_safe_sql_query",
      "Run a read-only SQL SELECT against approved RiskLens analytical tables when no specialised tool answers "
      "the question. Single SELECT/WITH statement only; max 200 rows. Key relations: v_portfolio (one row per "
      "customer: customer_id, population, default_flag, vintage, product, region, income_band, customer_segment, "
      "credit_score_band, current_exposure, pd, risk_tier, expected_loss, ews_score, ews_band, anomaly_score, "
      "anomaly_status, cc_util_3m, inst_late_rate_12m, prev_apps_365d ...), customer_360 (all features), "
      "ews_triggers (sk_id_curr, trigger_code, severity, description, evidence), anomaly_rules, "
      "customer_explanations (sk_id_curr, feature, shap_value, direction), model_registry, model_monitoring. "
      "Use population = 'portfolio' for the booked book.",
      {"type": "object", "properties": {"sql": {"type": "string", "description": "A single SELECT statement."}},
       "required": ["sql"]},
      "Running controlled SQL query...")
def run_safe_sql_query(sql: str) -> dict:
    return execute_safe(sql)
