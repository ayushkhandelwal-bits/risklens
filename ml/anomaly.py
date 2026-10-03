"""
Behavioural anomaly / suspicious-activity detection.

IMPORTANT: Home Credit has no verified fraud labels, so this is NOT a fraud
classifier and makes no claim to detect confirmed fraud. It surfaces customers
whose behaviour is unusual relative to the book, to prioritise human review.

Two complementary signals:
  1. Isolation Forest (unsupervised) on behavioural / application features ->
     percentile 0-100 within the book (how isolated the profile is).
  2. Transparent rule triggers commonly used in fraud-risk triage
     (application velocity, income/loan outliers, contact-detail changes...).

    anomaly_score = 0.6 x isolation percentile + 0.4 x rule score   (0-100)
    status: Normal / Watch / Suspicious  (bands in common/config.py)
"""
from __future__ import annotations

import io
import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from common.config import (ANOMALY_BANDS, ANOMALY_WEIGHT_MODEL, ANOMALY_WEIGHT_RULES, ISOLATION_CONTAMINATION,
                           RANDOM_SEED, band)
from common.db import get_engine
from ml.feature_engineering import load_customer_frame

log = logging.getLogger("risklens.anomaly")

IF_FEATURES = ["credit_to_income", "annuity_to_income", "credit_to_goods", "amt_income_total", "amt_credit",
               "employment_years", "prev_apps_90d", "prev_apps_365d", "prev_refused_365d", "bureau_enquiries_1m",
               "bureau_enquiries_12m", "bureau_opened_12m", "cc_util_3m", "cc_util_change", "cc_balance_change",
               "inst_payment_ratio", "inst_late_rate_change", "days_since_phone_change", "id_document_age_years",
               "address_mismatch_flags"]
LOG_FEATURES = ["amt_income_total", "amt_credit", "cc_balance_change", "days_since_phone_change"]
RULE_POINTS = 40


def _matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[IF_FEATURES].astype(float).replace([np.inf, -np.inf], np.nan).copy()
    for c in LOG_FEATURES:
        X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
    return X


def rule_triggers(df: pd.DataFrame) -> pd.DataFrame:
    """Return long-form rule hits: sk_id_curr, rule_code, description, evidence."""
    p99_cti = df["credit_to_income"].quantile(0.99)
    p99_inc = df["amt_income_total"].quantile(0.99)
    rules = [
        ("APPLICATION_VELOCITY", "Unusually high application frequency",
         (df.prev_apps_90d >= 3) | (df.bureau_enquiries_1m >= 2),
         lambda r: f"{int(r.prev_apps_90d)} applications in 90 days; {r.bureau_enquiries_1m:.0f} bureau enquiries in the last month"),
        ("LOAN_INCOME_OUTLIER", "Loan amount unusual relative to income",
         df.credit_to_income > p99_cti,
         lambda r: f"loan-to-income {r.credit_to_income:.1f}x (book 99th percentile {p99_cti:.1f}x)"),
        ("RECENT_CONTACT_CHANGE", "Recent phone and ID document change",
         (df.days_since_phone_change <= 30) & (df.id_document_age_years <= 0.5),
         lambda r: f"phone changed {r.days_since_phone_change:.0f} days and ID issued {r.id_document_age_years * 12:.0f} months before application"),
        ("ADDRESS_MISMATCH", "Multiple address / work-location mismatches",
         df.address_mismatch_flags >= 4,
         lambda r: f"{int(r.address_mismatch_flags)} of 6 registration / residence / work location checks mismatch"),
        ("UTILISATION_SPIKE", "Abnormal jump in card utilisation",
         df.cc_util_change >= 0.5,
         lambda r: f"utilisation rose {r.cc_util_change * 100:.0f} pts to {r.cc_util_3m * 100:.0f}%"),
        ("REPAYMENT_PATTERN", "Unusual repayment pattern",
         (df.inst_payment_ratio > 1.5) | (df.inst_payment_ratio < 0.8),
         lambda r: f"paid {r.inst_payment_ratio * 100:.0f}% of instalment amounts due"),
        ("INCOME_EMPLOYMENT_MISMATCH", "Very high income with short employment",
         (df.amt_income_total > p99_inc) & (df.employment_years < 1),
         lambda r: f"declared income {r.amt_income_total:,.0f} (top 1%) with {r.employment_years:.1f} years employed"),
        ("LOAN_GOODS_MISMATCH", "Loan far exceeds goods price",
         df.credit_to_goods > 1.5,
         lambda r: f"loan is {r.credit_to_goods:.2f}x the goods price"),
    ]
    rows = []
    for code, desc, mask, ev in rules:
        hit = df[mask.fillna(False)]
        for _, r in hit.iterrows():
            rows.append((int(r.sk_id_curr), code, desc, ev(r)))
    return pd.DataFrame(rows, columns=["sk_id_curr", "rule_code", "description", "evidence"])


def _copy(table: str, df: pd.DataFrame) -> None:
    raw = get_engine().raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute(f"TRUNCATE {table}")
            buf = io.StringIO()
            df.to_csv(buf, index=False, header=False, na_rep="\\N")
            buf.seek(0)
            with cur.copy(f"COPY {table} ({', '.join(df.columns)}) FROM STDIN WITH (FORMAT csv, NULL '\\N')") as cp:
                cp.write(buf.read())
            cur.execute(f"ANALYZE {table}")
        raw.commit()
    finally:
        raw.close()


def main() -> pd.DataFrame:
    t0 = time.time()
    df = load_customer_frame()
    X = _matrix(df)
    model = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler()),
                      ("iforest", IsolationForest(n_estimators=300, contamination=ISOLATION_CONTAMINATION,
                                                  random_state=RANDOM_SEED, n_jobs=-1))])
    model.fit(X)
    raw_score = -model.decision_function(X)          # higher = more anomalous
    iso_pct = pd.Series(raw_score).rank(pct=True).values * 100

    hits = rule_triggers(df)
    n_rules = hits.groupby("sk_id_curr").size().reindex(df.sk_id_curr).fillna(0).astype(int).values
    rule_score = np.minimum(100, n_rules * RULE_POINTS)
    score = np.round(ANOMALY_WEIGHT_MODEL * iso_pct + ANOMALY_WEIGHT_RULES * rule_score).astype(int)
    out = pd.DataFrame({"sk_id_curr": df.sk_id_curr.values, "isolation_raw": raw_score, "isolation_pct": iso_pct,
                        "rule_score": rule_score, "anomaly_score": score,
                        "anomaly_status": [band(s, ANOMALY_BANDS) for s in score], "n_rules": n_rules})
    _copy("anomaly_scores", out)
    _copy("anomaly_rules", hits)
    from backend.services.audit_service import log_event
    log_event("anomaly_scoring", tool="ml.anomaly", user="system",
              parameters={"customers": len(out), "contamination": ISOLATION_CONTAMINATION},
              result_summary=out.anomaly_status.value_counts().to_json())
    log.info("anomaly scoring: %s in %.1fs", out.anomaly_status.value_counts().to_dict(), time.time() - t0)
    return out


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
