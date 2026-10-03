"""Early warning + anomaly detection tests."""
import numpy as np
import pandas as pd

from common.config import ANOMALY_BANDS, EWS_BANDS, band
from ml.anomaly import rule_triggers
from tests.conftest import requires_db


def test_band_boundaries():
    assert [band(s, EWS_BANDS) for s in (0, 30, 31, 60, 61, 80, 81, 100)] == \
        ["Low", "Low", "Medium", "Medium", "High", "High", "Critical", "Critical"]
    assert [band(s, ANOMALY_BANDS) for s in (10, 60, 61, 80, 81)] == ["Normal", "Normal", "Watch", "Watch", "Suspicious"]


def test_anomaly_rules_fire_on_synthetic_profiles():
    base = dict(sk_id_curr=1, credit_to_income=2.0, amt_income_total=1e5, prev_apps_90d=0, bureau_enquiries_1m=0,
                days_since_phone_change=500, id_document_age_years=5, address_mismatch_flags=0, cc_util_change=0.0,
                cc_util_3m=0.2, inst_payment_ratio=1.0, employment_years=5, credit_to_goods=1.1)
    rows = [base] * 200 + [dict(base, sk_id_curr=2, prev_apps_90d=4, days_since_phone_change=5,
                                id_document_age_years=0.1, credit_to_goods=2.0)]
    hits = rule_triggers(pd.DataFrame(rows))
    codes = set(hits[hits.sk_id_curr == 2].rule_code)
    assert {"APPLICATION_VELOCITY", "RECENT_CONTACT_CHANGE", "LOAN_GOODS_MISMATCH"} <= codes
    assert hits[hits.sk_id_curr == 1].empty
    assert hits.evidence.str.len().min() > 10       # every hit carries human-readable evidence


@requires_db
def test_ews_score_equals_trigger_points():
    from common.db import read_sql
    df = read_sql("""SELECT e.ews_score, COALESCE(SUM(CASE t.severity WHEN 'CRITICAL' THEN 35 WHEN 'HIGH' THEN 20
                                                             WHEN 'MEDIUM' THEN 10 END), 0) AS pts
                     FROM early_warning_signals e LEFT JOIN ews_triggers t USING (sk_id_curr)
                     GROUP BY e.sk_id_curr, e.ews_score""")
    assert (df.ews_score == np.minimum(100, df.pts)).all()
    assert df.ews_score.between(0, 100).all()


@requires_db
def test_ews_bands_are_risk_ordered():
    from backend.services.ews_service import validation
    v = {r["band"]: r["default_rate"] for r in validation() if r["signal"] == "ews"}
    assert v["Low"] < v["Medium"] < v["High"] < v["Critical"]


@requires_db
def test_anomaly_scores_and_disclaimer():
    from common.db import read_sql
    a = read_sql("SELECT anomaly_score, anomaly_status, isolation_pct, rule_score FROM anomaly_scores")
    assert a.anomaly_score.between(0, 100).all()
    assert set(a.anomaly_status) <= {"Normal", "Watch", "Suspicious"}
    assert (a.anomaly_status == "Suspicious").mean() < 0.05          # a review queue, not half the book
    from backend.services.ews_service import customer_anomaly
    assert "not a confirmed-fraud" in customer_anomaly("C100002")["disclaimer"]


@requires_db
def test_queue_is_ranked_and_filterable():
    from fastapi.testclient import TestClient
    from backend.main import app
    c = TestClient(app)
    q = c.get("/investigations/queue", params={"limit": 20}).json()["rows"]
    pr = [r["priority_score"] for r in q]
    assert pr == sorted(pr, reverse=True)
    q2 = c.get("/investigations/queue", params={"limit": 20, "product": "Revolving loans"}).json()["rows"]
    assert all(r["product"] == "Revolving loans" for r in q2)
    w = c.get("/customers/C342659/warnings").json()
    assert w["triggers"] and all({"severity", "evidence"} <= set(t) for t in w["triggers"])
    assert c.get("/customers/C5/warnings").status_code == 404
