"""Feature engineering, risk scoring, expected loss and SHAP tests."""
import numpy as np
import pandas as pd
import pytest

from common.config import LGD_ASSUMPTIONS, CCF_REVOLVING, pd_tier
from ml.feature_engineering import FEATURE_NAMES, build_features, format_value
from ml.scoring import expected_loss_frame, risk_score_from_pd
from tests.conftest import requires_db


def test_pd_tiers_are_monotonic():
    tiers = [pd_tier(p) for p in (0.01, 0.06, 0.12, 0.5)]
    assert tiers == ["Low", "Medium", "High", "Very High"]


def test_risk_score_is_percentile():
    dev = list(np.linspace(0, 1, 101))
    s = risk_score_from_pd(np.array([0.0, 0.505, 0.999, 1.0]), dev)
    assert list(s) == [0, 50, 99, 100]


def test_expected_loss_formula():
    c = pd.DataFrame({"product": ["Cash loans", "Revolving loans"], "amt_credit": [100_000.0, 50_000.0]})
    el = expected_loss_frame(c, np.array([0.1, 0.2]))
    assert el.lgd.tolist() == [LGD_ASSUMPTIONS["Cash loans"], LGD_ASSUMPTIONS["Revolving loans"]]
    assert el.ead.tolist() == [100_000.0, 50_000.0 * CCF_REVOLVING]
    assert np.allclose(el.expected_loss, [0.1 * 0.45 * 100_000, 0.2 * 0.65 * 50_000 * CCF_REVOLVING])


def test_feature_builder_encodings():
    row = {f: 1.0 for f in FEATURE_NAMES}
    row.update({"sk_id_curr": 1, "product": "Revolving loans", "education": "Higher education",
                "flag_own_car": "N", "flag_own_realty": "Y", "credit_to_income": np.inf})
    X = build_features(pd.DataFrame([row]))
    assert list(X.columns) == FEATURE_NAMES
    assert X.loc[1, "is_revolving"] == 1 and X.loc[1, "education_level"] == 3
    assert X.loc[1, "own_car"] == 0 and X.loc[1, "own_realty"] == 1
    assert np.isnan(X.loc[1, "credit_to_income"])          # inf -> NaN, never fed to the model


def test_sensitive_attributes_excluded():
    for banned in ("gender", "code_gender", "family_status", "cnt_children", "default_flag", "target", "sk_id_curr"):
        assert banned not in FEATURE_NAMES


def test_value_formatting():
    assert format_value("inst_late_rate_12m", 0.25) == "25.0%"
    assert format_value("own_car", 1) == "yes"
    assert format_value("ext_source_mean", None) == "missing"


@requires_db
def test_registered_models_beat_random_and_are_calibrated():
    from common.db import read_sql
    reg = read_sql("SELECT model_name, role, metrics FROM model_registry WHERE is_active")
    assert set(reg.role) == {"champion", "challenger"}
    for m in reg.metrics:
        assert m["auc"] > 0.70 and abs(m["gini"] - (2 * m["auc"] - 1)) < 1e-9
        assert 0.9 < m["calibration_ratio"] < 1.1
        assert m["brier"] < m["brier_baseline"]


@requires_db
def test_scores_are_consistent():
    from common.db import read_sql
    r = read_sql("SELECT pd, lgd, ead, expected_loss, risk_tier FROM risk_scores")
    assert len(r) == read_sql("SELECT COUNT(*) n FROM customer_360").n.iloc[0]
    assert np.allclose(r.expected_loss, r.pd * r.lgd * r.ead)
    assert ((r.pd > 0) & (r.pd < 1)).all()
    assert (r.risk_tier == r.pd.map(pd_tier)).all()


@requires_db
def test_shap_reconstructs_pd_and_matches_model():
    from ml.explainability import explain_customer
    e = explain_customer(100002)
    assert e["reconstruction_error"] < 1e-5
    from common.db import read_sql
    stored = read_sql("SELECT pd FROM risk_scores WHERE sk_id_curr = 100002").pd.iloc[0]
    assert abs(stored - e["pd"]) < 1e-6
    assert all(d["shap_value"] > 0 for d in e["top_risk_increasing"])
    assert all(d["shap_value"] < 0 for d in e["top_risk_decreasing"])


@requires_db
def test_risk_endpoints(client=None):
    from fastapi.testclient import TestClient
    from backend.main import app
    c = TestClient(app)
    r = c.get("/customers/C100002/risk").json()
    assert r["customer_id"] == "C100002" and "EL = PD" in r["el_formula"]
    assert c.get("/customers/C123/explanation").status_code == 404
    m = c.get("/model/metrics").json()
    assert {x["role"] for x in m} == {"champion", "challenger"}
