"""Model monitoring (PSI / drift) tests."""
import numpy as np
import pandas as pd

from ml.monitoring import categorical_psi, psi, psi_status
from tests.conftest import requires_db


def test_psi_identical_distributions_is_zero():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=20000))
    v, bins = psi(x, x.copy())
    assert v < 1e-6 and len(bins) == 11


def test_psi_detects_shift_and_missingness():
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(size=20000))
    assert psi(base, pd.Series(rng.normal(size=20000)))[0] < 0.01            # same distribution
    assert psi(base, pd.Series(rng.normal(0.5, 1, 20000)))[0] > 0.10         # mean shift -> warning+
    assert psi(base, pd.Series(rng.normal(1.5, 1, 20000)))[0] > 0.25         # large shift -> critical
    miss = pd.Series(np.where(rng.random(20000) < 0.4, np.nan, rng.normal(size=20000)))
    assert psi(base, miss)[0] > 0.25                                         # missing-value bin counts


def test_psi_thresholds():
    assert [psi_status(v) for v in (0.05, 0.10, 0.2, 0.25, 1.0)] == ["Stable", "Warning", "Warning", "Critical", "Critical"]


def test_categorical_psi():
    a = pd.Series(["Cash"] * 90 + ["Revolving"] * 10)
    b = pd.Series(["Cash"] * 99 + ["Revolving"] * 1)
    v, det = categorical_psi(a, b)
    assert v > 0.1 and det["segment"] in ("Cash", "Revolving")
    assert categorical_psi(a, a.copy())[0] < 1e-9


@requires_db
def test_monitoring_report_and_api():
    from fastapi.testclient import TestClient
    from backend.main import app
    c = TestClient(app)
    d = c.get("/model/drift", params={"top_n": 60}).json()
    assert d["overall_status"] in ("Stable", "Warning", "Critical")
    psis = [f["psi"] for f in d["features"]]
    assert psis == sorted(psis, reverse=True) and len(psis) == 52
    assert all("bins" not in f for f in d["features"])
    assert c.get("/model/drift", params={"include_bins": True, "top_n": 1}).json()["features"][0]["bins"]
    assert {"auc", "gini", "ks", "brier"} <= set(d["performance"])
    assert c.get("/model/performance/segments").json()
