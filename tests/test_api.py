"""API endpoint tests (run against the live database through FastAPI's TestClient)."""
import pytest
from fastapi.testclient import TestClient

from tests.conftest import requires_db

pytestmark = requires_db


@pytest.fixture(scope="module")
def client():
    from backend.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["database"] is True


def test_filters_change_results(client):
    base = client.get("/portfolio/summary").json()
    rev = client.get("/portfolio/summary", params={"product": "Revolving loans"}).json()
    both = client.get("/portfolio/summary", params={"product": "Revolving loans", "region": "Region Tier 3"}).json()
    assert base["customers"] > rev["customers"] > both["customers"] > 0
    assert base["exposure"] > rev["exposure"] > both["exposure"]
    assert base["default_rate"] != rev["default_rate"]


def test_filter_result_matches_direct_sql(client):
    from common.db import read_sql
    api = client.get("/portfolio/summary", params={"credit_score_band": "E · Poor", "vintage": ["2025-Q1", "2025-Q2"]}).json()
    sql = read_sql("""SELECT COUNT(*) n, AVG(default_flag::float) dr FROM customer_360
                      WHERE population='portfolio' AND credit_score_band='E · Poor' AND vintage IN ('2025-Q1','2025-Q2')""").iloc[0]
    assert api["customers"] == sql.n
    assert abs(api["default_rate"] - sql.dr) < 1e-12


def test_trend_and_breakdown_respect_filters(client):
    t_all = client.get("/portfolio/trend").json()
    t_e = client.get("/portfolio/trend", params={"credit_score_band": "E · Poor"}).json()
    assert len(t_all) == len(t_e) == 8
    assert all(e["default_rate"] > a["default_rate"] for a, e in zip(t_all, t_e))
    seg = client.get("/portfolio/breakdown/product", params={"product": "Cash loans"}).json()
    assert [r["segment"] for r in seg] == ["Cash loans"]


def test_invalid_filter_rejected(client):
    r = client.get("/portfolio/summary", params={"vintage": "1999-Q9"})
    assert r.status_code == 422 and r.json()["error"] == "invalid_filter"
    r = client.get("/portfolio/summary", params={"population": "everyone"})
    assert r.status_code == 422
    r = client.get("/portfolio/breakdown/not_a_column")
    assert r.status_code == 400


def test_filter_injection_is_parameterised(client):
    r = client.get("/portfolio/summary", params={"product": "Cash loans' OR '1'='1"})
    assert r.status_code == 422      # unknown value — never reaches SQL text


def test_customer_lookup_returns_correct_customer(client):
    r = client.get("/customers/C100002")
    assert r.status_code == 200
    d = r.json()
    assert d["identity"]["customer_id"] == "C100002"
    assert client.get("/customers/100002").json()["identity"]["customer_id"] == "C100002"   # id normalisation


def test_invalid_customer_ids(client):
    assert client.get("/customers/C999999999").status_code == 404
    r = client.get("/customers/DROP TABLE")
    assert r.status_code == 404 and "valid customer id" in r.json()["message"]


def test_customer_list_sorting_and_search(client):
    r = client.get("/portfolio/customers", params={"sort": "exposure", "limit": 5}).json()
    ex = [row["exposure"] for row in r["rows"]]
    assert ex == sorted(ex, reverse=True) and r["total"] == 50000
    s = client.get("/portfolio/customers", params={"search": "C100002"}).json()
    assert s["total"] >= 1 and s["rows"][0]["customer_id"].startswith("C100002")


def test_behaviour_trend_and_insight(client):
    b = client.get("/portfolio/behaviour-trend").json()
    assert len(b) == 24 and b[0]["month"] == -24
    i = client.get("/portfolio/insight").json()
    assert i["lines"] and "vintages" in i["headline"]
