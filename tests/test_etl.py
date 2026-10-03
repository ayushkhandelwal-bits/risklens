"""ETL + data validation tests (pure-python parts run without a database)."""
import numpy as np
import pandas as pd

from etl.cleaning import clean
from dataclasses import asdict

from etl.data_quality import checks_from_profile, content_checks as _content_checks


def content_checks(f):
    return pd.DataFrame([asdict(c) for c in _content_checks(f)])
from etl.transformations import transform
from tests.conftest import requires_db


def _frames():
    app = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3], "TARGET": [0, 1, 0],
        "DAYS_EMPLOYED": [-100, 365243, -2000], "DAYS_BIRTH": [-12000, -15000, -20000],
        "CODE_GENDER": ["F", "XNA", "M"], "AMT_INCOME_TOTAL": [1e5, 2e5, 3e5], "AMT_CREDIT": [5e5, 4e5, 3e5],
    })
    test = app.drop(columns="TARGET").assign(SK_ID_CURR=[10, 11, 12])
    prev = pd.DataFrame({"SK_ID_PREV": [100, 101], "SK_ID_CURR": [1, 2], "AMT_CREDIT": [1.0, 2.0],
                         **{c: [365243, -5] for c in ("DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE",
                                                      "DAYS_LAST_DUE_1ST_VERSION", "DAYS_LAST_DUE", "DAYS_TERMINATION")}})
    inst = pd.DataFrame({"SK_ID_PREV": [100, 999], "SK_ID_CURR": [1, 2], "NUM_INSTALMENT_VERSION": [1, 1],
                         "NUM_INSTALMENT_NUMBER": [1, 1], "DAYS_ENTRY_PAYMENT": [-5, -6], "AMT_PAYMENT": [10.0, 5.0]})
    cc = pd.DataFrame({"SK_ID_PREV": [100], "SK_ID_CURR": [1], "MONTHS_BALANCE": [-1],
                       "AMT_BALANCE": [5000.0], "AMT_CREDIT_LIMIT_ACTUAL": [1000.0]})
    pos = pd.DataFrame({"SK_ID_PREV": [100], "SK_ID_CURR": [1], "MONTHS_BALANCE": [-1]})
    bureau = pd.DataFrame({"SK_ID_CURR": [1], "SK_ID_BUREAU": [7], "DAYS_CREDIT_ENDDATE": [10.0]})
    bb = pd.DataFrame({"SK_ID_BUREAU": [7], "MONTHS_BALANCE": [-1], "STATUS": ["0"]})
    return {"application_train": app, "application_test": test, "previous_application": prev,
            "installments": inst, "credit_card": cc, "pos_cash": pos, "bureau": bureau, "bureau_balance": bb}


def test_cleaning_handles_sentinels_and_names():
    out, log = clean(_frames())
    app = out["application_train"]
    assert all(c == c.lower() for c in app.columns)
    assert app.loc[app.sk_id_curr == 2, "days_employed"].isna().all()
    assert app.loc[app.sk_id_curr == 2, "days_employed_anomaly"].iloc[0] == 1
    assert app.loc[app.sk_id_curr == 2, "code_gender"].isna().all()
    assert out["previous_application"]["days_first_drawing"].isna().sum() == 1
    assert any(r["action"] == "days_employed_365243_to_null" and r["rows_affected"] == 1 for r in log)


def test_transform_unifies_populations():
    out, _ = clean(_frames())
    t = transform(out)
    app = t["raw_application"]
    assert set(app.population) == {"portfolio", "intake"}
    assert app.loc[app.population == "intake", "target"].isna().all()
    assert app.customer_id.str.startswith("C").all()
    assert app.sk_id_curr.is_unique


def test_quality_checks_detect_problems():
    dq = content_checks(_frames())
    row = lambda t, c: dq[(dq.table_name == t) & (dq.check_name == c)].iloc[0]
    assert row("application", "impossible_days_employed").metric_value == 2  # train + intake copy
    assert row("application", "impossible_days_employed").status == "WARN"
    assert row("credit_card", "impossible_amt_balance").metric_value == 1       # balance 5x limit
    assert row("installments", "missing_previous_application_pct").metric_value == 50.0  # SK_ID_PREV 999 orphan
    assert set(dq.status) <= {"PASS", "WARN", "FAIL"}


def test_duplicate_keys_are_flagged():
    f = _frames()
    f["application_train"] = pd.concat([f["application_train"], f["application_train"].iloc[[0]]])
    dq = content_checks(f)
    r = dq[(dq.table_name == "application_train") & (dq.check_name == "duplicate_key_rows")].iloc[0]
    assert r.metric_value == 1 and r.status == "FAIL"


def test_profile_checks():
    prof = {"row_counts_full": {"application_train": 10, "application_test": 5, "bureau": 3, "bureau_balance": 9,
                                "previous_application": 4, "installments": 8, "pos_cash": 8, "credit_card": 8},
            "duplicate_applicant_ids": 0, "bureau_without_applicant_pct": 0.0, "duplicate_bureau_ids": 0,
            "bureau_balance_accounts": 3, "bureau_balance_without_bureau_pct": 60.0, "duplicate_prev_ids": 0,
            **{f"{k}_without_previous_application_pct": 2.0 for k in ("installments", "pos_cash", "credit_card")},
            **{f"{k}_distinct_prev": 4 for k in ("installments", "pos_cash", "credit_card")}}
    checks = {c.check_name + c.table_name: c.status for c in checks_from_profile(prof)}
    assert checks["bureau_balance_without_bureau_pctsource.bureau_balance"] == "FAIL"
    assert checks["duplicate_applicant_idssource.application"] == "PASS"


@requires_db
def test_customer_360_integrity():
    from common.db import read_sql
    c = read_sql("SELECT COUNT(*) n, COUNT(DISTINCT sk_id_curr) u FROM customer_360").iloc[0]
    a = read_sql("SELECT COUNT(*) n FROM raw_application").iloc[0]
    assert c.n == c.u == a.n                      # one row per applicant, none lost in joins
    d = read_sql("SELECT AVG(default_flag::float) dr FROM customer_360 WHERE population='portfolio'").iloc[0]
    assert 0.05 < d.dr < 0.11                     # Home Credit base default rate ~8%
    v = read_sql("SELECT COUNT(DISTINCT vintage) n FROM customer_360 WHERE population='portfolio'").iloc[0]
    assert v.n == 8


@requires_db
def test_leakage_guard_no_future_behaviour():
    from common.db import read_sql
    # Customer 360 aggregates must only use behaviour before the application date.
    r = read_sql("""SELECT MAX(inst_count) mx FROM customer_360""").iloc[0]
    assert r.mx > 0
    sql = open("sql/customer_360.sql").read()
    for guard in ("days_instalment < 0", "months_balance < 0", "days_credit < 0", "days_decision < 0"):
        assert guard in sql
