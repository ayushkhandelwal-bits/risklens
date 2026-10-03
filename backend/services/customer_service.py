"""Customer 360 service — consolidated profile for one customer."""
from __future__ import annotations

import re

import pandas as pd

from common.db import read_sql


class CustomerNotFound(LookupError):
    pass


def normalise_id(customer_id: str) -> tuple[str, int]:
    """Accept 'C100002', 'c100002' or '100002'."""
    s = str(customer_id).strip().upper()
    m = re.fullmatch(r"C?(\d{1,9})", s)
    if not m:
        raise CustomerNotFound(f"'{customer_id}' is not a valid customer id (expected e.g. C100002)")
    return f"C{int(m.group(1))}", int(m.group(1))


def _rec(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


def get_profile(customer_id: str) -> dict:
    cid, sk = normalise_id(customer_id)
    c = read_sql("SELECT * FROM customer_360 WHERE sk_id_curr = :sk", {"sk": sk})
    if c.empty:
        raise CustomerNotFound(f"Customer {cid} not found")
    p = _rec(c)[0]
    risk = read_sql("""SELECT pd, pd_challenger, risk_score, risk_tier, lgd, ead, expected_loss, dataset_split,
                              model_name, model_version, scored_at::text AS scored_at
                       FROM risk_scores WHERE sk_id_curr = :sk""", {"sk": sk})
    ews = read_sql("SELECT ews_score, ews_band, n_triggers FROM early_warning_signals WHERE sk_id_curr = :sk", {"sk": sk})
    anom = read_sql("""SELECT anomaly_score, anomaly_status, isolation_pct, rule_score, n_rules
                       FROM anomaly_scores WHERE sk_id_curr = :sk""", {"sk": sk})

    sections = {
        "identity": {k: p[k] for k in ("customer_id", "population", "vintage", "product", "region",
                                       "income_band", "customer_segment", "credit_score_band")},
        "observed_outcome": {"default_flag": p["default_flag"]},
        "demographics": {k: p[k] for k in ("gender", "age_years", "education", "family_status", "occupation",
                                           "housing", "employment_years", "not_employed_flag", "cnt_children",
                                           "flag_own_car", "flag_own_realty")},
        "application": {k: p[k] for k in ("amt_income_total", "amt_credit", "amt_annuity", "amt_goods_price",
                                          "credit_to_income", "annuity_to_income", "credit_to_goods",
                                          "loan_term_months", "ext_source_1", "ext_source_2", "ext_source_3",
                                          "ext_source_mean")},
        "exposure": {k: p[k] for k in ("current_exposure", "total_exposure", "bureau_debt_total",
                                       "total_debt_to_income")},
        "payment_behaviour": {k: p[k] for k in ("inst_count", "inst_late_rate", "inst_avg_days_late",
                                                "inst_max_days_late", "inst_missed_count", "inst_payment_ratio",
                                                "inst_late_rate_12m", "inst_late_rate_prior", "inst_late_rate_change",
                                                "inst_missed_12m", "pos_dpd_months", "pos_max_dpd",
                                                "pos_dpd_months_12m")},
        "credit_utilisation": {k: p[k] for k in ("cc_util_avg", "cc_util_max", "cc_util_3m", "cc_util_prior",
                                                 "cc_util_change", "cc_balance_3m", "cc_balance_prior",
                                                 "cc_balance_change", "cc_dpd_months")},
        "previous_applications": {k: p[k] for k in ("prev_app_count", "prev_approved", "prev_refused",
                                                    "prev_approval_rate", "prev_apps_365d", "prev_apps_90d",
                                                    "prev_refused_365d", "prev_avg_credit",
                                                    "days_since_last_application")},
        "bureau": {k: p[k] for k in ("bureau_accounts", "bureau_active", "bureau_closed", "bureau_delinquent",
                                     "bureau_debt_to_credit", "bureau_max_overdue_amt", "bureau_history_years",
                                     "bureau_opened_12m", "bureau_opened_prior_12m", "bureau_dpd_months_12m",
                                     "bureau_dpd_months_prior", "bureau_enquiries_1m", "bureau_enquiries_3m",
                                     "bureau_enquiries_12m")},
        "behavioural_flags": {k: p[k] for k in ("id_document_age_years", "days_since_phone_change",
                                                "address_mismatch_flags", "social_circle_defaults_30d")},
        "risk": _rec(risk)[0] if not risk.empty else None,
        "early_warning": _rec(ews)[0] if not ews.empty else None,
        "anomaly": _rec(anom)[0] if not anom.empty else None,
    }
    return sections


def previous_applications(customer_id: str, limit: int = 20) -> list[dict]:
    _, sk = normalise_id(customer_id)
    df = read_sql("""SELECT sk_id_prev, name_contract_type AS product, name_contract_status AS status,
                            amt_application, amt_credit, -days_decision AS days_ago, code_reject_reason,
                            name_client_type, channel_type
                     FROM raw_previous_application WHERE sk_id_curr = :sk
                     ORDER BY days_decision DESC LIMIT :l""", {"sk": sk, "l": limit})
    return _rec(df)


def bureau_history(customer_id: str) -> list[dict]:
    _, sk = normalise_id(customer_id)
    df = read_sql("""SELECT credit_type, credit_active, -days_credit AS days_since_opened, amt_credit_sum,
                            amt_credit_sum_debt, amt_credit_sum_overdue, credit_day_overdue
                     FROM raw_bureau WHERE sk_id_curr = :sk ORDER BY days_credit DESC""", {"sk": sk})
    return _rec(df)


def payment_timeline(customer_id: str) -> list[dict]:
    """Monthly payment behaviour (months before application) from the Payment System."""
    _, sk = normalise_id(customer_id)
    df = read_sql("""
        WITH inst AS (
            SELECT sk_id_prev, num_instalment_number, num_instalment_version,
                   MAX(days_instalment) d_due, MAX(days_entry_payment) d_paid,
                   MAX(amt_instalment) due, SUM(amt_payment) paid
            FROM raw_installments WHERE sk_id_curr = :sk AND days_instalment < 0
            GROUP BY 1,2,3)
        SELECT FLOOR(d_due / 30.44)::int AS month,
               COUNT(*) AS instalments,
               SUM((d_paid > d_due)::int) AS late,
               MAX(GREATEST(d_paid - d_due, 0)) AS max_days_late,
               SUM(due) AS amount_due, SUM(paid) AS amount_paid
        FROM inst WHERE d_due >= -730 GROUP BY 1 ORDER BY 1""", {"sk": sk})
    return _rec(df)


def utilisation_timeline(customer_id: str) -> list[dict]:
    _, sk = normalise_id(customer_id)
    df = read_sql("""SELECT months_balance AS month, SUM(amt_balance) AS balance,
                            SUM(amt_credit_limit_actual) AS credit_limit,
                            SUM(amt_balance) / NULLIF(SUM(amt_credit_limit_actual),0) AS utilisation,
                            MAX(sk_dpd) AS dpd
                     FROM raw_credit_card WHERE sk_id_curr = :sk AND months_balance BETWEEN -24 AND -1
                     GROUP BY 1 ORDER BY 1""", {"sk": sk})
    return _rec(df)
