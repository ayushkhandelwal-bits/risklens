"""
Feature engineering for the credit-risk models.

Features come from the SQL Customer 360 (aggregations over every source system)
plus a few model-ready encodings done here. Every feature has a business label
and a group so explanations read like an analyst wrote them.

Deliberately excluded:
  * gender, family status, children count as *protected / sensitive* attributes
    (fair-lending hygiene; also not needed for performance),
  * IDs, the target, and any post-application information (leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.db import read_sql

# name -> (business label, group, unit)  unit: pct | ratio | amount | count | years | days | score | flag | months
FEATURES: dict[str, tuple[str, str, str]] = {
    # Application / affordability
    "amt_income_total":        ("Declared income", "Application", "amount"),
    "amt_credit":              ("Loan amount", "Application", "amount"),
    "amt_annuity":             ("Instalment amount", "Application", "amount"),
    "credit_to_income":        ("Loan-to-income ratio", "Application", "ratio"),
    "annuity_to_income":       ("Instalment-to-income ratio", "Application", "ratio"),
    "credit_to_goods":         ("Loan-to-goods-price ratio", "Application", "ratio"),
    "loan_term_months":        ("Implied loan term", "Application", "months"),
    "is_revolving":            ("Revolving product", "Application", "flag"),
    "age_years":               ("Applicant age", "Applicant", "years"),
    "employment_years":        ("Years in current employment", "Applicant", "years"),
    "not_employed_flag":       ("Not in employment (pensioner/unemployed)", "Applicant", "flag"),
    "education_level":         ("Education level", "Applicant", "score"),
    "own_car":                 ("Owns a car", "Applicant", "flag"),
    "own_realty":              ("Owns real estate", "Applicant", "flag"),
    "region_rating":           ("Region risk rating (1 best – 3 worst)", "Applicant", "score"),
    "id_document_age_years":   ("Years since ID document issued", "Applicant", "years"),
    "days_since_phone_change": ("Days since phone number change", "Applicant", "days"),
    "address_mismatch_flags":  ("Address / work-location mismatches", "Applicant", "count"),
    "social_circle_defaults_30d": ("Defaults in social circle (30 DPD)", "Applicant", "count"),
    # External scores
    "ext_source_1":            ("External credit score 1", "External score", "score"),
    "ext_source_2":            ("External credit score 2", "External score", "score"),
    "ext_source_3":            ("External credit score 3", "External score", "score"),
    "ext_source_mean":         ("External credit score (average)", "External score", "score"),
    # Bureau
    "bureau_accounts":         ("Bureau credit accounts", "Credit bureau", "count"),
    "bureau_active":           ("Active bureau accounts", "Credit bureau", "count"),
    "bureau_delinquent":       ("Delinquent bureau accounts", "Credit bureau", "count"),
    "bureau_debt_to_credit":   ("Bureau debt-to-credit (utilisation)", "Credit bureau", "ratio"),
    "bureau_history_years":    ("Credit history length", "Credit bureau", "years"),
    "bureau_opened_12m":       ("Bureau accounts opened in last 12m", "Credit bureau", "count"),
    "bureau_dpd_months_12m":   ("Bureau months past due (last 12m)", "Credit bureau", "count"),
    "bureau_enquiries_3m":     ("Credit enquiries (last 3m)", "Credit bureau", "count"),
    "bureau_enquiries_12m":    ("Credit enquiries (last 12m)", "Credit bureau", "count"),
    "total_debt_to_income":    ("Total debt-to-income", "Exposure", "ratio"),
    # Previous applications
    "prev_app_count":          ("Previous applications", "Previous loans", "count"),
    "prev_refused":            ("Previously refused applications", "Previous loans", "count"),
    "prev_approval_rate":      ("Previous approval rate", "Previous loans", "pct"),
    "prev_apps_365d":          ("Applications in last 12m", "Previous loans", "count"),
    "prev_refused_365d":       ("Refusals in last 12m", "Previous loans", "count"),
    "days_since_last_application": ("Days since last application", "Previous loans", "days"),
    # Payment behaviour
    "inst_late_rate":          ("Share of instalments paid late", "Payment behaviour", "pct"),
    "inst_avg_days_late":      ("Average days late", "Payment behaviour", "days"),
    "inst_max_days_late":      ("Maximum days late", "Payment behaviour", "days"),
    "inst_payment_ratio":      ("Paid / due ratio", "Payment behaviour", "ratio"),
    "inst_late_rate_12m":      ("Late-payment rate (last 12m)", "Payment behaviour", "pct"),
    "inst_late_rate_change":   ("Change in late-payment rate (12m vs prior)", "Payment behaviour", "pct"),
    "inst_missed_12m":         ("Missed / under-paid instalments (12m)", "Payment behaviour", "count"),
    "pos_dpd_months_12m":      ("POS/cash months past due (12m)", "Payment behaviour", "count"),
    # Credit card
    "cc_util_avg":             ("Average card utilisation", "Credit card", "pct"),
    "cc_util_3m":              ("Card utilisation (last 3m)", "Credit card", "pct"),
    "cc_util_change":          ("Change in card utilisation", "Credit card", "pct"),
    "cc_balance_change":       ("Change in card balance", "Credit card", "amount"),
    "cc_dpd_months":           ("Card months past due", "Credit card", "count"),
}
FEATURE_NAMES = list(FEATURES)
EDUCATION = {"Lower secondary": 0, "Secondary / secondary special": 1, "Incomplete higher": 2,
             "Higher education": 3, "Academic degree": 4}


def label(feature: str) -> str:
    return FEATURES.get(feature, (feature,))[0]


def format_value(feature: str, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "missing"
    unit = FEATURES.get(feature, ("", "", ""))[2]
    if unit == "pct":
        return f"{v * 100:.1f}%"
    if unit == "ratio":
        return f"{v:.2f}"
    if unit == "amount":
        return f"{v:,.0f}"
    if unit == "flag":
        return "yes" if v >= 0.5 else "no"
    if unit in ("years",):
        return f"{v:.1f} yrs"
    if unit == "days":
        return f"{v:,.0f} days"
    if unit == "score":
        return f"{v:.2f}" if v < 10 else f"{v:.0f}"
    return f"{v:,.0f}"


def load_customer_frame(population: str | None = None) -> pd.DataFrame:
    where = "WHERE population = :p" if population else ""
    return read_sql(f"SELECT * FROM customer_360 {where} ORDER BY sk_id_curr",
                    {"p": population} if population else None)


def build_features(c360: pd.DataFrame) -> pd.DataFrame:
    df = c360.copy()
    df["is_revolving"] = (df["product"] == "Revolving loans").astype(int)
    df["education_level"] = df["education"].map(EDUCATION)
    df["own_car"] = (df["flag_own_car"] == "Y").astype(int)
    df["own_realty"] = (df["flag_own_realty"] == "Y").astype(int)
    X = df[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce").astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    X.index = df["sk_id_curr"].values
    return X
