"""
Step 2 — Data quality checks.

Two layers:
  * Source-level integrity (full files, key columns only): duplicate keys,
    invalid IDs and orphaned relationships between source systems.
  * Sample-level content checks: null percentages, impossible values,
    sentinel codes and type consistency.

Every check produces a row with a status (PASS / WARN / FAIL) that is loaded
into PostgreSQL (`data_quality_checks`) and shown on the dashboard.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from etl.ingestion import resolve_path

log = logging.getLogger("risklens.etl")


@dataclass
class Check:
    table_name: str
    check_name: str
    category: str
    metric_value: float
    threshold: str
    status: str
    detail: str


def _status(value: float, warn: float, fail: float) -> str:
    if value >= fail:
        return "FAIL"
    if value >= warn:
        return "WARN"
    return "PASS"


# ---------------------------------------------------------------------------
# Source-level integrity (full files)
# ---------------------------------------------------------------------------
def checks_from_profile(prof: dict) -> list[Check]:
    """Build source-integrity checks from source_profile.json (computed on the FULL
    files by scripts/extract_sample.py on the machine that holds the Kaggle data)."""
    rc = prof["row_counts_full"]
    c: list[Check] = []
    d = prof["duplicate_applicant_ids"]
    c.append(Check("source.application", "duplicate_applicant_ids", "duplicates", d, "0", "PASS" if d == 0 else "FAIL",
                   f"{rc['application_train'] + rc['application_test']:,} applicant ids in full files"))
    v = prof["bureau_without_applicant_pct"]
    c.append(Check("source.bureau", "bureau_without_applicant_pct", "referential_integrity", round(v, 3),
                   "warn>=1%, fail>=10%", _status(v, 1, 10), f"{rc['bureau']:,} bureau records (full file)"))
    d = prof["duplicate_bureau_ids"]
    c.append(Check("source.bureau", "duplicate_bureau_ids", "duplicates", d, "0", "PASS" if d == 0 else "FAIL",
                   "duplicated SK_ID_BUREAU (full file)"))
    v = prof["bureau_balance_without_bureau_pct"]
    c.append(Check("source.bureau_balance", "bureau_balance_without_bureau_pct", "referential_integrity", round(v, 3),
                   "warn>=1%, fail>=50%", _status(v, 1, 50),
                   f"{prof['bureau_balance_accounts']:,} accounts with monthly history ({rc['bureau_balance']:,} rows); "
                   "share missing from bureau.csv"))
    d = prof["duplicate_prev_ids"]
    c.append(Check("source.previous_application", "duplicate_prev_ids", "duplicates", d, "0",
                   "PASS" if d == 0 else "FAIL", f"{rc['previous_application']:,} previous applications (full file)"))
    for k in ("installments", "pos_cash", "credit_card"):
        v = prof[f"{k}_without_previous_application_pct"]
        c.append(Check(f"source.{k}", "records_without_previous_application_pct", "referential_integrity", round(v, 3),
                       "warn>=1%, fail>=50%", _status(v, 1, 50),
                       f"{prof[f'{k}_distinct_prev']:,} distinct SK_ID_PREV ({rc[k]:,} rows); share not in previous_application"))
    return c


def source_integrity_checks(raw_dir: Path | None = None) -> list[Check]:
    from common.config import RAW_DATA_DIR
    base = Path(raw_dir or RAW_DATA_DIR)
    profile = next((p for p in (base / "source_profile.json", base / "raw" / "source_profile.json") if p.exists()), None)
    if profile:
        import json
        return checks_from_profile(json.loads(profile.read_text()))
    checks: list[Check] = []
    app_ids = pd.concat([
        pd.read_csv(resolve_path("application_train", raw_dir), usecols=["SK_ID_CURR"]).SK_ID_CURR,
        pd.read_csv(resolve_path("application_test", raw_dir), usecols=["SK_ID_CURR"]).SK_ID_CURR,
    ])
    dup = int(app_ids.duplicated().sum())
    checks.append(Check("source.application", "duplicate_applicant_ids", "duplicates", dup, "0",
                        "PASS" if dup == 0 else "FAIL", f"{len(app_ids):,} applicant ids, {dup} duplicates"))
    app_set = set(app_ids.values)

    bureau = pd.read_csv(resolve_path("bureau", raw_dir), usecols=["SK_ID_CURR", "SK_ID_BUREAU"])
    orphan = float((~bureau.SK_ID_CURR.isin(app_set)).mean() * 100)
    checks.append(Check("source.bureau", "bureau_without_applicant_pct", "referential_integrity", round(orphan, 3),
                        "warn>=1%, fail>=10%", _status(orphan, 1, 10),
                        f"{len(bureau):,} bureau records; share whose SK_ID_CURR is not in application files"))
    dupb = int(bureau.SK_ID_BUREAU.duplicated().sum())
    checks.append(Check("source.bureau", "duplicate_bureau_ids", "duplicates", dupb, "0",
                        "PASS" if dupb == 0 else "FAIL", f"{dupb} duplicated SK_ID_BUREAU"))
    bureau_set = set(bureau.SK_ID_BUREAU.values)
    del bureau

    bb = pd.read_csv(resolve_path("bureau_balance", raw_dir), usecols=["SK_ID_BUREAU"]).SK_ID_BUREAU.unique()
    orphan_bb = float((~np.isin(bb, list(bureau_set))).mean() * 100)
    checks.append(Check("source.bureau_balance", "bureau_balance_without_bureau_pct", "referential_integrity",
                        round(orphan_bb, 3), "warn>=1%, fail>=50%", _status(orphan_bb, 1, 50),
                        f"{len(bb):,} distinct bureau accounts with monthly history; share missing from bureau.csv"))
    del bb

    prev = pd.read_csv(resolve_path("previous_application", raw_dir), usecols=["SK_ID_PREV", "SK_ID_CURR"])
    dupp = int(prev.SK_ID_PREV.duplicated().sum())
    checks.append(Check("source.previous_application", "duplicate_prev_ids", "duplicates", dupp, "0",
                        "PASS" if dupp == 0 else "FAIL", f"{len(prev):,} previous applications"))
    prev_set = set(prev.SK_ID_PREV.values)
    del prev

    for name in ("installments", "pos_cash", "credit_card"):
        ids = pd.read_csv(resolve_path(name, raw_dir), usecols=["SK_ID_PREV"]).SK_ID_PREV.unique()
        pct = float((~np.isin(ids, list(prev_set))).mean() * 100)
        checks.append(Check(f"source.{name}", "records_without_previous_application_pct", "referential_integrity",
                            round(pct, 3), "warn>=1%, fail>=50%", _status(pct, 1, 50),
                            f"{len(ids):,} distinct SK_ID_PREV; share not found in previous_application.csv"))
    return checks


# ---------------------------------------------------------------------------
# Sample-level content checks
# ---------------------------------------------------------------------------
PRIMARY_KEYS = {
    "application_train": ["SK_ID_CURR"],
    "application_test": ["SK_ID_CURR"],
    "bureau": ["SK_ID_BUREAU"],
    "bureau_balance": ["SK_ID_BUREAU", "MONTHS_BALANCE"],
    "previous_application": ["SK_ID_PREV"],
    "installments": ["SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER", "DAYS_ENTRY_PAYMENT", "AMT_PAYMENT"],
    "pos_cash": ["SK_ID_PREV", "MONTHS_BALANCE"],
    "credit_card": ["SK_ID_PREV", "MONTHS_BALANCE"],
}

# (table, column, rule description, predicate returning boolean Series of violations)
IMPOSSIBLE_VALUE_RULES = [
    ("application", "DAYS_BIRTH", "age outside 18-100 years",
     lambda s: (s > -18 * 365) | (s < -100 * 365)),
    ("application", "DAYS_EMPLOYED", "sentinel 365243 (≈1000 years employed)",
     lambda s: s == 365243),
    ("application", "AMT_INCOME_TOTAL", "income <= 0 or > 10M",
     lambda s: (s <= 0) | (s > 1e7)),
    ("application", "AMT_CREDIT", "credit amount <= 0",
     lambda s: s <= 0),
    ("application", "CODE_GENDER", "unknown code 'XNA'",
     lambda s: s == "XNA"),
    ("previous_application", "DAYS_FIRST_DRAWING", "sentinel 365243",
     lambda s: s == 365243),
    ("previous_application", "AMT_CREDIT", "negative credit amount",
     lambda s: s < 0),
    ("installments", "AMT_PAYMENT", "negative payment",
     lambda s: s < 0),
    ("credit_card", "AMT_CREDIT_LIMIT_ACTUAL", "negative credit limit",
     lambda s: s < 0),
    ("credit_card", "AMT_BALANCE", "balance > 3x credit limit",
     None),  # handled specially (two columns)
    ("bureau", "DAYS_CREDIT_ENDDATE", "end date > 100 years in future",
     lambda s: s > 36500),
]


def content_checks(frames: dict[str, pd.DataFrame]) -> list[Check]:
    checks: list[Check] = []
    for name, df in frames.items():
        # Completeness
        null_pct = float(df.isna().mean().mean() * 100)
        cols_over_50 = int((df.isna().mean() > 0.5).sum())
        checks.append(Check(name, "overall_null_pct", "completeness", round(null_pct, 2),
                            "warn>=20%, fail>=50%", _status(null_pct, 20, 50),
                            f"{df.shape[1]} columns, {cols_over_50} columns more than 50% null"))
        # Duplicates on the natural key
        pk = PRIMARY_KEYS.get(name)
        if pk:
            dup = int(df.duplicated(subset=pk).sum())
            pct = dup / max(len(df), 1) * 100
            checks.append(Check(name, "duplicate_key_rows", "duplicates", dup, "warn>0.1%, fail>=1%",
                                _status(pct, 0.1, 1), f"natural key {pk}; {pct:.3f}% of {len(df):,} rows"))
        # Invalid IDs
        for idcol in ("SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"):
            if idcol in df.columns:
                bad = int((df[idcol].isna() | (df[idcol] <= 0)).sum())
                checks.append(Check(name, f"invalid_{idcol.lower()}", "validity", bad, "0",
                                    "PASS" if bad == 0 else "FAIL", "null or non-positive identifiers"))
        # Type consistency: numeric columns that pandas parsed as object
        mixed = [c for c in df.columns if df[c].dtype == object and
                 pd.to_numeric(df[c].dropna().head(500), errors="coerce").notna().mean() > 0.9]
        checks.append(Check(name, "mixed_type_columns", "consistency", len(mixed), "0",
                            "PASS" if not mixed else "WARN", ", ".join(mixed[:5]) or "all columns typed consistently"))

    # Impossible / sentinel values
    app = pd.concat([frames["application_train"], frames["application_test"]], ignore_index=True)
    lookup = {"application": app, **frames}
    for table, col, desc, rule in IMPOSSIBLE_VALUE_RULES:
        df = lookup.get(table)
        if df is None or col not in df.columns:
            continue
        if rule is None:  # balance vs limit
            viol = (df["AMT_BALANCE"] > 3 * df["AMT_CREDIT_LIMIT_ACTUAL"]) & (df["AMT_CREDIT_LIMIT_ACTUAL"] > 0)
        else:
            viol = rule(df[col]).fillna(False)
        n = int(viol.sum())
        pct = n / max(len(df), 1) * 100
        known_code = "sentinel" in desc or "XNA" in desc
        # Documented source codes are expected and cleaned -> WARN (visible) rather than FAIL.
        status = ("WARN" if n else "PASS") if known_code else _status(pct, 0.5, 25)
        checks.append(Check(table, f"impossible_{col.lower()}", "sentinel_code" if known_code else "validity", n,
                            "documented code: cleaned" if known_code else "warn>0.5%, fail>=25%",
                            status,
                            f"{desc}: {n:,} rows ({pct:.2f}%) — {'cleaned to NULL' if 'sentinel' in desc or 'XNA' in desc else 'flagged'}"))

    # Referential integrity inside the sample
    app_ids = set(app.SK_ID_CURR)
    prev_ids = set(frames["previous_application"].SK_ID_PREV)
    for name in ("installments", "pos_cash", "credit_card"):
        df = frames[name]
        miss_app = int((~df.SK_ID_CURR.isin(app_ids)).sum())
        miss_prev = float((~df.SK_ID_PREV.isin(prev_ids)).mean() * 100)
        checks.append(Check(name, "missing_applicant_relationship", "referential_integrity", miss_app, "0",
                            "PASS" if miss_app == 0 else "FAIL", "rows whose SK_ID_CURR is not a sampled applicant"))
        checks.append(Check(name, "missing_previous_application_pct", "referential_integrity", round(miss_prev, 3),
                            "warn>=1%, fail>=50%", _status(miss_prev, 1, 50),
                            "share of rows whose SK_ID_PREV is absent from previous_application"))
    return checks


def run_checks(frames: dict[str, pd.DataFrame], raw_dir: Path | None = None, full_source: bool = True) -> pd.DataFrame:
    checks = source_integrity_checks(raw_dir) if full_source else []
    checks += content_checks(frames)
    df = pd.DataFrame([asdict(c) for c in checks])
    log.info("Data quality: %s PASS, %s WARN, %s FAIL",
             (df.status == "PASS").sum(), (df.status == "WARN").sum(), (df.status == "FAIL").sum())
    return df
