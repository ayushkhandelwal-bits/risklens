"""
Step 1 — Data ingestion.

Treats each Home Credit file as a separate operational source system:

    application_train/test  -> Loan Origination System (LOS)
    bureau / bureau_balance -> Credit Bureau feed
    previous_application    -> Previous Loan System
    installments_payments   -> Payment System
    credit_card_balance     -> Credit Card System
    POS_CASH_balance        -> POS / Cash Loan System

The full dataset (~58M rows) is larger than a laptop demo needs, so we draw a
reproducible random sample of applicants and then pull *every* related record
for those applicants (chunked reads keep memory bounded). Referential
integrity is therefore preserved inside the sample.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import RANDOM_SEED, RAW_DATA_DIR, SAMPLE_INTAKE, SAMPLE_PORTFOLIO

log = logging.getLogger("risklens.etl")

SOURCE_FILES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "installments": "installments_payments.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "credit_card": "credit_card_balance.csv",
}

SOURCE_SYSTEMS = {
    "application_train": "Loan Origination System",
    "application_test": "Loan Origination System (recent intake)",
    "bureau": "Credit Bureau",
    "bureau_balance": "Credit Bureau (monthly status)",
    "previous_application": "Previous Loan System",
    "installments": "Payment System",
    "pos_cash": "POS / Cash Loan System",
    "credit_card": "Credit Card System",
}

CHUNK = 1_000_000


def resolve_path(name: str, raw_dir: Path | None = None) -> Path:
    """Find a source file as .csv, .csv.gz or .zip (large files may be compressed)."""
    raw_dir = Path(raw_dir or RAW_DATA_DIR)
    base = SOURCE_FILES[name]
    stem = base[:-4]
    for cand in (base, base + ".gz", stem + ".zip", stem + ".csv.zip"):
        for p in (raw_dir / cand, raw_dir / "raw" / cand, raw_dir / "fresh" / cand):
            if p.exists():
                return p
    raise FileNotFoundError(f"Source file for '{name}' not found in {raw_dir} (looked for {base}, .gz, .zip)")


def _read_filtered(path: Path, key: str, keep: set[int]) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, chunksize=CHUNK, low_memory=False):
        parts.append(chunk[chunk[key].isin(keep)])
    return pd.concat(parts, ignore_index=True)


def sample_applicants(raw_dir: Path | None = None,
                      n_portfolio: int = SAMPLE_PORTFOLIO,
                      n_intake: int = SAMPLE_INTAKE) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    train = pd.read_csv(resolve_path("application_train", raw_dir), low_memory=False)
    test = pd.read_csv(resolve_path("application_test", raw_dir), low_memory=False)
    log.info("application_train: %s rows, application_test: %s rows", len(train), len(test))
    if n_portfolio and n_portfolio < len(train):
        train = train.iloc[np.sort(rng.choice(len(train), n_portfolio, replace=False))]
    if n_intake and n_intake < len(test):
        test = test.iloc[np.sort(rng.choice(len(test), n_intake, replace=False))]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def ingest(raw_dir: Path | None = None,
           n_portfolio: int = SAMPLE_PORTFOLIO,
           n_intake: int = SAMPLE_INTAKE) -> dict[str, pd.DataFrame]:
    """Return a dict of source DataFrames restricted to the sampled applicants."""
    train, test = sample_applicants(raw_dir, n_portfolio, n_intake)
    ids = set(train.SK_ID_CURR) | set(test.SK_ID_CURR)
    out = {"application_train": train, "application_test": test}

    for name in ("bureau", "previous_application", "installments", "pos_cash", "credit_card"):
        out[name] = _read_filtered(resolve_path(name, raw_dir), "SK_ID_CURR", ids)
        log.info("%-22s %10s rows (sampled)", name, f"{len(out[name]):,}")

    bureau_ids = set(out["bureau"].SK_ID_BUREAU)
    out["bureau_balance"] = _read_filtered(resolve_path("bureau_balance", raw_dir), "SK_ID_BUREAU", bureau_ids)
    log.info("%-22s %10s rows (sampled)", "bureau_balance", f"{len(out['bureau_balance']):,}")
    return out
