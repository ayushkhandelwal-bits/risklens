"""
RiskLens — local data extract (run this on the machine that holds the Kaggle files).

Reads the full Home Credit CSVs, runs source-level integrity checks on the FULL
files, then writes a reproducible, referentially-consistent sample of
applicants with ALL of their related records as gzipped CSVs.

Usage (Windows example):
    python extract_sample.py --src "C:\\Users\\Admin\\OneDrive\\Documents\\risklens_data" --out "C:\\Users\\Admin\\risklens-sample-data"

Only needs pandas + numpy. Accepts .csv, .csv.gz or .zip for every source file.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

FILES = {
    "application_train": "application_train",
    "application_test": "application_test",
    "bureau": "bureau",
    "bureau_balance": "bureau_balance",
    "previous_application": "previous_application",
    "installments": "installments_payments",
    "pos_cash": "POS_CASH_balance",
    "credit_card": "credit_card_balance",
}
CHUNK = 1_000_000


def find(src: Path, stem: str) -> Path:
    for name in (f"{stem}.csv", f"{stem}.csv.gz", f"{stem}.zip", f"{stem}.csv.zip"):
        for p in (src / name, src / "fresh" / name):
            if p.exists():
                return p
    raise FileNotFoundError(f"{stem} (.csv/.zip) not found in {src}")


def read_col(path: Path, cols):
    return pd.read_csv(path, usecols=cols)


def filtered(path: Path, key: str, keep) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, chunksize=CHUNK, low_memory=False):
        parts.append(chunk[chunk[key].isin(keep)])
    return pd.concat(parts, ignore_index=True)


def main(src: Path, out: Path, n_portfolio: int, n_intake: int, seed: int) -> None:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    p = {k: find(src, v) for k, v in FILES.items()}
    for k, v in p.items():
        print(f"  {k:22s} <- {v.name}")

    # ---------------- full-file source integrity profile ----------------
    print("Profiling full source files (integrity checks)...")
    prof: dict = {"row_counts_full": {}}
    train_ids = read_col(p["application_train"], ["SK_ID_CURR"]).SK_ID_CURR
    test_ids = read_col(p["application_test"], ["SK_ID_CURR"]).SK_ID_CURR
    app_ids = pd.concat([train_ids, test_ids])
    prof["row_counts_full"]["application_train"] = int(len(train_ids))
    prof["row_counts_full"]["application_test"] = int(len(test_ids))
    prof["duplicate_applicant_ids"] = int(app_ids.duplicated().sum())

    b = read_col(p["bureau"], ["SK_ID_CURR", "SK_ID_BUREAU"])
    prof["row_counts_full"]["bureau"] = int(len(b))
    prof["bureau_without_applicant_pct"] = float((~b.SK_ID_CURR.isin(app_ids)).mean() * 100)
    prof["duplicate_bureau_ids"] = int(b.SK_ID_BUREAU.duplicated().sum())
    bureau_set = b.SK_ID_BUREAU.values
    del b

    bb = read_col(p["bureau_balance"], ["SK_ID_BUREAU"]).SK_ID_BUREAU
    prof["row_counts_full"]["bureau_balance"] = int(len(bb))
    u = bb.unique()
    prof["bureau_balance_accounts"] = int(len(u))
    prof["bureau_balance_without_bureau_pct"] = float((~np.isin(u, bureau_set)).mean() * 100)
    del bb, u

    pr = read_col(p["previous_application"], ["SK_ID_PREV", "SK_ID_CURR"])
    prof["row_counts_full"]["previous_application"] = int(len(pr))
    prof["duplicate_prev_ids"] = int(pr.SK_ID_PREV.duplicated().sum())
    prev_set = pr.SK_ID_PREV.values
    del pr
    for k in ("installments", "pos_cash", "credit_card"):
        s = read_col(p[k], ["SK_ID_PREV"]).SK_ID_PREV
        prof["row_counts_full"][k] = int(len(s))
        u = s.unique()
        prof[f"{k}_distinct_prev"] = int(len(u))
        prof[f"{k}_without_previous_application_pct"] = float((~np.isin(u, prev_set)).mean() * 100)
        del s, u
    (out / "source_profile.json").write_text(json.dumps(prof, indent=2))
    print(json.dumps(prof, indent=2))

    # ---------------- sample applicants ----------------
    rng = np.random.default_rng(seed)
    train = pd.read_csv(p["application_train"], low_memory=False)
    test = pd.read_csv(p["application_test"], low_memory=False)
    train = train.iloc[np.sort(rng.choice(len(train), min(n_portfolio, len(train)), replace=False))]
    test = test.iloc[np.sort(rng.choice(len(test), min(n_intake, len(test)), replace=False))]
    ids = set(train.SK_ID_CURR) | set(test.SK_ID_CURR)

    def save(df: pd.DataFrame, name: str) -> None:
        f = out / f"{name}.csv.gz"
        df.to_csv(f, index=False, compression="gzip")
        print(f"  wrote {f.name:34s} {len(df):>10,} rows  {f.stat().st_size / 1e6:6.1f} MB")

    print("Writing sample extract...")
    save(train, "application_train")
    save(test, "application_test")
    bureau = filtered(p["bureau"], "SK_ID_CURR", ids)
    save(bureau, "bureau")
    save(filtered(p["bureau_balance"], "SK_ID_BUREAU", set(bureau.SK_ID_BUREAU)), "bureau_balance")
    for k in ("previous_application", "installments", "pos_cash", "credit_card"):
        save(filtered(p[k], "SK_ID_CURR", ids), FILES[k])
    (out / "README.md").write_text(
        f"RiskLens sample extract of the Kaggle Home Credit Default Risk dataset.\n"
        f"Seed {seed}; {len(train):,} portfolio applicants (application_train) and {len(test):,} intake applicants "
        f"(application_test) with all related records. Keep this repository PRIVATE (Kaggle data terms).\n")
    print(f"Done in {(time.time() - t0) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--portfolio", type=int, default=50000)
    ap.add_argument("--intake", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.src, a.out, a.portfolio, a.intake, a.seed)
