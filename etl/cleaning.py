"""
Step 3a — Cleaning.

Only well-understood, documented fixes are applied. Each one is logged so the
cleaning itself is auditable:

* Column names -> snake_case lower (PostgreSQL friendly).
* DAYS_EMPLOYED == 365243 is a known Home Credit sentinel (pensioners /
  unemployed). Set to NULL and keep an explicit flag.
* CODE_GENDER == 'XNA' -> NULL.
* previous_application DAYS_* == 365243 sentinels -> NULL.
* Exact duplicate rows are dropped.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("risklens.etl")

SENTINEL = 365243


def _snake(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def clean(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    log_rows: list[dict] = []
    out = {}
    for name, df in frames.items():
        df = df.copy()
        before = len(df)
        df = df.drop_duplicates()
        if len(df) != before:
            log_rows.append({"table": name, "action": "drop_exact_duplicates", "rows_affected": before - len(df)})
        df = _snake(df)

        if name.startswith("application"):
            n = int((df["days_employed"] == SENTINEL).sum())
            df["days_employed_anomaly"] = (df["days_employed"] == SENTINEL).astype(int)
            df.loc[df["days_employed"] == SENTINEL, "days_employed"] = np.nan
            log_rows.append({"table": name, "action": "days_employed_365243_to_null", "rows_affected": n})
            n = int((df["code_gender"] == "XNA").sum())
            df.loc[df["code_gender"] == "XNA", "code_gender"] = np.nan
            log_rows.append({"table": name, "action": "code_gender_XNA_to_null", "rows_affected": n})

        if name == "previous_application":
            for c in ("days_first_drawing", "days_first_due", "days_last_due_1st_version",
                      "days_last_due", "days_termination"):
                n = int((df[c] == SENTINEL).sum())
                df.loc[df[c] == SENTINEL, c] = np.nan
                log_rows.append({"table": name, "action": f"{c}_365243_to_null", "rows_affected": n})

        out[name] = df
    for r in log_rows:
        log.info("clean %-22s %-38s %s", r["table"], r["action"], f"{r['rows_affected']:,}")
    return out, log_rows
