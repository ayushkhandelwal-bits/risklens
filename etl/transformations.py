"""
Step 3b — Transformations into the RiskLens raw layer.

* application_train + application_test are unified into one `raw_application`
  table with a `population` column:
      'portfolio' -> booked loans with an observed repayment outcome (TARGET)
      'intake'    -> recent applications without an outcome yet (used for
                     drift monitoring and scoring new customers)
* A display customer id ('C' + SK_ID_CURR) is added.

Business segmentation (vintage, region, income band, ...) is derived in SQL
(sql/customer_360.sql) so the logic is visible in the data layer.
"""
from __future__ import annotations

import pandas as pd

TABLE_MAP = {
    "bureau": "raw_bureau",
    "bureau_balance": "raw_bureau_balance",
    "previous_application": "raw_previous_application",
    "installments": "raw_installments",
    "pos_cash": "raw_pos_cash",
    "credit_card": "raw_credit_card",
}


def transform(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    train = frames["application_train"].copy()
    test = frames["application_test"].copy()
    train["population"] = "portfolio"
    test["population"] = "intake"
    test["target"] = pd.NA
    app = pd.concat([train, test], ignore_index=True)
    app.insert(0, "customer_id", "C" + app["sk_id_curr"].astype(str))
    app["target"] = app["target"].astype("Int64")

    out = {"raw_application": app}
    for src, tgt in TABLE_MAP.items():
        out[tgt] = frames[src]
    return out
