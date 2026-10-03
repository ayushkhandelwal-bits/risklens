"""
Risk engine — score every customer with the registered models.

Writes:
  risk_scores               PD (champion + challenger), risk score, tier, LGD, EAD, Expected Loss
  customer_explanations     top-5 risk-increasing and top-5 risk-decreasing SHAP drivers per customer
  global_feature_importance mean |SHAP| per population

Expected Loss = PD x LGD x EAD
  PD   model prediction (champion XGBoost)
  LGD  ASSUMPTION by product (common/config.py LGD_ASSUMPTIONS) — Home Credit has no recovery data
  EAD  DERIVED: cash loans -> loan amount; revolving -> credit limit x CCF assumption
"""
from __future__ import annotations

import io
import logging
import time

import numpy as np
import pandas as pd

from common.config import CCF_REVOLVING, LGD_ASSUMPTIONS, MODEL_VERSION, pd_tier
from common.db import execute, get_engine
from ml import train_logistic
from ml.explainability import bulk_contributions, clear_cache, load_champion, load_metadata
from ml.feature_engineering import build_features, load_customer_frame

log = logging.getLogger("risklens.ml")


def risk_score_from_pd(pds: np.ndarray, dev_percentiles: list[float]) -> np.ndarray:
    """0-100: percentile of the PD within the development population (100 = riskiest)."""
    return np.clip(np.searchsorted(np.asarray(dev_percentiles), pds, side="right") - 1, 0, 100).astype(int)


def expected_loss_frame(c360: pd.DataFrame, pds: np.ndarray) -> pd.DataFrame:
    lgd = c360["product"].map(LGD_ASSUMPTIONS).fillna(max(LGD_ASSUMPTIONS.values())).values
    ead = np.where(c360["product"].values == "Revolving loans", c360["amt_credit"].values * CCF_REVOLVING,
                   c360["amt_credit"].values)
    return pd.DataFrame({"lgd": lgd, "ead": ead, "expected_loss": pds * lgd * ead})


def _copy(table: str, df: pd.DataFrame) -> None:
    raw = get_engine().raw_connection()
    try:
        with raw.cursor() as cur:
            cur.execute(f"TRUNCATE {table}")
            buf = io.StringIO()
            df.to_csv(buf, index=False, header=False, na_rep="\\N")
            buf.seek(0)
            cols = ", ".join(df.columns)
            with cur.copy(f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, NULL '\\N')") as cp:
                cp.write(buf.read())
        raw.commit()
    finally:
        raw.close()


def main() -> None:
    t0 = time.time()
    clear_cache()
    meta = load_metadata()
    booster = load_champion()
    import joblib
    from common.config import MODEL_PATH
    lr = joblib.load(MODEL_PATH / meta["challenger"]["path"])

    c360 = load_customer_frame()
    X = build_features(c360)
    import xgboost as xgb
    pd_champ = booster.predict(xgb.DMatrix(X, feature_names=list(X.columns)))
    pd_chall = train_logistic.predict(lr, X)

    split_of = {}
    for name, ids in meta["splits"].items():
        for i in ids:
            split_of[i] = name
    el = expected_loss_frame(c360, pd_champ)
    scores = pd.DataFrame({
        "sk_id_curr": c360["sk_id_curr"].values,
        "customer_id": c360["customer_id"].values,
        "model_name": meta["champion"]["name"],
        "model_version": meta["model_version"],
        "pd": pd_champ,
        "pd_challenger": pd_chall,
        "risk_score": risk_score_from_pd(pd_champ, meta["dev_pd_percentiles"]),
        "risk_tier": [pd_tier(p) for p in pd_champ],
        "lgd": el.lgd.values, "ead": el.ead.values, "expected_loss": el.expected_loss.values,
        "dataset_split": [split_of.get(int(i), "intake") for i in c360["sk_id_curr"].values],
    })
    _copy("risk_scores", scores)
    log.info("risk_scores: %s rows", len(scores))

    # ---------------------------------------------------------------- SHAP drivers
    sv, base = bulk_contributions(X)
    feats = np.array(X.columns)
    order = np.argsort(-sv, axis=1)
    rows = []
    vals = X.values
    for i, sk in enumerate(X.index):
        pos = [j for j in order[i, :5] if sv[i, j] > 0]
        neg = [j for j in order[i, ::-1][:5] if sv[i, j] < 0]
        for r, j in enumerate(pos, start=1):
            rows.append((sk, r, feats[j], vals[i, j], sv[i, j], "increases_risk"))
        for r, j in enumerate(neg, start=6):
            rows.append((sk, r, feats[j], vals[i, j], sv[i, j], "decreases_risk"))
    expl = pd.DataFrame(rows, columns=["sk_id_curr", "rank", "feature", "feature_value", "shap_value", "direction"])
    expl["feature_value"] = expl["feature_value"].astype(float)
    _copy("customer_explanations", expl)
    log.info("customer_explanations: %s rows", len(expl))

    imp = []
    for pop in ("portfolio", "intake"):
        m = (c360["population"] == pop).values
        for j, f in enumerate(feats):
            imp.append((meta["champion"]["name"], MODEL_VERSION, pop, f, float(np.abs(sv[m, j]).mean()),
                        float(sv[m, j].mean())))
    _copy("global_feature_importance", pd.DataFrame(imp, columns=["model_name", "model_version", "population",
                                                                  "feature", "mean_abs_shap", "mean_shap"]))

    from backend.services.audit_service import log_event
    log_event("risk_scoring", tool="ml.scoring", user="system",
              parameters={"model": meta["champion"]["name"], "version": meta["model_version"], "customers": len(scores)},
              result_summary=f"scored {len(scores):,} customers; mean PD {pd_champ.mean():.4f}; "
                             f"total EL {scores.expected_loss.sum():,.0f}")
    execute("ANALYZE risk_scores; ANALYZE customer_explanations;")
    log.info("scoring finished in %.1fs", time.time() - t0)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
