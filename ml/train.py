"""
Train, evaluate and register the champion (XGBoost) and challenger (Logistic
Regression) PD models, then score the whole book.

    python -m ml.train

Methodology
  * Development data: the booked portfolio (application_train sample) — the only
    population with an observed outcome.
  * Split 60 / 20 / 20 (train / validation / test), stratified on the outcome,
    fixed seed. Validation is used for early stopping and threshold selection;
    the TEST split is touched once, for the reported metrics.
  * Features are as-of application (see ml/feature_engineering.py and the
    leakage guards in sql/customer_360.sql).
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common.config import MODEL_PATH, MODEL_VERSION, RANDOM_SEED
from common.db import execute
from ml import train_logistic, train_xgboost
from ml.evaluation import best_f1_threshold, evaluate
from ml.feature_engineering import FEATURE_NAMES, build_features, load_customer_frame

log = logging.getLogger("risklens.ml")


def split(c360: pd.DataFrame):
    ids = c360["sk_id_curr"].values
    y = c360["default_flag"].astype(int).values
    tr, tmp, ytr, ytmp = train_test_split(ids, y, test_size=0.4, stratify=y, random_state=RANDOM_SEED)
    va, te, _, _ = train_test_split(tmp, ytmp, test_size=0.5, stratify=ytmp, random_state=RANDOM_SEED)
    return {"train": tr, "valid": va, "test": te}


def register(name, role, algorithm, metrics, curves, params, n_train, n_test, artifact) -> None:
    execute("DELETE FROM model_registry WHERE model_name = :n AND model_version = :v", {"n": name, "v": MODEL_VERSION})
    execute("UPDATE model_registry SET is_active = FALSE WHERE role = :r", {"r": role})
    execute("""INSERT INTO model_registry (model_name, model_version, role, algorithm, training_date, features,
                   hyperparameters, metrics, curves, train_rows, test_rows, artifact_path, is_active)
               VALUES (:n, :v, :r, :a, now(), CAST(:f AS JSONB), CAST(:h AS JSONB), CAST(:m AS JSONB),
                       CAST(:c AS JSONB), :tr, :te, :p, TRUE)""",
            {"n": name, "v": MODEL_VERSION, "r": role, "a": algorithm, "f": json.dumps(FEATURE_NAMES),
             "h": json.dumps(params), "m": json.dumps(metrics), "c": json.dumps(curves), "tr": n_train,
             "te": n_test, "p": str(artifact)})


def main() -> dict:
    t0 = time.time()
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    c360 = load_customer_frame("portfolio")
    X = build_features(c360)
    y = pd.Series(c360["default_flag"].astype(int).values, index=X.index)
    parts = split(c360)
    Xtr, Xva, Xte = X.loc[parts["train"]], X.loc[parts["valid"]], X.loc[parts["test"]]
    ytr, yva, yte = y.loc[parts["train"]], y.loc[parts["valid"]], y.loc[parts["test"]]
    log.info("split: train %s / valid %s / test %s (default rate %.4f / %.4f / %.4f)",
             len(Xtr), len(Xva), len(Xte), ytr.mean(), yva.mean(), yte.mean())

    results = {}
    # ---------------------------------------------------------------- champion
    xgb_model = train_xgboost.train(Xtr, ytr, Xva, yva)
    thr = best_f1_threshold(yva.values, train_xgboost.predict(xgb_model, Xva))
    m_xgb, c_xgb = evaluate(yte.values, train_xgboost.predict(xgb_model, Xte), thr)
    m_xgb["best_iteration"] = int(xgb_model.attr("best_iteration"))
    m_xgb["train_auc"] = evaluate(ytr.values, train_xgboost.predict(xgb_model, Xtr), thr)[0]["auc"]
    xgb_path = MODEL_PATH / f"xgboost_pd_v{MODEL_VERSION}.json"
    xgb_model.save_model(str(xgb_path))
    results["xgboost"] = m_xgb
    log.info("XGBoost  test AUC %.4f Gini %.4f KS %.4f Brier %.5f (best iter %s)",
             m_xgb["auc"], m_xgb["gini"], m_xgb["ks"], m_xgb["brier"], m_xgb["best_iteration"])

    # ---------------------------------------------------------------- challenger
    lr_model = train_logistic.train(Xtr, ytr)
    thr_lr = best_f1_threshold(yva.values, train_logistic.predict(lr_model, Xva))
    m_lr, c_lr = evaluate(yte.values, train_logistic.predict(lr_model, Xte), thr_lr)
    m_lr["train_auc"] = evaluate(ytr.values, train_logistic.predict(lr_model, Xtr), thr_lr)[0]["auc"]
    lr_path = MODEL_PATH / f"logistic_pd_v{MODEL_VERSION}.joblib"
    joblib.dump(lr_model, lr_path)
    results["logistic"] = m_lr
    log.info("Logistic test AUC %.4f Gini %.4f KS %.4f Brier %.5f", m_lr["auc"], m_lr["gini"], m_lr["ks"], m_lr["brier"])

    # ---------------------------------------------------------------- metadata
    dev_pd = np.sort(train_xgboost.predict(xgb_model, Xtr))
    meta = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURE_NAMES,
        "champion": {"name": "XGBoost", "path": xgb_path.name, "threshold": thr},
        "challenger": {"name": "LogisticRegression", "path": lr_path.name, "threshold": thr_lr},
        "splits": {k: [int(i) for i in v] for k, v in parts.items()},
        # development PD distribution (percentiles) -> risk score 0-100
        "dev_pd_percentiles": [float(np.quantile(dev_pd, q / 100)) for q in range(101)],
        "metrics": results,
    }
    (MODEL_PATH / "model_metadata.json").write_text(json.dumps(meta))
    register("XGBoost", "champion", "xgboost.Booster (hist, depth 3)", m_xgb, c_xgb,
             train_xgboost.PARAMS | {"best_iteration": m_xgb["best_iteration"]}, len(Xtr), len(Xte), xgb_path)
    register("LogisticRegression", "challenger", "winsorise + median impute + L2 logistic", m_lr, c_lr,
             train_logistic.PARAMS, len(Xtr), len(Xte), lr_path)

    from backend.services.audit_service import log_event
    log_event("model_registered", tool="ml.train", user="system",
              parameters={"version": MODEL_VERSION, "features": len(FEATURE_NAMES)},
              result_summary=f"XGBoost test AUC {m_xgb['auc']:.4f}; Logistic test AUC {m_lr['auc']:.4f}")
    log.info("training finished in %.1fs", time.time() - t0)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    import warnings
    warnings.filterwarnings("ignore")
    res = main()
    print(json.dumps({k: {m: round(v, 4) for m, v in r.items() if isinstance(v, float)} for k, r in res.items()}, indent=1))
    from ml.scoring import main as score
    score()
