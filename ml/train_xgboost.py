"""Champion model: gradient-boosted trees (XGBoost).

Why XGBoost: captures non-linear effects and interactions (e.g. utilisation x
payment behaviour), handles missing values natively (bureau/card history is
absent for many customers), and is exactly explainable with TreeSHAP.
No class re-weighting is used so predicted probabilities stay calibrated PDs.
Hyper-parameters chosen on the VALIDATION split (small grid over depth, min_child_weight,
colsample); the least-overfit configuration within 0.001 AUC of the best was kept."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 3,
    "learning_rate": 0.02,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "min_child_weight": 60,
    "reg_lambda": 5.0,
    "tree_method": "hist",
    "seed": 42,
}
MAX_ROUNDS = 4000
EARLY_STOP = 200


def train(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series) -> xgb.Booster:
    dtr = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
    dva = xgb.DMatrix(X_valid, label=y_valid, feature_names=list(X_valid.columns))
    full = xgb.train(PARAMS, dtr, num_boost_round=MAX_ROUNDS, evals=[(dva, "valid")],
                     early_stopping_rounds=EARLY_STOP, verbose_eval=False)
    best = full.best_iteration
    # Keep exactly the trees used for prediction so PD and SHAP explain the same model.
    model = full[: best + 1]
    model.set_attr(best_iteration=str(best))
    return model


def predict(model: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    d = xgb.DMatrix(X, feature_names=list(X.columns))
    return model.predict(d)


def shap_contributions(model: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Exact TreeSHAP values (log-odds), last column = base value."""
    d = xgb.DMatrix(X, feature_names=list(X.columns))
    return model.predict(d, pred_contribs=True)
