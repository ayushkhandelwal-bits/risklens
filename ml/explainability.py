"""
SHAP explainability for the champion model.

* explain_customer(): on-demand explanation with shap.TreeExplainer — used by the
  API, the Investigation Center and the AI analyst's explain_prediction tool.
* bulk_contributions(): exact TreeSHAP for the whole book via XGBoost's native
  pred_contribs (same algorithm, vectorised) — used to store top drivers.

SHAP values are additive in log-odds space:
    logit(PD) = base_value + sum(shap_values)
which we verify for every explanation (`reconstruction_error`).
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from common.config import MODEL_PATH
from ml.feature_engineering import FEATURES, build_features, format_value, label, load_customer_frame


class ModelUnavailable(FileNotFoundError):
    pass


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    p = MODEL_PATH / "model_metadata.json"
    if not p.exists():
        raise ModelUnavailable(f"{p} not found — run `python -m ml.train`")
    return json.loads(p.read_text())


@lru_cache(maxsize=1)
def load_champion() -> xgb.Booster:
    meta = load_metadata()
    p = MODEL_PATH / meta["champion"]["path"]
    if not p.exists():
        raise ModelUnavailable(f"{p} not found — run `python -m ml.train`")
    b = xgb.Booster()
    b.load_model(str(p))
    return b


@lru_cache(maxsize=1)
def tree_explainer() -> shap.TreeExplainer:
    return shap.TreeExplainer(load_champion())


def clear_cache() -> None:
    load_metadata.cache_clear()
    load_champion.cache_clear()
    tree_explainer.cache_clear()


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def describe_driver(feature: str, value, shap_value: float) -> str:
    """Human sentence for one driver, e.g. 'High card utilisation (last 3m) (92.0%) increases risk'."""
    direction = "increases" if shap_value > 0 else "reduces"
    return f"{label(feature)} = {format_value(feature, value)} {direction} risk"


def explain_frame(X: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Explain one or more rows of the model feature matrix."""
    expl = tree_explainer()
    sv = expl.shap_values(X)
    base = float(np.atleast_1d(expl.expected_value)[0])
    pds = _sigmoid(base + sv.sum(axis=1))
    model_pd = load_champion().predict(xgb.DMatrix(X, feature_names=list(X.columns)))
    out = []
    for i in range(len(X)):
        contrib = pd.Series(sv[i], index=X.columns)
        vals = X.iloc[i]
        items = [{"feature": f, "label": label(f), "group": FEATURES[f][1],
                  "value": None if pd.isna(vals[f]) else float(vals[f]),
                  "display_value": format_value(f, vals[f]), "shap_value": float(contrib[f]),
                  "description": describe_driver(f, vals[f], float(contrib[f]))} for f in X.columns]
        pos = sorted([d for d in items if d["shap_value"] > 0], key=lambda d: -d["shap_value"])[:top_n]
        neg = sorted([d for d in items if d["shap_value"] < 0], key=lambda d: d["shap_value"])[:top_n]
        groups = (pd.DataFrame(items).groupby("group").shap_value.sum().sort_values(ascending=False))
        out.append({
            "pd": float(model_pd[i]),
            "base_value_logodds": base, "base_pd": float(_sigmoid(base)),
            "top_risk_increasing": pos, "top_risk_decreasing": neg,
            "group_contributions": [{"group": g, "shap_value": float(v)} for g, v in groups.items()],
            "reconstruction_error": float(abs(pds[i] - model_pd[i])),
            "all": items,
        })
    return out


def explain_customer(sk_id_curr: int, top_n: int = 5) -> dict:
    from common.db import read_sql
    c = read_sql("SELECT * FROM customer_360 WHERE sk_id_curr = :sk", {"sk": int(sk_id_curr)})
    if c.empty:
        raise LookupError(f"customer {sk_id_curr} not found")
    X = build_features(c)
    res = explain_frame(X, top_n)[0]
    meta = load_metadata()
    res.update({"model_name": meta["champion"]["name"], "model_version": meta["model_version"],
                "method": "SHAP TreeExplainer (exact, log-odds contributions) on the champion XGBoost model"})
    return res


def bulk_contributions(X: pd.DataFrame) -> tuple[np.ndarray, float]:
    contrib = load_champion().predict(xgb.DMatrix(X, feature_names=list(X.columns)), pred_contribs=True)
    return contrib[:, :-1], float(contrib[0, -1])


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    X = build_features(load_customer_frame("portfolio").head(200))
    sv, base = bulk_contributions(X)
    ref = tree_explainer().shap_values(X)
    print("max |native - shap.TreeExplainer| =", float(np.abs(sv - ref).max()))
