"""Credit-risk model evaluation: discrimination, calibration and classification metrics.
Accuracy is intentionally not reported (8% default base rate makes it meaningless)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (brier_score_loss, f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)


def ks_statistic(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.nanargmax(f1[:-1])])


def calibration_table(y, p, bins: int = 10) -> list[dict]:
    q = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum():
            out.append({"bin": b + 1, "predicted": float(p[m].mean()), "observed": float(y[m].mean()),
                        "n": int(m.sum())})
    return out


def roc_points(y, p, n: int = 60) -> list[dict]:
    fpr, tpr, _ = roc_curve(y, p)
    keep = np.unique(np.linspace(0, len(fpr) - 1, n).astype(int))
    return [{"fpr": float(fpr[i]), "tpr": float(tpr[i])} for i in keep]


def evaluate(y, p, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    auc = roc_auc_score(y, p)
    pred = (p >= threshold).astype(int)
    cal = calibration_table(y, p)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "brier_baseline": float(brier_score_loss(y, np.full_like(p, y.mean()))),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "threshold": float(threshold),
        "flagged_share": float(pred.mean()),
        "mean_pd": float(p.mean()),
        "observed_default_rate": float(y.mean()),
        "calibration_ratio": float(p.mean() / y.mean()),
        "max_calibration_gap": float(max(abs(c["predicted"] - c["observed"]) for c in cal)),
        "n": int(len(y)),
    }, {"roc": roc_points(y, p), "calibration": cal}
