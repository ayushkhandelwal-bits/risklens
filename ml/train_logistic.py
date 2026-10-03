"""Challenger model: regularised logistic regression (transparent, scorecard-style baseline)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PARAMS = {"C": 0.05, "max_iter": 3000, "solver": "lbfgs"}


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each feature to its 1st–99th percentile (learned on training data only)."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def train(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipe = Pipeline([
        ("winsor", Winsorizer()),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(**PARAMS)),
    ])
    pipe.fit(X_train.values, y_train.values)
    return pipe


def predict(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X.values)[:, 1]
