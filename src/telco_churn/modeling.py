"""Churn model candidates and threshold metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def candidate_models(seed: int) -> dict[str, Pipeline]:
    estimators = {
        "dummy": DummyClassifier(strategy="prior"),
        "weighted_logistic_regression": LogisticRegression(
            max_iter=1_000, class_weight="balanced", random_state=seed
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=80,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }
    return {
        name: Pipeline([("preprocess", preprocessor()), ("model", estimator)])
        for name, estimator in estimators.items()
    }


def threshold_for_budget(scores: np.ndarray, budget_fraction: float) -> float:
    """Вернуть нижний score точной top-k очереди; ties разрешаются стабильным порядком."""
    selected = select_for_budget(scores, budget_fraction)
    return float(np.min(np.asarray(scores, dtype=float)[selected]))


def select_for_budget(scores: np.ndarray, budget_fraction: float) -> np.ndarray:
    """Выбрать ровно ceil(n * budget_fraction) клиентов с максимальным риском."""
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must be in (0, 1]")
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("scores must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("scores must contain only finite values")
    count = max(1, math.ceil(len(values) * budget_fraction))
    order = np.argsort(-values, kind="stable")
    selected = np.zeros(len(values), dtype=bool)
    selected[order[:count]] = True
    return selected


def classification_metrics(
    truth: pd.Series | np.ndarray, scores: np.ndarray, budget_fraction: float
) -> dict[str, object]:
    labels = select_for_budget(scores, budget_fraction).astype(int)
    threshold = threshold_for_budget(scores, budget_fraction)
    return {
        "pr_auc": float(average_precision_score(truth, scores)),
        "roc_auc": float(roc_auc_score(truth, scores)),
        "precision": float(precision_score(truth, labels, zero_division=0)),
        "recall": float(recall_score(truth, labels, zero_division=0)),
        "f1": float(f1_score(truth, labels, zero_division=0)),
        "selected_fraction": float(labels.mean()),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(truth, labels, labels=[0, 1]).tolist(),
    }
