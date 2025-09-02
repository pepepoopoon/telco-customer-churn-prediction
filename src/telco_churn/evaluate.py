"""Evaluate churn predictions, segments, errors, and permutation importance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import recall_score

from .data import FEATURES, TARGET, load_data, split_data
from .modeling import classification_metrics, select_for_budget


def _segment_report(frame: pd.DataFrame, labels: np.ndarray, column: str) -> dict[str, object]:
    result = {}
    for value in sorted(frame[column].astype(str).unique()):
        mask = frame[column].astype(str).eq(value).to_numpy()
        truth = frame.loc[mask, TARGET]
        result[value] = {
            "rows": int(mask.sum()),
            "churn_rate": float(truth.mean()),
            "recall": float(recall_score(truth, labels[mask], zero_division=0)),
            "error_rate": float((truth.to_numpy() != labels[mask]).mean()),
        }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/test_metrics.json"))
    parser.add_argument("--errors", type=Path, default=Path("reports/test_errors.csv"))
    parser.add_argument(
        "--importance", type=Path, default=Path("reports/permutation_importance.csv")
    )
    args = parser.parse_args(argv)
    artifact = joblib.load(args.artifact)
    _, _, test = split_data(load_data(args.data), seed=int(artifact["seed"]))
    scores = artifact["model"].predict_proba(test[FEATURES])[:, 1]
    budget_fraction = float(artifact["budget_fraction"])
    labels = select_for_budget(scores, budget_fraction).astype(int)
    metrics = classification_metrics(test[TARGET], scores, budget_fraction)
    metrics["model_name"] = artifact["model_name"]
    metrics["segments"] = {
        "Contract": _segment_report(test, labels, "Contract"),
        "InternetService": _segment_report(test, labels, "InternetService"),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    identity = test["customerID"] if "customerID" in test else test.index.astype(str)
    errors = test[["Contract", "InternetService", "tenure", TARGET]].copy()
    errors.insert(0, "customerID", identity.to_numpy())
    errors["score"] = scores
    errors["prediction"] = labels
    errors["error_type"] = np.select(
        [(test[TARGET] == 0) & (labels == 1), (test[TARGET] == 1) & (labels == 0)],
        ["false_positive", "false_negative"],
        default="correct",
    )
    errors = errors[errors["error_type"] != "correct"].sort_values("score", ascending=False)
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.errors, index=False)

    importance = permutation_importance(
        artifact["model"],
        test[FEATURES],
        test[TARGET],
        scoring="average_precision",
        n_repeats=3,
        random_state=int(artifact["seed"]),
        n_jobs=1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    args.importance.parent.mkdir(parents=True, exist_ok=True)
    importance_frame.to_csv(args.importance, index=False)
    print(f"metrics, errors, and importance written under {args.metrics.parent}")


if __name__ == "__main__":
    main()
