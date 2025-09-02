"""Run deterministic, synthetic churn experiments and persist their evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, recall_score

from .data import CATEGORICAL_FEATURES, FEATURES, TARGET, split_data, validate_frame
from .generate_smoke_data import generate_smoke_frame
from .modeling import candidate_models, classification_metrics, select_for_budget

RESULT_SCHEMA_VERSION = 1


def missing_diagnostics(frame: pd.DataFrame) -> dict[str, object]:
    """Summarize feature-level missingness after schema normalization."""
    missing = frame[FEATURES].isna()
    rows = len(frame)
    by_feature = {
        column: {
            "count": int(missing[column].sum()),
            "fraction": float(missing[column].mean()),
        }
        for column in FEATURES
    }
    return {
        "rows_with_missing": int(missing.any(axis=1).sum()),
        "row_fraction": float(missing.any(axis=1).mean()),
        "by_feature": by_feature,
        "rows": rows,
    }


def unseen_category_diagnostics(train: pd.DataFrame, evaluation: pd.DataFrame) -> dict[str, object]:
    """Count categorical values present outside the training vocabulary."""
    any_unseen = pd.Series(False, index=evaluation.index)
    by_feature: dict[str, dict[str, object]] = {}
    for column in CATEGORICAL_FEATURES:
        known = set(train[column].dropna().astype(str))
        values = evaluation[column].dropna().astype(str)
        mask = evaluation[column].notna() & ~evaluation[column].astype(str).isin(known)
        any_unseen |= mask
        by_feature[column] = {
            "values": sorted(set(values) - known),
            "count": int(mask.sum()),
            "fraction": float(mask.mean()),
        }
    return {
        "rows_with_unseen": int(any_unseen.sum()),
        "row_fraction": float(any_unseen.mean()),
        "by_feature": by_feature,
    }


def segment_diagnostics(
    frame: pd.DataFrame,
    scores: np.ndarray,
    budget_fraction: float,
    columns: tuple[str, ...] = ("Contract", "InternetService"),
) -> dict[str, object]:
    """Report operating quality for business-relevant customer segments."""
    labels = select_for_budget(scores, budget_fraction).astype(int)
    report: dict[str, object] = {}
    for column in columns:
        segments: dict[str, object] = {}
        for value in sorted(frame[column].astype(str).unique()):
            mask = frame[column].astype(str).eq(value).to_numpy()
            truth = frame.loc[mask, TARGET].to_numpy()
            segment_scores = np.asarray(scores)[mask]
            segments[value] = {
                "rows": int(mask.sum()),
                "churn_rate": float(truth.mean()),
                "mean_score": float(segment_scores.mean()),
                "selected_fraction": float(labels[mask].mean()),
                "recall": float(recall_score(truth, labels[mask], zero_division=0)),
                "error_rate": float((truth != labels[mask]).mean()),
                "pr_auc": (
                    float(average_precision_score(truth, segment_scores))
                    if len(np.unique(truth)) == 2
                    else None
                ),
            }
        report[column] = segments
    return report


def run_experiment(
    *, seed: int = 20250809, rows: int = 480, budget_fraction: float = 0.20
) -> dict[str, object]:
    """Train and evaluate one reproducible synthetic experiment."""
    if rows < 80:
        raise ValueError("rows must be at least 80")
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must be in (0, 1]")

    frame = validate_frame(generate_smoke_frame(rows=rows, seed=seed))
    train, validation, test = split_data(frame, seed=seed)
    model_results: dict[str, dict[str, object]] = {}
    score_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in ("dummy", "weighted_logistic_regression"):
        model = candidate_models(seed)[name]
        model.fit(train[FEATURES], train[TARGET])
        validation_scores = model.predict_proba(validation[FEATURES])[:, 1]
        test_scores = model.predict_proba(test[FEATURES])[:, 1]
        model_results[name] = {
            "validation": classification_metrics(
                validation[TARGET], validation_scores, budget_fraction
            ),
            "test": classification_metrics(test[TARGET], test_scores, budget_fraction),
        }
        score_cache[name] = (validation_scores, test_scores)
    baseline_test = model_results["dummy"]["test"]
    logistic_test = model_results["weighted_logistic_regression"]["test"]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "config": {
            "seed": seed,
            "rows": rows,
            "budget_fraction": budget_fraction,
        },
        "data": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "churn_rate": float(frame[TARGET].mean()),
        },
        "selected_model": "weighted_logistic_regression",
        "models": model_results,
        "delta_vs_baseline": {
            "test_pr_auc": float(logistic_test["pr_auc"]) - float(baseline_test["pr_auc"]),
            "test_recall": float(logistic_test["recall"]) - float(baseline_test["recall"]),
        },
        "diagnostics": {
            "missing": missing_diagnostics(frame),
            "unseen_categories": {
                "validation": unseen_category_diagnostics(train, validation),
                "test": unseen_category_diagnostics(train, test),
            },
            "segments": segment_diagnostics(
                test,
                score_cache["weighted_logistic_regression"][1],
                budget_fraction,
            ),
        },
    }


def write_result(result: dict[str, object], output: Path) -> None:
    """Write a stable JSON representation of one experiment."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20250809)
    parser.add_argument("--rows", type=int, default=480)
    parser.add_argument("--budget-fraction", type=float, default=0.20)
    args = parser.parse_args(argv)
    result = run_experiment(
        seed=args.seed,
        rows=args.rows,
        budget_fraction=args.budget_fraction,
    )
    write_result(result, args.output)
    print(f"experiment result written to {args.output}")


if __name__ == "__main__":
    main()
