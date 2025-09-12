"""Run deterministic, synthetic churn experiments and persist their evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, recall_score
from sklearn.model_selection import train_test_split

from .data import CATEGORICAL_FEATURES, FEATURES, TARGET, split_data, validate_frame
from .generate_smoke_data import generate_smoke_frame
from .modeling import candidate_models, classification_metrics, select_for_budget

RESULT_SCHEMA_VERSION = 1
SCENARIOS = ("retention_budget", "learning_curve", "seed_stability")


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


def permutation_diagnostics(
    model: object,
    frame: pd.DataFrame,
    *,
    seed: int,
    repeats: int = 3,
) -> dict[str, object]:
    """Measure held-out feature importance with deterministic permutations."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    result = permutation_importance(
        model,
        frame[FEATURES],
        frame[TARGET],
        scoring="average_precision",
        n_repeats=repeats,
        random_state=seed,
        n_jobs=1,
    )
    features = sorted(
        (
            {
                "feature": feature,
                "importance_mean": float(mean),
                "importance_std": float(std),
            }
            for feature, mean, std in zip(
                FEATURES,
                result.importances_mean,
                result.importances_std,
                strict=True,
            )
        ),
        key=lambda item: (-item["importance_mean"], item["feature"]),
    )
    return {"scoring": "average_precision", "repeats": repeats, "features": features}


def subsample_train(frame: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    """Select a deterministic stratified fraction for a learning-curve point."""
    if not 0 < fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    if fraction == 1:
        return frame.copy()
    subset, _ = train_test_split(
        frame,
        train_size=fraction,
        random_state=seed,
        stratify=frame[TARGET],
    )
    return subset.reset_index(drop=True)


def scenario_result(
    *,
    scenario: str,
    seed: int,
    train: pd.DataFrame,
    available_train: pd.DataFrame,
    logistic_test: dict[str, object],
    budget_fraction: float,
    train_fraction: float,
) -> dict[str, object]:
    """Build the scenario-specific summary from shared model evidence."""
    if scenario == "learning_curve":
        return {
            "used_train_rows": len(train),
            "available_train_rows": len(available_train),
            "train_fraction": train_fraction,
            "test_pr_auc": float(logistic_test["pr_auc"]),
        }
    if scenario == "seed_stability":
        return {
            "seed": seed,
            "test_pr_auc": float(logistic_test["pr_auc"]),
            "test_roc_auc": float(logistic_test["roc_auc"]),
            "test_recall": float(logistic_test["recall"]),
        }
    return {
        "selected_customers": int(
            logistic_test["confusion_matrix"][0][1] + logistic_test["confusion_matrix"][1][1]
        ),
        "captured_churners": int(logistic_test["confusion_matrix"][1][1]),
        "retention_budget_fraction": budget_fraction,
    }


def run_experiment(
    *,
    seed: int = 20250809,
    rows: int = 480,
    budget_fraction: float = 0.20,
    scenario: str = "retention_budget",
    train_fraction: float = 1.0,
) -> dict[str, object]:
    """Train and evaluate one reproducible synthetic experiment."""
    if rows < 80:
        raise ValueError("rows must be at least 80")
    if not 0 < budget_fraction <= 1:
        raise ValueError("budget_fraction must be in (0, 1]")
    if scenario not in SCENARIOS:
        raise ValueError(f"unsupported scenario: {scenario}")
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction must be in (0, 1]")
    frame = validate_frame(generate_smoke_frame(rows=rows, seed=seed))
    available_train, validation, test = split_data(frame, seed=seed)
    train = subsample_train(available_train, train_fraction, seed)
    model_results: dict[str, dict[str, object]] = {}
    score_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fitted_models: dict[str, object] = {}
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
        fitted_models[name] = model
    baseline_test = model_results["dummy"]["test"]
    logistic_test = model_results["weighted_logistic_regression"]["test"]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "config": {
            "seed": seed,
            "rows": rows,
            "budget_fraction": budget_fraction,
            "scenario": scenario,
            "train_fraction": train_fraction,
        },
        "data": {
            "train_rows": len(train),
            "available_train_rows": len(available_train),
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
        "scenario_result": scenario_result(
            scenario=scenario,
            seed=seed,
            train=train,
            available_train=available_train,
            logistic_test=logistic_test,
            budget_fraction=budget_fraction,
            train_fraction=train_fraction,
        ),
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
            "permutation_importance": permutation_diagnostics(
                fitted_models["weighted_logistic_regression"],
                test,
                seed=seed,
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
    parser.add_argument("--scenario", choices=SCENARIOS, default="retention_budget")
    parser.add_argument("--train-fraction", type=float, default=1.0)
    args = parser.parse_args(argv)
    result = run_experiment(
        seed=args.seed,
        rows=args.rows,
        budget_fraction=args.budget_fraction,
        scenario=args.scenario,
        train_fraction=args.train_fraction,
    )
    write_result(result, args.output)
    print(f"experiment result written to {args.output}")


if __name__ == "__main__":
    main()
