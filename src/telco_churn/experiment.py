"""Run deterministic, synthetic churn experiments and persist their evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import FEATURES, TARGET, split_data, validate_frame
from .generate_smoke_data import generate_smoke_frame
from .modeling import candidate_models, classification_metrics

RESULT_SCHEMA_VERSION = 1


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
