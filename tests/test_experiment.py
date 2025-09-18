from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from telco_churn.data import validate_frame
from telco_churn.experiment import (
    main,
    missing_diagnostics,
    run_experiment,
    segment_diagnostics,
    unseen_category_diagnostics,
)
from telco_churn.generate_smoke_data import generate_smoke_frame


def test_experiment_runner_is_reproducible() -> None:
    first = run_experiment(seed=17, rows=120, budget_fraction=0.15)
    second = run_experiment(seed=17, rows=120, budget_fraction=0.15)

    assert first == second
    assert first["schema_version"] == 1
    assert first["data"]["train_rows"] == 72
    assert first["models"]["weighted_logistic_regression"]["test"]["selected_fraction"] > 0


def test_experiment_cli_writes_stable_json(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    main(
        [
            "--output",
            str(output),
            "--seed",
            "19",
            "--rows",
            "100",
            "--budget-fraction",
            "0.1",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["config"] == {
        "budget_fraction": 0.1,
        "missing_fraction": 0.0,
        "rows": 100,
        "scenario": "retention_budget",
        "seed": 19,
        "train_fraction": 1.0,
        "unseen_fraction": 0.0,
    }
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_experiment_compares_logistic_model_with_prior_baseline() -> None:
    result = run_experiment(seed=23, rows=180)

    assert set(result["models"]) == {"dummy", "weighted_logistic_regression"}
    assert result["models"]["dummy"]["test"]["roc_auc"] == 0.5
    expected_delta = (
        result["models"]["weighted_logistic_regression"]["test"]["pr_auc"]
        - result["models"]["dummy"]["test"]["pr_auc"]
    )
    assert result["delta_vs_baseline"]["test_pr_auc"] == expected_delta


def test_missing_diagnostics_reports_normalized_blanks_and_injected_nulls() -> None:
    frame = generate_smoke_frame(rows=120, seed=29)
    frame.loc[:5, "MonthlyCharges"] = None
    diagnostics = missing_diagnostics(validate_frame(frame))

    assert diagnostics["by_feature"]["MonthlyCharges"]["count"] == 6
    assert diagnostics["by_feature"]["TotalCharges"]["count"] >= 1
    assert diagnostics["rows_with_missing"] >= 6


def test_unseen_category_diagnostics_reports_new_evaluation_values() -> None:
    frame = validate_frame(generate_smoke_frame(rows=120, seed=31))
    train = frame.iloc[:80].copy()
    evaluation = frame.iloc[80:].copy()
    evaluation.loc[evaluation.index[:4], "Contract"] = "Trial contract"

    diagnostics = unseen_category_diagnostics(train, evaluation)

    assert diagnostics["rows_with_unseen"] == 4
    assert diagnostics["by_feature"]["Contract"]["values"] == ["Trial contract"]
    assert diagnostics["by_feature"]["Contract"]["fraction"] == 0.1


def test_segment_diagnostics_preserves_segment_population() -> None:
    frame = validate_frame(generate_smoke_frame(rows=120, seed=37))
    scores = np.linspace(0.0, 1.0, len(frame))

    diagnostics = segment_diagnostics(frame, scores, 0.2)

    assert set(diagnostics) == {"Contract", "InternetService"}
    assert sum(segment["rows"] for segment in diagnostics["Contract"].values()) == len(frame)
    assert all(
        0 <= segment["selected_fraction"] <= 1
        for segment in diagnostics["InternetService"].values()
    )


def test_permutation_diagnostics_cover_features_in_ranked_order() -> None:
    diagnostics = run_experiment(seed=41, rows=120)["diagnostics"]["permutation_importance"]

    assert len(diagnostics["features"]) == 19
    means = [row["importance_mean"] for row in diagnostics["features"]]
    assert means == sorted(means, reverse=True)
    assert diagnostics["repeats"] == 3


def test_retention_budget_scenario_reports_exact_selected_population() -> None:
    result = run_experiment(seed=43, rows=200, budget_fraction=0.075)

    assert result["scenario_result"]["selected_customers"] == 3
    assert result["scenario_result"]["retention_budget_fraction"] == 0.075
    assert result["models"]["weighted_logistic_regression"]["test"]["selected_fraction"] == 0.075


def test_learning_curve_scenario_uses_stratified_train_fraction() -> None:
    result = run_experiment(
        seed=47,
        rows=200,
        scenario="learning_curve",
        train_fraction=0.5,
    )

    assert result["data"]["available_train_rows"] == 120
    assert result["data"]["train_rows"] == 60
    assert result["scenario_result"]["used_train_rows"] == 60
    assert 0 <= result["scenario_result"]["test_pr_auc"] <= 1


def test_seed_stability_scenario_records_seed_and_ranking_metrics() -> None:
    result = run_experiment(seed=53, rows=160, scenario="seed_stability")

    assert result["config"]["seed"] == 53
    assert result["scenario_result"]["seed"] == 53
    assert set(result["scenario_result"]) == {
        "seed",
        "test_pr_auc",
        "test_roc_auc",
        "test_recall",
    }


def test_data_quality_scenario_injects_missing_and_unseen_values() -> None:
    result = run_experiment(
        seed=59,
        rows=200,
        scenario="data_quality",
        missing_fraction=0.1,
        unseen_fraction=0.15,
    )

    assert result["scenario_result"]["injected_missing_rows"] == 4
    assert result["scenario_result"]["injected_unseen_rows"] == 6
    assert result["diagnostics"]["missing"]["by_feature"]["MonthlyCharges"]["count"] == 8
    assert (
        result["diagnostics"]["unseen_categories"]["test"]["by_feature"]["Contract"]["count"] == 6
    )
