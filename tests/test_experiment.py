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
    assert payload["config"] == {"budget_fraction": 0.1, "rows": 100, "seed": 19}
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
