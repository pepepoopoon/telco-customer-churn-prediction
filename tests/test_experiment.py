from __future__ import annotations

import json
from pathlib import Path

from telco_churn.experiment import main, run_experiment


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
