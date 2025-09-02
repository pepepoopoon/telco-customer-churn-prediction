from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from telco_churn.data import FEATURES, SchemaError, split_data, validate_frame
from telco_churn.evaluate import main as evaluate
from telco_churn.generate_smoke_data import generate_smoke_frame
from telco_churn.modeling import classification_metrics, select_for_budget
from telco_churn.predict import main as predict
from telco_churn.train import main as train


class TelcoWorkflowTest(unittest.TestCase):
    def test_retention_budget_is_exact_when_scores_are_tied(self) -> None:
        scores = np.full(10, 0.5)
        selected = select_for_budget(scores, 0.20)
        metrics = classification_metrics(np.array([0, 1] * 5), scores, 0.20)

        self.assertEqual(selected.sum(), 2)
        self.assertEqual(np.flatnonzero(selected).tolist(), [0, 1])
        self.assertEqual(metrics["selected_fraction"], 0.20)

    def test_schema_handles_blank_total_charges_and_excludes_identifier(self) -> None:
        frame = validate_frame(generate_smoke_frame(120))
        pd.testing.assert_frame_equal(generate_smoke_frame(120), generate_smoke_frame(120))
        self.assertNotIn("customerID", FEATURES)
        self.assertTrue(frame["TotalCharges"].isna().any())
        train_frame, validation, test = split_data(frame)
        self.assertEqual(len(frame), len(train_frame) + len(validation) + len(test))
        with self.assertRaises(SchemaError):
            validate_frame(frame.drop(columns=["Contract"]))

        duplicated = frame.copy()
        duplicated.loc[1, "customerID"] = duplicated.loc[0, "customerID"]
        with self.assertRaisesRegex(SchemaError, "customerID must be unique"):
            validate_frame(duplicated)

    def test_end_to_end_without_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "smoke.csv"
            artifact = root / "model.joblib"
            validation = root / "validation.json"
            metrics = root / "metrics.json"
            errors = root / "errors.csv"
            importance = root / "importance.csv"
            predictions = root / "predictions.csv"
            frame = generate_smoke_frame(180)
            frame.to_csv(data_path, index=False)
            train(
                ["--data", str(data_path), "--artifact", str(artifact), "--report", str(validation)]
            )
            evaluate(
                [
                    "--data",
                    str(data_path),
                    "--artifact",
                    str(artifact),
                    "--metrics",
                    str(metrics),
                    "--errors",
                    str(errors),
                    "--importance",
                    str(importance),
                ]
            )
            inference = root / "inference.csv"
            frame.drop(columns=["Churn"]).head(8).to_csv(inference, index=False)
            predict(
                [
                    "--data",
                    str(inference),
                    "--artifact",
                    str(artifact),
                    "--output",
                    str(predictions),
                ]
            )
            self.assertIn("segments", json.loads(metrics.read_text(encoding="utf-8")))
            self.assertEqual(set(pd.read_csv(importance)["feature"]), set(FEATURES))
            self.assertEqual(len(pd.read_csv(predictions)), 8)


if __name__ == "__main__":
    unittest.main()
