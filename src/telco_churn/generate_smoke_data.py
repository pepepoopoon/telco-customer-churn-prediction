"""Create deterministic synthetic telco data for smoke tests only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_smoke_frame(rows: int = 400, seed: int = 20250809) -> pd.DataFrame:
    if rows < 80:
        raise ValueError("rows must be at least 80")
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, rows)
    tenure[0] = 0
    contract = rng.choice(["Month-to-month", "One year", "Two year"], rows, p=[0.56, 0.25, 0.19])
    internet = rng.choice(["DSL", "Fiber optic", "No"], rows, p=[0.34, 0.46, 0.20])
    tech_support = np.where(
        internet == "No", "No internet service", rng.choice(["Yes", "No"], rows, p=[0.38, 0.62])
    )
    monthly = np.clip(
        24 + 31 * (internet == "Fiber optic") + 15 * (internet == "DSL") + rng.normal(20, 13, rows),
        18,
        125,
    )
    total = np.maximum(0, monthly * tenure + rng.normal(0, 90, rows))
    logit = (
        -1.8
        + 1.25 * (contract == "Month-to-month")
        + 0.65 * (internet == "Fiber optic")
        + 0.55 * (tech_support == "No")
        - 0.035 * tenure
        + 0.008 * (monthly - 60)
    )
    churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    frame = pd.DataFrame(
        {
            "customerID": [f"SMOKE-{index:05d}" for index in range(rows)],
            "gender": rng.choice(["Female", "Male"], rows),
            "SeniorCitizen": rng.choice([0, 1], rows, p=[0.84, 0.16]),
            "Partner": rng.choice(["Yes", "No"], rows),
            "Dependents": rng.choice(["Yes", "No"], rows, p=[0.30, 0.70]),
            "tenure": tenure,
            "PhoneService": rng.choice(["Yes", "No"], rows, p=[0.90, 0.10]),
            "MultipleLines": rng.choice(["Yes", "No", "No phone service"], rows),
            "InternetService": internet,
            "OnlineSecurity": np.where(
                internet == "No", "No internet service", rng.choice(["Yes", "No"], rows)
            ),
            "OnlineBackup": np.where(
                internet == "No", "No internet service", rng.choice(["Yes", "No"], rows)
            ),
            "DeviceProtection": np.where(
                internet == "No", "No internet service", rng.choice(["Yes", "No"], rows)
            ),
            "TechSupport": tech_support,
            "StreamingTV": np.where(
                internet == "No", "No internet service", rng.choice(["Yes", "No"], rows)
            ),
            "StreamingMovies": np.where(
                internet == "No", "No internet service", rng.choice(["Yes", "No"], rows)
            ),
            "Contract": contract,
            "PaperlessBilling": rng.choice(["Yes", "No"], rows, p=[0.60, 0.40]),
            "PaymentMethod": rng.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                rows,
            ),
            "MonthlyCharges": monthly.round(2),
            "TotalCharges": total.round(2).astype(object),
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )
    frame.loc[frame["tenure"] == 0, "TotalCharges"] = " "
    return frame


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/smoke.csv"))
    parser.add_argument("--rows", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20250809)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_smoke_frame(args.rows, args.seed).to_csv(args.output, index=False)
    print(f"synthetic smoke data written to {args.output}")


if __name__ == "__main__":
    main()
