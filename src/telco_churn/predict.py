"""Score customers with a trained churn artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import FEATURES, load_data
from .modeling import select_for_budget


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("reports/predictions.csv"))
    args = parser.parse_args(argv)
    artifact = joblib.load(args.artifact)
    frame = load_data(args.data, require_target=False)
    scores = artifact["model"].predict_proba(frame[FEATURES])[:, 1]
    selected = select_for_budget(scores, float(artifact["budget_fraction"]))
    identity = frame["customerID"] if "customerID" in frame else frame.index.astype(str)
    output = pd.DataFrame(
        {
            "customerID": identity,
            "churn_probability": scores,
            "selected_for_retention": selected,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"predictions written to {args.output}")


if __name__ == "__main__":
    main()
