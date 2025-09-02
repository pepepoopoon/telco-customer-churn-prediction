"""IBM Telco-style schema and leakage-safe splits."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Churn"
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class SchemaError(ValueError):
    """Raised when customer data violates the scoring contract."""


def _binary_target(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    mapped = values.map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0})
    if mapped.isna().any():
        bad = sorted(values[mapped.isna()].unique().tolist())
        raise SchemaError(f"unsupported target values: {bad[:5]}")
    return mapped.astype(int)


def validate_frame(frame: pd.DataFrame, *, require_target: bool = True) -> pd.DataFrame:
    required = FEATURES + ([TARGET] if require_target else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    if frame.empty:
        raise SchemaError("dataset is empty")
    clean = frame.copy()
    if "customerID" in clean:
        customer_id = clean["customerID"].astype("string").str.strip()
        if customer_id.isna().any() or customer_id.eq("").any():
            raise SchemaError("customerID cannot be empty when provided")
        if customer_id.duplicated().any():
            duplicates = customer_id[customer_id.duplicated(keep=False)].unique().tolist()
            raise SchemaError(f"customerID must be unique; duplicates: {duplicates[:5]}")
        clean["customerID"] = customer_id
    for column in NUMERIC_FEATURES:
        raw = clean[column].replace(r"^\s*$", pd.NA, regex=True)
        converted = pd.to_numeric(raw, errors="coerce")
        if converted.isna().sum() > raw.isna().sum():
            raise SchemaError(f"{column} contains non-numeric values")
        clean[column] = converted
    if (clean[["tenure", "MonthlyCharges", "TotalCharges"]].dropna() < 0).any().any():
        raise SchemaError("tenure and charges cannot be negative")
    senior = clean["SeniorCitizen"].astype(str).str.strip()
    if not set(senior.unique()).issubset({"0", "1"}):
        raise SchemaError("SeniorCitizen must contain only 0 or 1")
    clean["SeniorCitizen"] = senior
    if require_target:
        clean[TARGET] = _binary_target(clean[TARGET])
        if clean[TARGET].nunique() != 2:
            raise SchemaError("target must contain both classes")
    return clean


def load_data(path: str | Path, *, require_target: bool = True) -> pd.DataFrame:
    return validate_frame(pd.read_csv(path), require_target=require_target)


def split_data(
    frame: pd.DataFrame, *, seed: int = 20250809
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_validation, test = train_test_split(
        frame, test_size=0.20, random_state=seed, stratify=frame[TARGET]
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        random_state=seed,
        stratify=train_validation[TARGET],
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )
