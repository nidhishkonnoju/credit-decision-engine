from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_DIR = Path(__file__).resolve().parents[1] / "Dataset"
PROTECTED_COLUMNS = {"PROSPECTID", "Approved_Flag", "GENDER", "MARITALSTATUS"}
AUDITED_LABEL_PROXY_COLUMNS = {
    "Credit_Score",
    "CC_enq",
    "CC_enq_L12m",
    "CC_enq_L6m",
    "PL_enq",
    "PL_enq_L12m",
    "PL_enq_L6m",
    "Tot_Missed_Pmnt",
    "enq_L12m",
    "enq_L3m",
    "enq_L6m",
    "first_prod_enq2",
    "last_prod_enq2",
    "max_deliq_12mts",
    "max_deliq_6mts",
    "max_delinquency_level",
    "max_recent_level_of_deliq",
    "num_deliq_12mts",
    "num_deliq_6_12mts",
    "num_deliq_6mts",
    "num_std",
    "num_std_12mts",
    "num_std_6mts",
    "num_times_30p_dpd",
    "num_times_60p_dpd",
    "num_times_delinquent",
    "pct_CC_enq_L6m_of_L12m",
    "pct_CC_enq_L6m_of_ever",
    "pct_PL_enq_L6m_of_L12m",
    "pct_PL_enq_L6m_of_ever",
    "recent_level_of_deliq",
    "time_since_first_deliquency",
    "time_since_recent_deliquency",
    "time_since_recent_enq",
    "tot_enq",
}


def load_credit_data(dataset_dir: str | Path = DATASET_DIR) -> pd.DataFrame:
    """Load the bank and bureau datasets and merge them on PROSPECTID."""
    dataset_dir = Path(dataset_dir)
    external = pd.read_excel(dataset_dir / "External_Cibil_Dataset.xlsx")
    internal = pd.read_excel(dataset_dir / "Internal_Bank_Dataset.xlsx")

    merged = external.merge(
        internal,
        on="PROSPECTID",
        how="inner",
        suffixes=("_external", "_internal"),
    )
    return merged


def define_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Treat P4 as reject/high-risk and P1+P2+P3 as approve/low-risk."""
    processed = df.copy()
    processed["target"] = processed["Approved_Flag"].map({"P1": 0, "P2": 0, "P3": 0, "P4": 1})
    processed["target"] = processed["target"].fillna(0).astype(int)
    return processed


def handle_missing_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace common numeric sentinel values with NaN before imputation."""
    processed = df.copy()
    for column in processed.columns:
        if column == "Approved_Flag":
            continue
        if pd.api.types.is_numeric_dtype(processed[column]):
            processed[column] = pd.to_numeric(processed[column], errors="coerce")
            processed[column] = processed[column].replace(-99999, np.nan)
    return processed


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a simple preprocessor for numeric and categorical feature columns."""
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )

    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_preprocessed_data(
    dataset_dir: str | Path = DATASET_DIR,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Return the cleaned dataset and train/test splits ready for modeling."""
    df = load_credit_data(dataset_dir)
    df = handle_missing_sentinels(df)
    df = define_binary_target(df)

    feature_columns = [
        column
        for column in df.columns
        if column not in PROTECTED_COLUMNS
        and column not in AUDITED_LABEL_PROXY_COLUMNS
        and column != "target"
    ]

    X = df[feature_columns]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return {
        "df": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": feature_columns,
        "preprocessor": preprocessor,
        "X_train_processed": X_train_processed,
        "X_test_processed": X_test_processed,
        "positive_rate": float(y.mean()),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
    }


def print_preprocessing_summary(data: dict) -> None:
    """Print a brief summary of the binary target and prepared splits."""
    print("Dataset shape:", data["df"].shape)
    print("Positive rate (P4 reject):", round(data["positive_rate"], 4))
    print("Train positive rate:", round(data["train_positive_rate"], 4))
    print("Test positive rate:", round(data["test_positive_rate"], 4))
    print("Feature columns used:", len(data["feature_columns"]))
    print("Excluded protected/label-proxy fields:", len(PROTECTED_COLUMNS | AUDITED_LABEL_PROXY_COLUMNS))
    print("Train split shape:", data["X_train"].shape)
    print("Test split shape:", data["X_test"].shape)

