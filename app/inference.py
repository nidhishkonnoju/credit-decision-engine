from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.modeling import summarize_decision
from app.preprocessing import handle_missing_sentinels


def _normalize_csv_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common CSV messiness to the same null semantics used during training."""
    normalized = df.copy()
    for column in normalized.columns:
        if normalized[column].dtype == "object":
            normalized[column] = normalized[column].astype(str)
            normalized[column] = normalized[column].replace({"nan": pd.NA, "NaN": pd.NA, "NA": pd.NA, "N/A": pd.NA, "": pd.NA})
            normalized[column] = normalized[column].str.replace(r"[,$\s]", "", regex=True)
            normalized[column] = normalized[column].replace("-99999", pd.NA)
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def load_applicant_json(path: str | Path, feature_columns: list[str]) -> pd.DataFrame:
    """Load one applicant from a JSON object and validate its model fields."""
    with Path(path).open(encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if not isinstance(payload, dict):
        raise ValueError("Applicant input must be a JSON object.")

    missing = [column for column in feature_columns if column not in payload]
    if missing:
        raise ValueError(f"Applicant input is missing fields: {', '.join(missing)}")

    return pd.DataFrame([{column: payload[column] for column in feature_columns}])


def predict_applicant(model_result: dict[str, Any], applicant: pd.DataFrame) -> dict[str, Any]:
    """Generate a thresholded decision and SHAP reasons for one applicant."""
    feature_columns = model_result["raw_feature_columns"]
    missing = [column for column in feature_columns if column not in applicant.columns]
    if missing:
        raise ValueError(f"Applicant input is missing model fields: {', '.join(missing)}")

    applicant = handle_missing_sentinels(applicant[feature_columns])
    return summarize_decision(
        model_result["model"],
        model_result["preprocessor"],
        applicant,
        threshold=model_result["threshold"],
        stable_feature_names=model_result.get("stable_feature_names"),
        review_threshold=model_result.get("review_threshold"),
    )


def batch_score_csv(csv_path: str | Path, model_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Score a CSV of applicants by reusing the per-row decision logic in a loop."""
    csv_path = Path(csv_path)
    raw = pd.read_csv(csv_path)
    normalized = _normalize_csv_values(raw)

    required_columns = model_result["raw_feature_columns"]
    missing = [column for column in required_columns if column not in normalized.columns]
    if missing:
        raise ValueError(f"CSV input is missing required columns: {', '.join(missing)}")

    normalized = handle_missing_sentinels(normalized[required_columns])
    results: list[dict[str, Any]] = []
    for _, row in normalized.iterrows():
        row_df = pd.DataFrame([row])
        results.append(predict_applicant(model_result, row_df))
    return results