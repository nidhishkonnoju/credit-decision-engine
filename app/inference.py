from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.modeling import summarize_decision
from app.preprocessing import handle_missing_sentinels


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
    )