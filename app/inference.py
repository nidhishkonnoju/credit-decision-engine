from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.modeling import summarize_decision
from app.preprocessing import handle_missing_sentinels


def _normalize_csv_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common CSV messiness to the same null semantics used during training.

    Only columns where the majority of non-null values are parseable as numbers
    are coerced. Categorical columns (e.g., EDUCATION) are left as-is.
    """
    normalized = df.copy()
    for column in normalized.columns:
        if pd.api.types.is_string_dtype(normalized[column]):
            cleaned = normalized[column].astype(str)
            cleaned = cleaned.replace({"nan": pd.NA, "NaN": pd.NA, "NA": pd.NA, "N/A": pd.NA, "": pd.NA})
            cleaned = cleaned.str.replace(r"[,$\s]", "", regex=True)
            cleaned = cleaned.replace("-99999", pd.NA)
            numeric_attempt = pd.to_numeric(cleaned, errors="coerce")
            non_null_count = cleaned.notna().sum()
            numeric_count = numeric_attempt.notna().sum()
            # Only coerce if the majority of non-null values are parseable as numbers
            if non_null_count > 0 and numeric_count / non_null_count > 0.5:
                normalized[column] = numeric_attempt
            else:
                # Preserve the original string values with only null normalization
                normalized[column] = normalized[column].replace(
                    {"nan": pd.NA, "NaN": pd.NA, "NA": pd.NA, "N/A": pd.NA, "": pd.NA}
                )
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

    normalized_features = handle_missing_sentinels(normalized[required_columns])
    results: list[dict[str, Any]] = []
    for i, (_, row) in enumerate(normalized_features.iterrows()):
        row_df = pd.DataFrame([row])
        pred = predict_applicant(model_result, row_df)
        if "applicant_id" in raw.columns:
            pred["applicant_id"] = str(raw.iloc[i]["applicant_id"])
        if "profile_name" in raw.columns:
            pred["profile_name"] = str(raw.iloc[i]["profile_name"])
        results.append(pred)
    return results


def load_applicant_from_csv(
    csv_path: str | Path,
    selector: int | str,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a single applicant row from a CSV by 0-based index or applicant_id.

    Returns:
        tuple of (applicant_df, metadata_dict)
    """
    csv_path = Path(csv_path)
    raw = pd.read_csv(csv_path)
    normalized = _normalize_csv_values(raw)

    missing = [column for column in feature_columns if column not in normalized.columns]
    if missing:
        raise ValueError(f"CSV input is missing required columns: {', '.join(missing)}")

    row_idx: int | None = None
    if isinstance(selector, int) or (isinstance(selector, str) and selector.isdigit()):
        idx = int(selector)
        if 0 <= idx < len(normalized):
            row_idx = idx
        else:
            raise IndexError(
                f"Row index {idx} out of range for CSV with {len(normalized)} rows (valid: 0 to {len(normalized)-1})."
            )
    elif isinstance(selector, str) and "applicant_id" in raw.columns:
        matches = raw.index[
            raw["applicant_id"].astype(str).str.strip().str.lower() == selector.strip().lower()
        ].tolist()
        if matches:
            row_idx = matches[0]
        else:
            raise ValueError(f"No applicant found with applicant_id='{selector}'.")
    else:
        raise ValueError(f"Invalid row selector: {selector}")

    metadata: dict[str, Any] = {"row_index": row_idx}
    if "applicant_id" in raw.columns:
        metadata["applicant_id"] = str(raw.iloc[row_idx]["applicant_id"])
    if "profile_name" in raw.columns:
        metadata["profile_name"] = str(raw.iloc[row_idx]["profile_name"])

    applicant_df = pd.DataFrame([normalized.iloc[row_idx][feature_columns]])
    return applicant_df, metadata