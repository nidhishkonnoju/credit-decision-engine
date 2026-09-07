from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

ARTIFACT_VERSION = 1
DEFAULT_ARTIFACT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "credit_model_v1.joblib"


def save_model_artifact(
    model_result: dict[str, Any],
    path: str | Path = DEFAULT_ARTIFACT_PATH,
) -> Path:
    """Persist the complete model scoring contract to a versioned artifact."""
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "artifact_version": ARTIFACT_VERSION,
            "model_result": model_result,
        },
        artifact_path,
    )
    return artifact_path


def load_model_artifact(path: str | Path = DEFAULT_ARTIFACT_PATH) -> dict[str, Any]:
    """Load and validate a persisted model scoring contract."""
    artifact_path = Path(path)
    artifact = joblib.load(artifact_path)
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported model artifact version: {artifact.get('artifact_version')}"
        )

    model_result = artifact.get("model_result")
    required_keys = {
        "model",
        "preprocessor",
        "raw_feature_columns",
        "feature_names",
        "threshold",
        "review_threshold",
        "stable_feature_names",
    }
    if not isinstance(model_result, dict) or not required_keys <= model_result.keys():
        raise ValueError("Model artifact is missing required scoring-contract fields.")
    return model_result
