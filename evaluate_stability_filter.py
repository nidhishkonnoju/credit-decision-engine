from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.cam import FIVE_CATEGORIES, generate_cam
from app.modeling import summarize_decision
from app.persistence import DEFAULT_ARTIFACT_PATH, load_model_artifact
from app.preprocessing import DATASET_DIR, build_preprocessed_data

DEFAULT_SAMPLE_PATH = Path("evaluation") / "phase4_sample.json"
DEFAULT_RESULTS_PATH = Path("evaluation") / "phase4_results.json"
SAMPLE_SIZE = 50


def load_or_create_sample(data: dict[str, Any], path: Path, size: int) -> list[int]:
    """Persist test-row indices so repeated evaluations use the same applicants."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        indices = saved["row_indices"] if isinstance(saved, dict) else saved
        available = set(data["X_test"].index.tolist())
        if not set(indices) <= available:
            raise ValueError("Saved evaluation sample contains rows outside the current test split.")
        expected_ids = [int(data["df"].loc[index, "PROSPECTID"]) for index in indices]
        saved_ids = saved.get("prospect_ids") if isinstance(saved, dict) else None
        if saved_ids is not None and saved_ids != expected_ids:
            raise ValueError("Saved evaluation sample PROSPECTID values do not match the current dataset.")
        if not isinstance(saved, dict):
            path.write_text(
                json.dumps(
                    {"row_indices": indices, "prospect_ids": expected_ids},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return [int(index) for index in indices]

    indices = [int(index) for index in data["X_test"].index[:size].tolist()]
    path.write_text(
        json.dumps(
            {
                "row_indices": indices,
                "prospect_ids": [int(data["df"].loc[index, "PROSPECTID"]) for index in indices],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return indices


def evaluate(
    model_result: dict[str, Any],
    data: dict[str, Any],
    sample_indices: list[int],
) -> dict[str, Any]:
    """Compare unfiltered and stability-filtered explanations on fixed applicants."""
    stable_features = set(model_result["stable_feature_names"])
    unstable_features = set(model_result["filtered_stability_features"])
    raw_memos_with_unstable_reason = 0
    filtered_reason_count = 0
    raw_reason_count = 0
    empty_sections = {category: 0 for category in sorted(FIVE_CATEGORIES)}
    decisions = {"APPROVE": 0, "REVIEW": 0, "REJECT": 0}

    for index in sample_indices:
        applicant = data["X_test"].loc[[index]]
        raw_summary = summarize_decision(
            model_result["model"],
            model_result["preprocessor"],
            applicant,
            threshold=model_result["threshold"],
            review_threshold=model_result["review_threshold"],
        )
        raw_features = {reason["feature"] for reason in raw_summary["top_reasons"]}
        raw_reason_count += len(raw_features)
        if raw_features & unstable_features:
            raw_memos_with_unstable_reason += 1

        cam = generate_cam(applicant, model_result)
        filtered_reason_count += len(cam["stable_reason_features"])
        decisions[cam["decision"]] += 1
        for category, section in cam["sections"].items():
            if not section["reasons"]:
                empty_sections[category] += 1

    sample_count = len(sample_indices)
    return {
        "sample_size": sample_count,
        "sample_indices": sample_indices,
        "sample_prospect_ids": [int(data["df"].loc[index, "PROSPECTID"]) for index in sample_indices],
        "stable_feature_count": len(stable_features),
        "unstable_feature_count": len(unstable_features),
        "memos_with_unstable_raw_reason": raw_memos_with_unstable_reason,
        "memos_with_unstable_raw_reason_rate": round(raw_memos_with_unstable_reason / sample_count, 4),
        "raw_reason_count": raw_reason_count,
        "filtered_reason_count": filtered_reason_count,
        "reasons_removed_by_stability_filter": raw_reason_count - filtered_reason_count,
        "empty_section_counts": empty_sections,
        "empty_section_rates": {
            category: round(count / sample_count, 4)
            for category, count in empty_sections.items()
        },
        "decision_counts": decisions,
        "stability_rule": model_result["stability_result"]["stability_rule"],
        "stability_metrics": {
            key: model_result["stability_result"][key]
            for key in ("n_seeds", "overall_w", "top_5_w", "diversity_band_w")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate raw versus stability-filtered CAM reasons.")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    args = parser.parse_args()

    data = build_preprocessed_data(DATASET_DIR)
    model_result = load_model_artifact(args.artifact)
    sample_indices = load_or_create_sample(data, args.sample, SAMPLE_SIZE)
    results = evaluate(model_result, data, sample_indices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Phase 4 stability-filter evaluation")
    print(f"Fixed sample size: {results['sample_size']}")
    print(f"Memos with unstable raw reason: {results['memos_with_unstable_raw_reason']} ({results['memos_with_unstable_raw_reason_rate']:.2%})")
    print(f"Raw reasons: {results['raw_reason_count']}")
    print(f"Filtered reasons: {results['filtered_reason_count']}")
    print(f"Reasons removed: {results['reasons_removed_by_stability_filter']}")
    print(f"Decision counts: {results['decision_counts']}")
    print(f"Empty 5C sections: {results['empty_section_counts']}")
    print(f"Results written to: {args.output}")


if __name__ == "__main__":
    main()
