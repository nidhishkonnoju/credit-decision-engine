from __future__ import annotations

import argparse

from app.inference import load_applicant_json, predict_applicant
from app.modeling import fit_credit_model, summarize_decision
from app.preprocessing import DATASET_DIR, build_preprocessed_data, print_preprocessing_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the credit model and score one applicant.")
    parser.add_argument("--input", type=str, help="Path to a JSON object containing applicant features.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = build_preprocessed_data(DATASET_DIR)
    print_preprocessing_summary(data)

    model_result = fit_credit_model(data)
    print("Model metrics:", model_result["metrics"])
    print("Cross-validation metrics:", model_result["cross_validation_metrics"])

    if args.input:
        applicant = load_applicant_json(args.input, data["feature_columns"])
        decision = predict_applicant(model_result, applicant)
        print("Applicant decision:", decision["decision"])
        print("Rejection risk:", decision["probability"])
        print("Decision threshold:", decision["threshold"])
        print("SHAP reasons:")
        for reason in decision["top_reasons"]:
            print(f"- {reason['feature']}: {reason['direction']} ({reason['contribution']})")
    else:
        sample_row = data["X_test"].iloc[[0]].copy()
        decision = summarize_decision(
            model_result["model"],
            model_result["preprocessor"],
            sample_row,
            threshold=model_result["threshold"],
        )
        print("Sample decision:", decision)
    print("Training and decision pipeline complete.")
