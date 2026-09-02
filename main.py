from __future__ import annotations

import argparse

from app.cam import generate_credit_appraisal_memo
from app.inference import load_applicant_json, predict_applicant
from app.modeling import fit_credit_model, summarize_decision
from app.preprocessing import DATASET_DIR, build_preprocessed_data, print_preprocessing_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the credit model and score one applicant or a batch CSV.")
    parser.add_argument("--input", type=str, help="Path to a JSON object containing applicant features.")
    parser.add_argument("--batch", type=str, help="Path to a CSV file of applicants with the same model columns.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = build_preprocessed_data(DATASET_DIR)
    print_preprocessing_summary(data)

    model_result = fit_credit_model(data)
    print("Model metrics:", model_result["metrics"])
    print("Cross-validation metrics:", model_result["cross_validation_metrics"])

    if args.batch:
        from app.inference import batch_score_csv

        scores = batch_score_csv(args.batch, model_result)
        counts = {}
        for label in ["APPROVE", "REVIEW", "REJECT"]:
            counts[label] = sum(1 for item in scores if item["decision"] == label)
        print("Batch decision counts:", counts)
        print("Batch results:")
        for row in scores:
            memo = generate_credit_appraisal_memo(row)
            print(f"- {memo['applicant_facing']['decision']}: {memo['applicant_facing']['summary']}")
    elif args.input:
        applicant = load_applicant_json(args.input, data["feature_columns"])
        decision = predict_applicant(model_result, applicant)
        memo = generate_credit_appraisal_memo(decision)
        print("Applicant decision:", memo["applicant_facing"]["decision"])
        print("Estimated rejection risk:", memo["applicant_facing"]["probability"])
        print("Decision threshold:", memo["applicant_facing"]["threshold"])
        if memo["applicant_facing"].get("review_threshold") is not None:
            print("Review threshold:", memo["applicant_facing"]["review_threshold"])
        print(memo["applicant_facing"]["summary"])
        print("Why this score was assigned:")
        for reason in memo["applicant_facing"]["reasons"]:
            print(f"- {reason}")
    else:
        sample_row = data["X_test"].iloc[[0]].copy()
        decision = summarize_decision(
            model_result["model"],
            model_result["preprocessor"],
            sample_row,
            threshold=model_result["threshold"],
            review_threshold=model_result["review_threshold"],
            stable_feature_names=model_result["stable_feature_names"],
        )
        memo = generate_credit_appraisal_memo(decision)
        print("Sample decision:", memo["applicant_facing"]["decision"])
        print(memo["applicant_facing"]["summary"])
        print("Sample reasons:")
        for reason in memo["applicant_facing"]["reasons"]:
            print(f"- {reason}")
    print("Training and decision pipeline complete.")
