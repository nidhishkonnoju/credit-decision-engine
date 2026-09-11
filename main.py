from __future__ import annotations

import argparse

from app.cam import generate_cam, generate_credit_appraisal_memo
from app.inference import load_applicant_json, predict_applicant
from app.modeling import fit_credit_model, summarize_decision
from app.persistence import DEFAULT_ARTIFACT_PATH, load_model_artifact, save_model_artifact
from app.preprocessing import DATASET_DIR, build_preprocessed_data, handle_missing_sentinels, print_preprocessing_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the credit model and score one applicant or a batch CSV.")
    parser.add_argument("--input", type=str, help="Path to a JSON object containing applicant features.")
    parser.add_argument("--batch", type=str, help="Path to a CSV file of applicants with the same model columns.")
    parser.add_argument("--row", type=str, default=None, help="Select a specific row index (0-based) or applicant_id from a CSV batch to score.")
    parser.add_argument("--interactive", action="store_true", help="Prompt interactively to choose which applicant from the CSV batch to process.")
    parser.add_argument("--retrain", action="store_true", help="Retrain and replace the persisted model artifact.")
    parser.add_argument("--cam", action="store_true", help="Print the structured 5C credit appraisal memo.")
    parser.add_argument("--pdf", nargs="?", const="auto", default=None, help="Generate a downloadable PDF CAM report. Optionally specify output path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = None
    if DEFAULT_ARTIFACT_PATH.exists() and not args.retrain:
        model_result = load_model_artifact(DEFAULT_ARTIFACT_PATH)
        print(f"Loaded model artifact: {DEFAULT_ARTIFACT_PATH}")
    else:
        data = build_preprocessed_data(DATASET_DIR)
        print_preprocessing_summary(data)
        model_result = fit_credit_model(data)
        artifact_path = save_model_artifact(model_result, DEFAULT_ARTIFACT_PATH)
        print(f"Saved model artifact: {artifact_path}")

    # --pdf implies --cam
    if args.pdf is not None:
        args.cam = True

    print("Model metrics:", model_result["metrics"])
    cross_val = model_result.get("cross_validation_metrics")
    if cross_val is not None:
        print("Cross-validation metrics:", cross_val)
    stability = model_result.get("stability_result")
    if stability:
        print(
            "Stability metrics:",
            {
                "seeds": stability["n_seeds"],
                "overall_w": stability["overall_w"],
                "top_5_w": stability["top_5_w"],
                "diversity_band_w": stability["diversity_band_w"],
                "stable_features": len(stability["stable_features"]),
            },
        )

    if args.batch:
        import pandas as pd
        from app.inference import batch_score_csv, load_applicant_from_csv

        selected_row = args.row
        if args.interactive and selected_row is None:
            raw_df = pd.read_csv(args.batch)
            print(f"\nCSV '{args.batch}' contains {len(raw_df)} applicants:")
            for idx, r in raw_df.iterrows():
                app_id = r.get("applicant_id", f"Row {idx}")
                profile = r.get("profile_name", "")
                income = r.get("NETMONTHLYINCOME", "N/A")
                age = r.get("AGE", "N/A")
                desc = f" - {profile}" if profile else ""
                print(f"  [{idx}] {app_id}{desc} | Age: {age} | Income: {income}")
            prompt_val = input("\nEnter applicant index or ID to process (or press Enter / 'all' to score full batch): ").strip()
            if prompt_val and prompt_val.lower() != "all":
                selected_row = prompt_val

        if selected_row is not None:
            applicant_df, meta = load_applicant_from_csv(
                args.batch, selected_row, model_result["raw_feature_columns"]
            )
            app_label = meta.get("applicant_id", f"Row {meta['row_index']}")
            if meta.get("profile_name"):
                app_label += f" ({meta['profile_name']})"
            print(f"Processing selected applicant: {app_label}")

            if args.cam:
                applicant = handle_missing_sentinels(applicant_df)
                cam = generate_cam(applicant, model_result)
                print("Structured 5C Credit Appraisal Memo")
                print(cam["summary"])
                for category, section in cam["sections"].items():
                    print(f"{category}:")
                    if section["reasons"]:
                        for reason in section["reasons"]:
                            print(f"- {reason['text']}")
                    else:
                        print(f"- {section['empty_message']}")
                print("Applicant decision:", cam["decision"])
                print("Estimated rejection risk:", cam["probability"])
                print("Decision threshold:", cam["threshold"])
                if cam["review_threshold"] is not None:
                    print("Review threshold:", cam["review_threshold"])
                if args.pdf is not None:
                    from app.pdf_report import generate_cam_pdf

                    pdf_path = generate_cam_pdf(cam, model_result, output_path=args.pdf)
                    print(f"PDF report saved to: {pdf_path}")
                print("Training and decision pipeline complete.")
                raise SystemExit(0)

            decision = predict_applicant(model_result, applicant_df)
            memo = generate_credit_appraisal_memo(decision)
            print("Applicant decision:", memo["applicant_facing"]["decision"])
            print("Estimated rejection risk:", memo["applicant_facing"]["probability"])
            print("Decision threshold:", memo["applicant_facing"]["threshold"])
            if memo["applicant_facing"].get("review_threshold") is not None:
                print("Review threshold:", memo["applicant_facing"]["review_threshold"])
            print(memo["applicant_facing"]["summary"])
            print("Why this score was assigned:")
            reasons = memo["applicant_facing"]["reasons"]
            if reasons:
                for reason in reasons:
                    print(f"- {reason}")
            else:
                print(f"- {memo['applicant_facing']['no_high_confidence_factors']}")
            if args.pdf is not None:
                from app.pdf_report import generate_cam_pdf

                cam = generate_cam(handle_missing_sentinels(applicant_df), model_result)
                pdf_path = generate_cam_pdf(cam, model_result, output_path=args.pdf)
                print(f"PDF report saved to: {pdf_path}")
            print("Training and decision pipeline complete.")
            raise SystemExit(0)

        scores = batch_score_csv(args.batch, model_result)
        counts = {}
        for label in ["APPROVE", "REVIEW", "REJECT"]:
            counts[label] = sum(1 for item in scores if item["decision"] == label)
        print("Batch decision counts:", counts)
        print("Batch results:")
        for i, row in enumerate(scores):
            memo = generate_credit_appraisal_memo(row)
            label = row.get("applicant_id", f"Row {i}")
            if row.get("profile_name"):
                label += f" ({row['profile_name']})"
            print(f"- [{label}] {memo['applicant_facing']['decision']}: {memo['applicant_facing']['summary']}")
    elif args.input:
        applicant = load_applicant_json(args.input, model_result["raw_feature_columns"])
        if args.cam:
            applicant = handle_missing_sentinels(applicant)
            cam = generate_cam(applicant, model_result)
            print("Structured 5C Credit Appraisal Memo")
            print(cam["summary"])
            for category, section in cam["sections"].items():
                print(f"{category}:")
                if section["reasons"]:
                    for reason in section["reasons"]:
                        print(f"- {reason['text']}")
                else:
                    print(f"- {section['empty_message']}")
            print("Applicant decision:", cam["decision"])
            print("Estimated rejection risk:", cam["probability"])
            print("Decision threshold:", cam["threshold"])
            if cam["review_threshold"] is not None:
                print("Review threshold:", cam["review_threshold"])
            if args.pdf is not None:
                from app.pdf_report import generate_cam_pdf

                pdf_path = generate_cam_pdf(cam, model_result, output_path=args.pdf)
                print(f"PDF report saved to: {pdf_path}")
            print("Training and decision pipeline complete.")
            raise SystemExit(0)
        decision = predict_applicant(model_result, applicant)
        memo = generate_credit_appraisal_memo(decision)
        print("Applicant decision:", memo["applicant_facing"]["decision"])
        print("Estimated rejection risk:", memo["applicant_facing"]["probability"])
        print("Decision threshold:", memo["applicant_facing"]["threshold"])
        if memo["applicant_facing"].get("review_threshold") is not None:
            print("Review threshold:", memo["applicant_facing"]["review_threshold"])
        print(memo["applicant_facing"]["summary"])
        print("Why this score was assigned:")
        reasons = memo["applicant_facing"]["reasons"]
        if reasons:
            for reason in reasons:
                print(f"- {reason}")
        else:
            print(f"- {memo['applicant_facing']['no_high_confidence_factors']}")
    else:
        if data is None:
            print("No applicant input supplied. Use --input or --batch to score an applicant.")
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
