import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.cam import (
    FEATURE_LABEL_MAP,
    FEATURE_TO_5C,
    FEATURE_TO_5C_RATIONALE,
    FIVE_CATEGORIES,
    generate_cam,
    generate_credit_appraisal_memo,
    validate_feature_mapping,
)
from app.inference import _normalize_csv_values, batch_score_csv, load_applicant_from_csv, predict_applicant
from app.modeling import (
    compute_seed_stability,
    evaluate_model,
    find_best_threshold,
    find_review_threshold,
    fit_credit_model,
    kendalls_w,
    summarize_decision,
)
from app.persistence import load_model_artifact, save_model_artifact
from app.preprocessing import build_preprocessed_data


class ModelingSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_preprocessed_data()
        cls.model_result = fit_credit_model(cls.data)

    def test_model_and_decision_summary_are_produced(self):
        self.assertIn("model", self.model_result)
        self.assertIn("metrics", self.model_result)
        self.assertIn("top_features", self.model_result)

        sample_row = self.data["X_test"].iloc[[0]].copy()
        summary = summarize_decision(self.model_result["model"], self.model_result["preprocessor"], sample_row)

        self.assertIn("decision", summary)
        self.assertIn("probability", summary)
        self.assertIn("top_reasons", summary)
        self.assertFalse(summary["stability_filtered"])

    def test_decision_uses_supplied_threshold(self):
        sample_row = self.data["X_test"].iloc[[0]].copy()
        probability = float(
            self.model_result["model"].predict_proba(
                self.model_result["preprocessor"].transform(sample_row)
            )[0, 1]
        )
        summary = summarize_decision(
            self.model_result["model"],
            self.model_result["preprocessor"],
            sample_row,
            threshold=probability + 0.01,
        )

        self.assertEqual(summary["decision"], "APPROVE")
        self.assertEqual(summary["threshold"], probability + 0.01)

    def test_top_reasons_use_real_encoded_feature_names(self):
        positive_row = self.data["X_test"][self.data["y_test"] == 1].iloc[[0]].copy()
        negative_row = self.data["X_test"][self.data["y_test"] == 0].iloc[[0]].copy()

        positive_summary = summarize_decision(self.model_result["model"], self.model_result["preprocessor"], positive_row)
        negative_summary = summarize_decision(self.model_result["model"], self.model_result["preprocessor"], negative_row)

        self.assertNotEqual(
            positive_summary["top_reasons"],
            negative_summary["top_reasons"],
        )

        feature_names = [reason["feature"] for reason in positive_summary["top_reasons"]]
        encoded_names = self.model_result["preprocessor"].get_feature_names_out().tolist()
        self.assertTrue(all(name in encoded_names for name in feature_names))
        self.assertTrue(all(not name.startswith("feature_") for name in feature_names))

        self.assertTrue(
            set(self.model_result["stable_feature_names"]) <= set(encoded_names)
        )

    def test_empty_stability_filter_does_not_fallback_to_raw_reasons(self):
        sample_row = self.data["X_test"].iloc[[0]].copy()
        summary = summarize_decision(
            self.model_result["model"],
            self.model_result["preprocessor"],
            sample_row,
            stable_feature_names=[],
        )

        self.assertTrue(summary["stability_filtered"])
        self.assertEqual(summary["top_reasons"], [])

        appraisal = generate_credit_appraisal_memo(summary)
        self.assertEqual(appraisal["applicant_facing"]["reasons"], [])
        self.assertEqual(
            appraisal["applicant_facing"]["no_high_confidence_factors"],
            "No high-confidence factors identified from the stability-filtered explanation.",
        )

    def test_inference_handles_training_sentinels_and_missing_fields(self):
        applicant = self.data["X_test"].iloc[[0]].copy()
        applicant["num_sub"] = -99999
        result = predict_applicant(self.model_result, applicant)
        self.assertIn(result["decision"], {"APPROVE", "REVIEW", "REJECT"})

        incomplete = applicant.drop(columns=[self.data["feature_columns"][0]])
        with self.assertRaises(ValueError):
            predict_applicant(self.model_result, incomplete)

    def test_applicant_facing_cam_has_no_encoded_names(self):
        sample_row = self.data["X_test"].iloc[[0]].copy()
        summary = summarize_decision(
            self.model_result["model"],
            self.model_result["preprocessor"],
            sample_row,
            threshold=self.model_result["threshold"],
            stable_feature_names=self.model_result["stable_feature_names"],
        )
        appraisal = generate_credit_appraisal_memo(summary)
        applicant_text = " ".join(appraisal["applicant_facing"]["reasons"])
        self.assertNotIn("__", applicant_text)
        self.assertNotIn("numeric__", applicant_text)
        self.assertNotIn("Age_Oldest_TL", applicant_text)
        self.assertTrue(
            all(
                re.search(r"\b(\w+)\s+\1\b", sentence, re.IGNORECASE) is None
                for sentence in appraisal["applicant_facing"]["reasons"]
            ),
            msg=f"Repeated consecutive word in CAM reasons: {appraisal['applicant_facing']['reasons']}",
        )

        mapped_reasons = generate_credit_appraisal_memo(
            {
                "decision": "REVIEW",
                "probability": 0.5,
                "threshold": 0.4,
                "review_threshold": 0.65,
                "top_reasons": [
                    {"feature": feature_name, "direction": "increases rejection risk"}
                    for feature_name in FEATURE_LABEL_MAP
                ],
            }
        )["applicant_facing"]["reasons"]
        self.assertTrue(
            all(re.search(r"\b(\w+)\s+\1\b", sentence, re.IGNORECASE) is None for sentence in mapped_reasons),
            msg=f"Repeated consecutive word in mapped CAM reasons: {mapped_reasons}",
        )
        self.assertTrue(appraisal["applicant_facing"]["summary"])

    def test_top_reasons_only_draw_from_stable_features(self):
        sample_row = self.data["X_test"].iloc[[0]].copy()
        summary = summarize_decision(
            self.model_result["model"],
            self.model_result["preprocessor"],
            sample_row,
            threshold=self.model_result["threshold"],
            stable_feature_names=self.model_result["stable_feature_names"],
        )
        self.assertTrue(all(reason["feature"] in self.model_result["stable_feature_names"] for reason in summary["top_reasons"]))

    def test_structured_cam_contains_all_5c_sections_and_only_stable_reasons(self):
        sample_row = self.data["X_test"].iloc[[0]].copy()
        cam = generate_cam(sample_row, self.model_result)

        self.assertEqual(set(cam["sections"]), FIVE_CATEGORIES)
        self.assertTrue(cam["stability_filtered"])
        section_features = {
            reason["feature"]
            for section in cam["sections"].values()
            for reason in section["reasons"]
        }
        self.assertTrue(section_features <= set(self.model_result["stable_feature_names"]))
        self.assertEqual(
            section_features,
            set(cam["stable_reason_features"]),
        )
        self.assertTrue(all(section["reasons"] or section["empty_message"] for section in cam["sections"].values()))

    def test_evaluation_and_thresholding_outputs_are_valid(self):
        metrics = evaluate_model(
            self.model_result["model"],
            self.data["X_test_processed"],
            self.data["y_test"],
            threshold=self.model_result["threshold"],
        )
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("f2", metrics)
        self.assertIn("predicted_reject_rate", metrics)

        self.assertIn("threshold", self.model_result["validation_threshold_result"])
        self.assertIn("f2", self.model_result["validation_threshold_result"])
        self.assertIn("recall", self.model_result["validation_threshold_result"])
        self.assertIn("cross_validation_metrics", self.model_result)
        self.assertIn("mean_pr_auc", self.model_result["cross_validation_metrics"])
        self.assertIn("mean_ks", self.model_result["cross_validation_metrics"])
        self.assertIn("mean_f2", self.model_result["cross_validation_metrics"])

        _, validation_features, _, validation_labels = train_test_split(
            self.data["X_train_processed"],
            self.data["y_train"],
            test_size=0.2,
            random_state=42,
            stratify=self.data["y_train"],
        )
        threshold_result = find_best_threshold(
            self.model_result["model"], validation_features, validation_labels
        )
        self.assertEqual(threshold_result["threshold"], self.model_result["threshold"])
        self.assertEqual(self.model_result["cross_validation_metrics"]["threshold"], self.model_result["threshold"])

    def test_review_threshold_creates_three_way_decisions(self):
        review_threshold = find_review_threshold(
            self.model_result["model"],
            self.data["X_train_processed"],
            self.data["y_train"],
            primary_threshold=self.model_result["threshold"],
        )
        sample_row = self.data["X_test"].iloc[[0]].copy()
        probability = float(
            self.model_result["model"].predict_proba(
                self.model_result["preprocessor"].transform(sample_row)
            )[0, 1]
        )

        self.assertGreater(review_threshold, self.model_result["threshold"])
        self.assertIn(
            summarize_decision(
                self.model_result["model"],
                self.model_result["preprocessor"],
                sample_row,
                threshold=self.model_result["threshold"],
                review_threshold=review_threshold,
            )["decision"],
            {"APPROVE", "REVIEW", "REJECT"},
        )
        self.assertTrue(probability >= 0.0)

    def test_batch_csv_scoring_reuses_single_applicant_logic(self):
        csv_path = "demo_batch.csv"
        probability_by_row = self.model_result["model"].predict_proba(
            self.model_result["preprocessor"].transform(self.data["X_test"])
        )[:, 1]
        ranked_rows = self.data["X_test"].copy()
        ranked_rows["_probability"] = probability_by_row
        positive_rows = ranked_rows.nlargest(3, "_probability").drop(columns=["_probability"])
        negative_rows = ranked_rows.nsmallest(3, "_probability").drop(columns=["_probability"])
        batch_df = pd.concat([positive_rows, negative_rows], axis=0)
        batch_df.to_csv(csv_path, index=False)

        try:
            results = batch_score_csv(csv_path, self.model_result)
            self.assertEqual(len(results), 6)
            self.assertTrue(all(row["decision"] in {"APPROVE", "REVIEW", "REJECT"} for row in results))
            self.assertTrue(any(row["decision"] == "APPROVE" for row in results))
            self.assertTrue(any(row["decision"] == "REJECT" for row in results))
        finally:
            import os
            if os.path.exists(csv_path):
                os.remove(csv_path)

    def test_batch_csv_normalizes_comma_formatted_numeric_values(self):
        applicant = self.data["X_test"].iloc[[0]].copy()
        numeric_income = float(applicant.iloc[0]["NETMONTHLYINCOME"])
        applicant["NETMONTHLYINCOME"] = applicant["NETMONTHLYINCOME"].astype("object")
        applicant.loc[applicant.index[0], "NETMONTHLYINCOME"] = f"{numeric_income:,.0f}"
        csv_path = Path("comma_income_batch.csv")
        applicant.to_csv(csv_path, index=False)

        try:
            results = batch_score_csv(csv_path, self.model_result)
            expected = predict_applicant(
                self.model_result,
                _normalize_csv_values(applicant.assign(NETMONTHLYINCOME=numeric_income)),
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["decision"], expected["decision"])
            self.assertEqual(results[0]["probability"], expected["probability"])
        finally:
            if csv_path.exists():
                csv_path.unlink()

    def test_model_artifact_round_trip_preserves_scoring_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / "credit_model_v1.joblib"
            save_model_artifact(self.model_result, artifact_path)
            loaded = load_model_artifact(artifact_path)

            self.assertEqual(loaded["raw_feature_columns"], self.model_result["raw_feature_columns"])
            self.assertEqual(loaded["threshold"], self.model_result["threshold"])
            self.assertEqual(loaded["review_threshold"], self.model_result["review_threshold"])

    def test_sample_applicants_csv_scoring(self):
        sample_path = Path("sample_applicants.csv")
        self.assertTrue(sample_path.exists(), "sample_applicants.csv should exist in repository root")
        results = batch_score_csv(sample_path, self.model_result)
        self.assertEqual(len(results), 6)
        decisions = [r["decision"] for r in results]
        self.assertTrue(set(decisions) <= {"APPROVE", "REVIEW", "REJECT"})
        self.assertGreaterEqual(
            sum(decision != "REJECT" for decision in decisions),
            2,
            msg=f"Expected at least two non-REJECT sample applicants, got {decisions}",
        )
        self.assertEqual(results[0].get("applicant_id"), "APP-1001")
        self.assertEqual(results[0].get("profile_name"), "Prime Low-Risk Borrower")

    def test_load_applicant_from_csv_by_index_and_id(self):
        sample_path = Path("sample_applicants.csv")
        # Test loading by row index
        df_0, meta_0 = load_applicant_from_csv(
            sample_path, 0, self.model_result["raw_feature_columns"]
        )
        self.assertEqual(len(df_0), 1)
        self.assertEqual(meta_0["applicant_id"], "APP-1001")
        self.assertEqual(meta_0["row_index"], 0)

        # Test loading by applicant_id
        df_id, meta_id = load_applicant_from_csv(
            sample_path, "APP-1003", self.model_result["raw_feature_columns"]
        )
        self.assertEqual(len(df_id), 1)
        self.assertEqual(meta_id["applicant_id"], "APP-1003")
        self.assertEqual(meta_id["row_index"], 2)

        # Test out of range index raises IndexError
        with self.assertRaises(IndexError):
            load_applicant_from_csv(
                sample_path, 999, self.model_result["raw_feature_columns"]
            )

        # Test invalid applicant_id raises ValueError
        with self.assertRaises(ValueError):
            load_applicant_from_csv(
                sample_path, "NONEXISTENT-ID", self.model_result["raw_feature_columns"]
            )


class SampleCSVTest(unittest.TestCase):
    """Fast tests verifying sample CSV batch scoring and row selection using persisted model."""

    @classmethod
    def setUpClass(cls):
        cls.model_result = load_model_artifact()

    def test_sample_csv_batch_scoring(self):
        sample_path = Path("sample_applicants.csv")
        self.assertTrue(sample_path.exists())
        results = batch_score_csv(sample_path, self.model_result)
        self.assertEqual(len(results), 6)
        decisions = [r["decision"] for r in results]
        self.assertTrue(set(decisions) <= {"APPROVE", "REVIEW", "REJECT"})
        self.assertGreaterEqual(
            sum(decision != "REJECT" for decision in decisions),
            2,
            msg=f"Expected at least two non-REJECT sample applicants, got {decisions}",
        )

    def test_load_applicant_from_csv(self):
        sample_path = Path("sample_applicants.csv")
        df_row, meta = load_applicant_from_csv(sample_path, "APP-1001", self.model_result["raw_feature_columns"])
        self.assertEqual(meta["applicant_id"], "APP-1001")
        self.assertEqual(len(df_row), 1)
        pred = predict_applicant(self.model_result, df_row)
        self.assertEqual(pred["decision"], "APPROVE")


class PreprocessingFairnessTest(unittest.TestCase):
    def test_protected_fields_do_not_appear_in_final_feature_set(self):
        data = build_preprocessed_data()
        feature_names = data["preprocessor"].get_feature_names_out().tolist()

        forbidden = ["GENDER", "MARITALSTATUS", "PROSPECTID", "Approved_Flag"]
        self.assertTrue(
            all(not any(token.lower() in name.lower() for token in forbidden) for name in feature_names),
            msg=f"Protected fields leaked into feature set: {feature_names[:10]}",
        )

    def test_every_raw_model_feature_has_exactly_one_5c_category(self):
        data = build_preprocessed_data()
        validate_feature_mapping(data["feature_columns"])
        self.assertEqual(set(data["feature_columns"]), set(FEATURE_TO_5C))
        self.assertTrue(all(category in FIVE_CATEGORIES for category in FEATURE_TO_5C.values()))
        self.assertEqual(set(data["feature_columns"]), set(FEATURE_TO_5C_RATIONALE))
        self.assertTrue(all(len(reason.split()) >= 8 for reason in FEATURE_TO_5C_RATIONALE.values()))


class SeedStabilityTest(unittest.TestCase):
    def test_kendalls_w_is_one_for_identical_rank_lists(self):
        rank_matrix = pd.DataFrame(
            [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        ).to_numpy()
        self.assertEqual(kendalls_w(rank_matrix), 1.0)

    def test_seed_stability_keeps_dominant_feature_highly_ranked(self):
        from xgboost import XGBClassifier

        rng = np.random.default_rng(42)
        features = pd.DataFrame(
            rng.normal(size=(120, 4)),
            columns=["dominant", "noise_a", "noise_b", "noise_c"],
        )
        labels = (features["dominant"] > 0).astype(int)
        result = compute_seed_stability(
            features,
            labels,
            features.iloc[:40],
            XGBClassifier(
                n_estimators=10,
                max_depth=2,
                eval_metric="logloss",
                n_jobs=1,
            ).get_params(),
            n_seeds=5,
            evaluation_sample_size=40,
        )

        self.assertEqual(result["rank_matrix"].shape, (5, 4))
        self.assertEqual(result["feature_names"], features.columns.tolist())
        self.assertEqual(result["rank_statistics"]["dominant"]["mean_rank"], 1.0)
        self.assertEqual(result["rank_statistics"]["dominant"]["rank_range"], 0)
        self.assertGreaterEqual(result["overall_w"], 0.0)
        self.assertLessEqual(result["overall_w"], 1.0)


if __name__ == "__main__":
    unittest.main()
