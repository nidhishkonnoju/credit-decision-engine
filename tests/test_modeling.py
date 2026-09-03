import re
import unittest

import pandas as pd
from sklearn.model_selection import train_test_split

from app.cam import FEATURE_LABEL_MAP, generate_credit_appraisal_memo
from app.inference import batch_score_csv, predict_applicant
from app.modeling import (
    evaluate_model,
    find_best_threshold,
    find_review_threshold,
    fit_credit_model,
    summarize_decision,
)
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


class PreprocessingFairnessTest(unittest.TestCase):
    def test_protected_fields_do_not_appear_in_final_feature_set(self):
        data = build_preprocessed_data()
        feature_names = data["preprocessor"].get_feature_names_out().tolist()

        forbidden = ["GENDER", "MARITALSTATUS", "PROSPECTID", "Approved_Flag"]
        self.assertTrue(
            all(not any(token.lower() in name.lower() for token in forbidden) for name in feature_names),
            msg=f"Protected fields leaked into feature set: {feature_names[:10]}",
        )


if __name__ == "__main__":
    unittest.main()
