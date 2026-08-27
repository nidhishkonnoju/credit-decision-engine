import unittest

from sklearn.model_selection import train_test_split

from app.modeling import evaluate_model, fit_credit_model, find_best_threshold, summarize_decision
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
