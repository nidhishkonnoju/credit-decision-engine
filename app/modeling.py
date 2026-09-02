from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

BOOTSTRAP_STABILITY_RUNS = 5  # majority agreement across independent resamples is a reasonable stability bar.
STABILITY_TOP_K = 5  # keep only the strongest recurring drivers from each resample.
STABILITY_SUPPORT_THRESHOLD = 0.6  # a feature must appear in a majority of runs to be considered stable.


def build_xgboost_model(scale_pos_weight: float) -> XGBClassifier:
    """Build the shared cost-aware model configuration used by training and CV."""
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )


def fit_credit_model(data: dict[str, Any]) -> dict[str, Any]:
    """Train a compact XGBoost model and select threshold on a validation split carved from training data."""
    X_train = data["X_train_processed"]
    X_test = data["X_test_processed"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    X_train_model, X_val, y_train_model, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    pos = float(y_train_model.value_counts().get(1, 0))
    neg = float(y_train_model.value_counts().get(0, 0))
    scale = neg / pos if pos > 0 else 1.0

    model = build_xgboost_model(scale)
    model.fit(X_train_model, y_train_model)

    threshold_result = find_best_threshold(model, X_val, y_val)
    review_threshold = find_review_threshold(model, X_val, y_val, primary_threshold=threshold_result["threshold"])
    final_metrics = evaluate_model(model, X_test, y_test, threshold=threshold_result["threshold"])
    cross_validation_metrics = cross_validate_model(
        X_train,
        y_train,
        threshold=threshold_result["threshold"],
    )

    feature_names = data["preprocessor"].get_feature_names_out().tolist()
    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        raise ValueError(
            f"Feature-name mismatch: {len(feature_names)} encoded names vs {len(importances)} model importances"
        )

    ranked = sorted(
        zip(feature_names, importances),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    stability_result = compute_bootstrap_stability(
        X_train_model,
        y_train_model,
        data["preprocessor"],
    )

    return {
        "model": model,
        "preprocessor": data["preprocessor"],
        "raw_feature_columns": data["feature_columns"],
        "feature_names": feature_names,
        "metrics": final_metrics,
        "threshold": threshold_result["threshold"],
        "review_threshold": review_threshold,
        "validation_threshold_result": threshold_result,
        "cross_validation_metrics": cross_validation_metrics,
        "stable_feature_names": stability_result["stable_features"],
        "stability_scores": stability_result["stability_scores"],
        "filtered_stability_features": stability_result["filtered_out"],
        "top_features": [
            {"feature": name, "importance": round(float(value), 4)}
            for name, value in ranked
        ],
    }


def evaluate_model(model: XGBClassifier, X_test: np.ndarray, y_test: pd.Series, threshold: float = 0.5) -> dict[str, float | int]:
    """Evaluate the model at a selected decision threshold for imbalanced credit risk."""
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "f2": round(float(fbeta_score(y_test, predictions, beta=2, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4),
        "ks": round(float(_ks_statistic(y_test, probabilities)), 4),
        "predicted_reject_rate": round(float(predictions.mean()), 4),
        "threshold": threshold,
    }


def find_best_threshold(model: XGBClassifier, X_val: np.ndarray, y_val: pd.Series, candidate_thresholds: list[float] | None = None) -> dict[str, float]:
    """Search validation data for the best F2 threshold for the minority class.

    F2 weights recall roughly twice as much as precision because the project
    assumes approving a risky applicant is costlier than rejecting a safe one.
    This function must receive validation data, not test data; using the test
    set here biases the final reported performance.
    """
    if candidate_thresholds is None:
        candidate_thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    probabilities = model.predict_proba(X_val)[:, 1]
    best_result: dict[str, float] | None = None
    best_score = -1.0

    for threshold in candidate_thresholds:
        predictions = (probabilities >= threshold).astype(int)
        score = fbeta_score(y_val, predictions, beta=2, zero_division=0)
        if score > best_score:
            best_score = score
            best_result = {
                "threshold": float(threshold),
                "f2": round(float(score), 4),
                "precision": round(float(precision_score(y_val, predictions, zero_division=0)), 4),
                "recall": round(float(recall_score(y_val, predictions, zero_division=0)), 4),
                "predicted_reject_rate": round(float(predictions.mean()), 4),
            }

    if best_result is None:
        raise ValueError("Threshold search produced no valid results.")

    return best_result


def find_review_threshold(
    model: XGBClassifier,
    X_val: np.ndarray,
    y_val: pd.Series,
    primary_threshold: float = 0.5,
    percentile: float = 0.75,
) -> float:
    """Choose a validation-derived upper cutoff for the review band.

    The review band sits above the operational cutoff and below a more conservative reject
    cutoff. We anchor the upper threshold at the 75th percentile of validation scores among
    applicants already above the primary threshold. We also require both a review band and a
    reject band to exist, so the final output stays operationally useful instead of collapsing
    into an empty review queue or an empty reject queue.
    """
    probabilities = model.predict_proba(X_val)[:, 1]
    above_primary = probabilities[probabilities >= primary_threshold]
    if above_primary.size == 0:
        return max(primary_threshold + 0.05, 0.55)

    upper_threshold = float(np.quantile(above_primary, percentile))
    threshold = max(upper_threshold, primary_threshold + 0.05)

    review_mask = (probabilities >= primary_threshold) & (probabilities < threshold)
    reject_mask = probabilities >= threshold
    if review_mask.any() and reject_mask.any():
        return threshold

    for candidate in np.linspace(primary_threshold + 0.05, 0.9, 18):
        review_mask = (probabilities >= primary_threshold) & (probabilities < candidate)
        reject_mask = probabilities >= candidate
        if review_mask.any() and reject_mask.any():
            return float(candidate)
    return float(max(primary_threshold + 0.1, 0.6))


def _ks_statistic(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Return the maximum separation between positive and negative score distributions."""
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
    return float(np.max(true_positive_rate - false_positive_rate))


def compute_bootstrap_stability(
    X_train: np.ndarray,
    y_train: pd.Series,
    preprocessor: Any,
    n_runs: int = BOOTSTRAP_STABILITY_RUNS,
    top_k: int = STABILITY_TOP_K,
    support_threshold: float = STABILITY_SUPPORT_THRESHOLD,
) -> dict[str, Any]:
    """Rank features by bootstrap stability and keep only those supported by a majority of runs."""
    feature_names = preprocessor.get_feature_names_out().tolist()
    support = {name: 0 for name in feature_names}

    for seed in range(n_runs):
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(len(X_train), size=len(X_train), replace=True)
        sample_X = X_train[sample_indices]
        sample_y = y_train.iloc[sample_indices]

        positive_count = float(sample_y.value_counts().get(1, 0))
        negative_count = float(sample_y.value_counts().get(0, 0))
        boot_model = build_xgboost_model(negative_count / positive_count if positive_count else 1.0)
        boot_model.fit(sample_X, sample_y)

        ranked = sorted(
            zip(feature_names, boot_model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        for name, _ in ranked:
            support[name] += 1

    stability_scores = {
        name: round(float(count / n_runs), 4)
        for name, count in support.items()
    }
    stable_features = [
        name
        for name, score in stability_scores.items()
        if score >= support_threshold
    ]
    filtered_out = [name for name in feature_names if name not in stable_features]

    return {
        "stable_features": stable_features,
        "filtered_out": filtered_out,
        "stability_scores": stability_scores,
    }


def cross_validate_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    n_splits: int = 5,
    threshold: float | None = None,
) -> dict[str, float | int]:
    """Evaluate cost-aware XGBoost with stratified folds on training data only.

    The fold-level threshold should match the production operating point, so the
    thresholded metrics are directly comparable to the untouched test-set metrics.
    """
    if threshold is None:
        threshold = 0.5

    fold_metrics: list[dict[str, float | int]] = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_indices, validation_indices in splitter.split(X_train, y_train):
        X_fold_train = X_train[train_indices]
        X_fold_validation = X_train[validation_indices]
        y_fold_train = y_train.iloc[train_indices]
        y_fold_validation = y_train.iloc[validation_indices]

        positive_count = float(y_fold_train.value_counts().get(1, 0))
        negative_count = float(y_fold_train.value_counts().get(0, 0))
        fold_model = build_xgboost_model(
            negative_count / positive_count if positive_count else 1.0
        )
        fold_model.fit(X_fold_train, y_fold_train)
        fold_metrics.append(
            evaluate_model(fold_model, X_fold_validation, y_fold_validation, threshold=threshold)
        )

    metric_names = ["precision", "recall", "f1", "f2", "pr_auc", "roc_auc", "ks"]
    return {
        "folds": n_splits,
        "threshold": float(threshold),
        **{
            f"mean_{metric}": round(float(np.mean([metrics[metric] for metrics in fold_metrics])), 4)
            for metric in metric_names
        },
    }


def summarize_decision(
    model: XGBClassifier,
    preprocessor: Any,
    row: pd.DataFrame,
    threshold: float = 0.5,
    stable_feature_names: list[str] | None = None,
    review_threshold: float | None = None,
) -> dict[str, Any]:
    """Return a per-applicant decision with SHAP-based local reasons, filtered to stable features when provided."""
    processed = preprocessor.transform(row)
    probability = float(model.predict_proba(processed)[0, 1])
    if review_threshold is not None:
        if review_threshold <= threshold:
            raise ValueError("Review threshold must be above the primary decision threshold.")
        if probability < threshold:
            decision = "APPROVE"
        elif probability < review_threshold:
            decision = "REVIEW"
        else:
            decision = "REJECT"
    else:
        decision = "REJECT" if probability >= threshold else "APPROVE"

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(processed)
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1])
    else:
        values = np.asarray(shap_values)

    if values.ndim > 1:
        values = values[0]

    feature_names = preprocessor.get_feature_names_out().tolist()
    if len(values) != len(feature_names):
        raise ValueError(
            f"SHAP feature mismatch: {len(feature_names)} names vs {len(values)} SHAP values"
        )

    top_indices = np.argsort(np.abs(values))[::-1][:5]
    if stable_feature_names:
        stable_set = set(stable_feature_names)
        top_indices = [idx for idx in top_indices if feature_names[idx] in stable_set]
        if not top_indices:
            top_indices = np.argsort(np.abs(values))[::-1][:5]

    top_reasons = [
        {
            "feature": feature_names[idx],
            "contribution": round(float(values[idx]), 4),
            "absolute_contribution": round(float(abs(values[idx])), 4),
            "direction": "increases rejection risk" if values[idx] > 0 else "reduces rejection risk",
        }
        for idx in top_indices
    ]

    return {
        "decision": decision,
        "probability": round(probability, 4),
        "threshold": threshold,
        "review_threshold": review_threshold,
        "top_reasons": top_reasons,
    }
