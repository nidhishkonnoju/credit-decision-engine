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

SEED_STABILITY_RUNS = 100
STABILITY_TOP_K = 10
STABILITY_RANK_RANGE_MAX = 3
STABILITY_DIVERSITY_BAND_SIZE = 6
STABILITY_EVALUATION_SAMPLE_SIZE = 512


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
        n_jobs=1,
        eval_metric="logloss",
    )


def fit_credit_model(data: dict[str, Any]) -> dict[str, Any]:
    """Train a compact XGBoost model and select threshold on a validation split carved from training data."""
    from app.cam import validate_feature_mapping

    validate_feature_mapping(data["feature_columns"])
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
    stability_result = compute_seed_stability(
        X_train_model,
        y_train_model,
        X_test,
        {
            **build_xgboost_model(scale).get_params(),
            "feature_names": feature_names,
        },
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
        "stability_result": stability_result,
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


def kendalls_w(rank_matrix: np.ndarray) -> float:
    """Calculate Kendall's coefficient of concordance for rank lists."""
    ranks = np.asarray(rank_matrix, dtype=float)
    if ranks.ndim != 2:
        raise ValueError("rank_matrix must be a two-dimensional array.")
    n_models, n_features = ranks.shape
    if n_models < 2 or n_features < 2:
        return 1.0

    rank_sums = ranks.sum(axis=0)
    expected_sum = n_models * (n_features + 1) / 2
    disagreement = float(np.sum((rank_sums - expected_sum) ** 2))
    denominator = n_models**2 * (n_features**3 - n_features)
    return float(np.clip(12 * disagreement / denominator, 0.0, 1.0))


def _rerank_columns(rank_matrix: np.ndarray, column_indices: np.ndarray) -> np.ndarray:
    """Rerank a selected feature subset within each model's rank list."""
    selected = rank_matrix[:, column_indices]
    return np.argsort(np.argsort(selected, axis=1), axis=1) + 1


def compute_seed_stability(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_test: np.ndarray,
    base_params: dict[str, Any],
    n_seeds: int = SEED_STABILITY_RUNS,
    evaluation_sample_size: int = STABILITY_EVALUATION_SAMPLE_SIZE,
) -> dict[str, Any]:
    """Replicate Lin & Wang (2025)'s seed-only SHAP stability procedure.

    Every model receives the same X_train/y_train values and the same fixed prefix
    of X_test for SHAP evaluation. Only random_state changes between models.
    A feature is memo-eligible when it is among the top 10 mean-ranked features
    and its rank range across seeds is at most 3.
    """
    if n_seeds < 2:
        raise ValueError("n_seeds must be at least 2 to calculate rank concordance.")

    params = dict(base_params)
    configured_feature_names = params.pop("feature_names", None)
    params.pop("random_state", None)
    params["n_jobs"] = params.get("n_jobs", 1)
    feature_names = configured_feature_names or [f"feature_{index}" for index in range(X_train.shape[1])]
    if configured_feature_names is None and hasattr(X_train, "columns"):
        feature_names = list(X_train.columns)

    evaluation_sample = X_test[:evaluation_sample_size]
    rank_lists: list[np.ndarray] = []
    for seed in range(n_seeds):
        model = XGBClassifier(**params, random_state=seed)
        model.fit(X_train, y_train)
        shap_values = shap.TreeExplainer(model).shap_values(evaluation_sample)
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[1])
        else:
            values = np.asarray(shap_values)
        if values.ndim == 3:
            values = values[:, :, 1]
        mean_absolute_shap = np.mean(np.abs(values), axis=0)
        rank_lists.append(np.argsort(np.argsort(-mean_absolute_shap)) + 1)

    rank_matrix = np.vstack(rank_lists)
    mean_ranks = rank_matrix.mean(axis=0)
    rank_variances = rank_matrix.var(axis=0)
    rank_ranges = rank_matrix.max(axis=0) - rank_matrix.min(axis=0)

    top_5_indices = np.argsort(mean_ranks)[:5]
    diversity_indices = np.argsort(rank_variances)[::-1][:STABILITY_DIVERSITY_BAND_SIZE]
    stable_indices = np.where(
        (np.argsort(np.argsort(mean_ranks)) < STABILITY_TOP_K)
        & (rank_ranges <= STABILITY_RANK_RANGE_MAX)
    )[0]
    stable_features = [feature_names[index] for index in stable_indices]
    filtered_out = [name for name in feature_names if name not in stable_features]
    rank_statistics = {
        name: {
            "mean_rank": round(float(mean_ranks[index]), 4),
            "rank_variance": round(float(rank_variances[index]), 4),
            "rank_range": int(rank_ranges[index]),
        }
        for index, name in enumerate(feature_names)
    }

    return {
        "stable_features": stable_features,
        "filtered_out": filtered_out,
        "stability_scores": {
            name: round(float(1 / (1 + rank_variances[index])), 4)
            for index, name in enumerate(feature_names)
        },
        "rank_matrix": rank_matrix,
        "feature_names": feature_names,
        "rank_statistics": rank_statistics,
        "overall_w": kendalls_w(rank_matrix),
        "top_5_features": [feature_names[index] for index in top_5_indices],
        "top_5_w": kendalls_w(_rerank_columns(rank_matrix, top_5_indices)),
        "diversity_band_features": [feature_names[index] for index in diversity_indices],
        "diversity_band_w": kendalls_w(_rerank_columns(rank_matrix, diversity_indices)),
        "n_seeds": n_seeds,
        "evaluation_sample_size": min(evaluation_sample_size, len(X_test)),
        "stability_rule": "top 10 mean-ranked features with rank range <= 3 across seeds",
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

    stability_filtered = stable_feature_names is not None
    if stability_filtered:
        stable_set = set(stable_feature_names)
        unknown_stable_features = stable_set.difference(feature_names)
        if unknown_stable_features:
            raise ValueError(
                "Stable feature names must match encoded preprocessor names: "
                f"{sorted(unknown_stable_features)}"
            )
        top_indices = np.argsort(np.abs(values))[::-1][:5]
        top_indices = [idx for idx in top_indices if feature_names[idx] in stable_set]
    else:
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
        "stability_filtered": stability_filtered,
        "top_reasons": top_reasons,
    }
