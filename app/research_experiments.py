"""Reproducible research experiments; separate from production scoring.

The production decision engine deliberately excludes ``AUDITED_LABEL_PROXY_COLUMNS``.
This module compares that audited policy with a proxy-inclusive diagnostic policy;
it never persists or serves a proxy-inclusive model.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from app.modeling import kendalls_w
from app.preprocessing import AUDITED_LABEL_PROXY_COLUMNS

EXPERIMENT_CONDITIONS = {
    "bundled_baseline": {"subsample": 0.8, "colsample_bytree": 0.8, "outer_resample": True},
    "row_sampling": {"subsample": 0.8, "colsample_bytree": 1.0, "outer_resample": False},
    "feature_sampling": {"subsample": 1.0, "colsample_bytree": 0.8, "outer_resample": False},
}
DETERMINISM_CHECK = {"subsample": 1.0, "colsample_bytree": 1.0, "outer_resample": False}


def _descending_ranks(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(-values)) + 1


def _encoded_source_name(encoded_name: str, raw_columns: list[str]) -> str:
    """Map a ColumnTransformer output name back to its raw source field."""
    for raw_column in sorted(raw_columns, key=len, reverse=True):
        if (
            encoded_name == raw_column
            or encoded_name.endswith(f"__{raw_column}")
            or f"__{raw_column}_" in encoded_name
            or f"missingindicator_{raw_column}" in encoded_name
        ):
            return raw_column
    return encoded_name


def _feature_status(encoded_name: str, raw_columns: list[str]) -> dict[str, str]:
    raw_name = _encoded_source_name(encoded_name, raw_columns)
    if raw_name in AUDITED_LABEL_PROXY_COLUMNS:
        return {
            "raw_feature": raw_name,
            "status": "audited_near_perfect_target_proxy",
            "note": "Excluded from production scoring; provenance has not been encoded as a leakage claim.",
        }
    return {"raw_feature": raw_name, "status": "audited_model_feature", "note": "Included by the production feature policy."}


def estimate_univariate_proxy_strength(X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> list[dict[str, Any]]:
    """Estimate per-field target-proxy strength with out-of-fold shallow trees.

    AUC is symmetrised (max(AUC, 1-AUC)), so a feature predictive in either
    direction has a score in [0.5, 1]. This is a diagnostic association measure,
    not evidence that the feature participates in constructing the label.
    """
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=2026)
    results: list[dict[str, Any]] = []
    for column in X.columns:
        one_column = X[[column]].copy()
        numeric = pd.api.types.is_numeric_dtype(one_column[column])
        if numeric:
            prep = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        else:
            prep = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ])
        oof = np.zeros(len(one_column), dtype=float)
        for train_idx, test_idx in splitter.split(one_column, y):
            train_values, test_values = one_column.iloc[train_idx], one_column.iloc[test_idx]
            transformed_train = prep.fit_transform(train_values)
            transformed_test = prep.transform(test_values)
            model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=2026)
            model.fit(transformed_train, y.iloc[train_idx])
            oof[test_idx] = model.predict_proba(transformed_test)[:, 1]
        auc = float(roc_auc_score(y, oof))
        results.append({
            "raw_feature": column,
            "proxy_strength_auc": round(max(auc, 1 - auc), 6),
            "audit_status": "audited_near_perfect_target_proxy" if column in AUDITED_LABEL_PROXY_COLUMNS else "audited_model_feature",
        })
    return sorted(results, key=lambda item: item["proxy_strength_auc"], reverse=True)


def _local_stability(shap_runs: list[np.ndarray], top_k: int = 5) -> list[dict[str, float | int]]:
    """Pairwise local SHAP similarity for the same applicants across model fits."""
    scores: list[dict[str, float | int]] = []
    for applicant_index in range(shap_runs[0].shape[0]):
        vectors = [run[applicant_index] for run in shap_runs]
        jaccards, cosines = [], []
        for left, right in combinations(vectors, 2):
            left_top = set(np.argsort(np.abs(left))[-top_k:])
            right_top = set(np.argsort(np.abs(right))[-top_k:])
            jaccards.append(len(left_top & right_top) / len(left_top | right_top))
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            cosines.append(float(np.dot(left, right) / denominator) if denominator else 1.0)
        scores.append({
            "local_applicant_position": applicant_index,
            "mean_top_k_jaccard": round(float(np.mean(jaccards)), 6),
            "mean_attribution_cosine": round(float(np.mean(cosines)), 6),
        })
    return scores


def run_stability_condition(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_global: np.ndarray,
    X_local: np.ndarray,
    feature_names: list[str],
    condition_name: str,
    n_seeds: int,
    model_params: dict[str, Any],
    n_workers: int = 1,
    return_raw: bool = False,
) -> dict[str, Any]:
    """Run one pre-specified stochastic condition on fixed evaluation samples.

    ``return_raw`` additionally exposes the per-seed global mean-absolute-SHAP
    vectors, per-seed rank vectors, and per-seed local attribution vectors so
    that analyses can be recomputed later without refitting any model. The
    default output is unchanged.
    """
    if condition_name not in EXPERIMENT_CONDITIONS and condition_name != "determinism_check":
        raise ValueError(f"Unknown condition: {condition_name}")
    spec = DETERMINISM_CHECK if condition_name == "determinism_check" else EXPERIMENT_CONDITIONS[condition_name]
    if n_workers < 1:
        raise ValueError("n_workers must be at least 1.")

    def fit_and_explain(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if spec["outer_resample"]:
            selected, _ = train_test_split(
                np.arange(len(y_train)), train_size=0.8, stratify=y_train, random_state=seed
            )
            train_X, train_y = X_train[selected], y_train.iloc[selected]
        else:
            train_X, train_y = X_train, y_train
        model = XGBClassifier(
            **model_params,
            subsample=spec["subsample"],
            colsample_bytree=spec["colsample_bytree"],
            random_state=seed,
            n_jobs=1,
            eval_metric="logloss",
        )
        model.fit(train_X, train_y)
        explainer = shap.TreeExplainer(model)
        # X_local is a prefix of X_global in the runner. One explanation pass
        # therefore supplies both global and local outcomes.
        global_values = np.asarray(explainer.shap_values(X_global))
        if global_values.ndim == 3:
            global_values = global_values[:, :, 1]
        local_values = global_values[: len(X_local)]
        mean_abs = np.mean(np.abs(global_values), axis=0)
        return mean_abs, _descending_ranks(mean_abs), local_values

    if n_workers == 1:
        fitted_runs = [fit_and_explain(seed) for seed in range(n_seeds)]
    else:
        # Every estimator itself uses one thread. Running independent seeds in
        # a bounded thread pool avoids both XGBoost oversubscription and copies
        # of the full design matrix that process-based parallelism would create.
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            fitted_runs = list(executor.map(fit_and_explain, range(n_seeds)))

    global_mean_abs = [run[0] for run in fitted_runs]
    global_ranks = [run[1] for run in fitted_runs]
    local_runs = [run[2] for run in fitted_runs]
    rank_matrix = np.vstack(global_ranks)
    mean_abs_matrix = np.vstack(global_mean_abs)
    mean_rank = rank_matrix.mean(axis=0)
    variance = rank_matrix.var(axis=0)
    ranges = rank_matrix.max(axis=0) - rank_matrix.min(axis=0)
    feature_rows = [
        {
            "encoded_feature": feature_names[index],
            "mean_rank": round(float(mean_rank[index]), 6),
            "rank_variance": round(float(variance[index]), 6),
            "rank_range": int(ranges[index]),
            "stability_score": round(float(1 / (1 + variance[index])), 6),
        }
        for index in range(len(feature_names))
    ]
    result = {
        "condition": condition_name,
        "parameters": {**spec, "n_seeds": n_seeds, "n_jobs": 1, "parallel_seed_workers": n_workers},
        "global": {
            "kendalls_w": round(float(kendalls_w(rank_matrix)), 6),
            "top_5_features": [feature_names[i] for i in np.argsort(mean_rank)[:5]],
            "feature_statistics": feature_rows,
        },
        "local": _local_stability(local_runs),
    }
    if return_raw:
        result["raw"] = {
            # Per-seed mean absolute SHAP value per encoded feature; the exact
            # quantity the per-seed descending ranks are computed from.
            "global_mean_abs_shap": mean_abs_matrix,
            # Per-seed descending rank per encoded feature (1 = largest).
            "global_ranks": rank_matrix,
            # Per-seed local SHAP attribution vectors: (n_seeds, n_local, n_features).
            "local_shap_values": np.stack(local_runs),
        }
    return result
