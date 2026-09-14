import numpy as np
import pandas as pd

from app.research_experiments import _feature_status, estimate_univariate_proxy_strength, run_stability_condition


def test_proxy_strength_identifies_dominant_synthetic_feature():
    rng = np.random.default_rng(7)
    dominant = rng.normal(size=180)
    X = pd.DataFrame({"dominant": dominant, "noise": rng.normal(size=180)})
    y = pd.Series((dominant > 0).astype(int))

    results = estimate_univariate_proxy_strength(X, y)

    assert results[0]["raw_feature"] == "dominant"
    assert results[0]["proxy_strength_auc"] > 0.9


def test_determinism_check_has_perfect_global_rank_agreement():
    rng = np.random.default_rng(11)
    X_train = rng.normal(size=(100, 3))
    y_train = pd.Series((X_train[:, 0] > 0).astype(int))
    X_eval = X_train[:20]
    result = run_stability_condition(
        X_train, y_train, X_eval, X_eval[:10], ["dominant", "noise_a", "noise_b"],
        "determinism_check", 3,
        {"objective": "binary:logistic", "n_estimators": 10, "max_depth": 2, "learning_rate": 0.2},
    )

    assert result["global"]["kendalls_w"] == 1.0
    assert len(result["local"]) == 10


def test_missingness_indicator_inherits_proxy_audit_status():
    status = _feature_status("numeric__missingindicator_Credit_Score", ["Credit_Score", "income"])
    assert status["raw_feature"] == "Credit_Score"
    assert status["status"] == "audited_near_perfect_target_proxy"


def test_return_raw_exposes_recomputable_per_seed_data():
    rng = np.random.default_rng(11)
    X_train = rng.normal(size=(100, 3))
    y_train = pd.Series((X_train[:, 0] > 0).astype(int))
    X_eval = X_train[:20]
    result = run_stability_condition(
        X_train, y_train, X_eval, X_eval[:10], ["dominant", "noise_a", "noise_b"],
        "row_sampling", 3,
        {"objective": "binary:logistic", "n_estimators": 10, "max_depth": 2, "learning_rate": 0.2},
        return_raw=True,
    )

    raw = result["raw"]
    assert raw["global_ranks"].shape == (3, 3)
    assert raw["global_mean_abs_shap"].shape == (3, 3)
    assert raw["local_shap_values"].shape == (3, 10, 3)
    # Ranks must be reproducible from the stored mean |SHAP| values.
    recomputed = np.argsort(np.argsort(-raw["global_mean_abs_shap"], axis=1), axis=1) + 1
    assert np.array_equal(recomputed, raw["global_ranks"])
    # Aggregate feature statistics must be reproducible from the raw ranks.
    mean_rank = raw["global_ranks"].mean(axis=0)
    for index, row in enumerate(result["global"]["feature_statistics"]):
        assert abs(row["mean_rank"] - mean_rank[index]) < 1e-6
