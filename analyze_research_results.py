"""Pre-specified statistical analysis of the committed 100-seed SHAP research results.

This script performs NO model fitting. It reads
``evaluation/research_results_100_seeds.json`` (produced by
``run_research_experiments.py``) and applies the analysis plan that was fixed
before any test was run:

Design (pre-specified)
----------------------
Observational units
- Condition level: Kendall's W (1 value per condition) is DESCRIPTIVE ONLY and
  is never used as a sample for hypothesis testing.
- Feature level: the 122 encoded model columns, identical across conditions,
  labelled proxy (n=61) / non-proxy (n=61) via the audited raw-field policy.
  Paired across conditions.
- Raw-feature level: the 83 raw source fields (35 audited proxies / 48 audited
  model features), used to join proxy_strength_auc and as a sensitivity unit.
- Applicant level: the same 100 fixed test applicants (X_test[:100]) in every
  condition. Paired across conditions.

Families of tests (Holm-Bonferroni within each family)
- F1 (RQ1, encoded level): Mann-Whitney U, proxy vs non-proxy, on mean_rank,
  rank_range, stability_score; 3 conditions x 3 metrics = 9 tests.
  Effect size: rank-biserial r (positive = proxy group larger); 95% bootstrap
  percentile CI over feature resamples (10,000, seed 2026).
- F2 (RQ1 sensitivity, raw level): Mann-Whitney U on raw-aggregated mean_rank
  and stability_score; 3 conditions x 2 metrics = 6 tests.
- F3 (RQ1 association): Spearman rho between raw proxy_strength_auc (n=83) and
  raw-aggregated mean_rank / stability_score per condition; 2 x 3 = 6 tests.
  Two-sided permutation p-values (10,000 permutations, seed 2026).
  This is observational association, NOT causal evidence.
- F4 (RQ1 enrichment): hypergeometric test for over-representation of proxy
  encoded columns among the top-10 / top-20 mean-ranked features per
  condition; 2 x 3 = 6 tests.
- F5 (RQ2, paired feature level): Wilcoxon signed-rank across condition pairs
  on mean_rank, rank_variance, rank_range; 3 pairs x 3 metrics = 9 tests.
  Effect size: matched-pairs rank-biserial r; 95% bootstrap CI of the median
  paired difference (10,000 resamples, seed 2026).
- F6 (RQ3, applicant level): Wilcoxon signed-rank of mean_top_k_jaccard vs
  mean_attribution_cosine within each stochastic condition; 3 tests.
- F7 (RQ3, applicant level): Wilcoxon signed-rank of mean_top_k_jaccard
  across condition pairs; 3 tests.

Global-vs-local correspondence across conditions uses only 3-4 condition-level
data points, so it is reported DESCRIPTIVELY (no p-value is manufactured).
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

RESULTS_PATH = Path("evaluation") / "research_results_100_seeds.json"
OUTPUT_PATH = Path("evaluation") / "research_statistical_analysis.json"
BOOTSTRAP_REPS = 10_000
RNG_SEED = 2026
STOCHASTIC_CONDITIONS = ["bundled_baseline", "row_sampling", "feature_sampling"]
CONDITION_PAIRS = list(combinations(STOCHASTIC_CONDITIONS, 2))

PROXY = "audited_near_perfect_target_proxy"


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (order preserved)."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p_values[idx]
        running_max = max(running_max, value)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def rank_biserial_mwu(group1: np.ndarray, group2: np.ndarray) -> float:
    """Rank-biserial r; positive = group1 tends to have larger values."""
    u1 = stats.mannwhitneyu(group1, group2, alternative="two-sided").statistic
    return 2.0 * float(u1) / (len(group1) * len(group2)) - 1.0



def bootstrap_mwu_ci(group1: np.ndarray, group2: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    estimates = np.empty(BOOTSTRAP_REPS)
    n1, n2 = len(group1), len(group2)
    for i in range(BOOTSTRAP_REPS):
        s1 = rng.choice(group1, size=n1, replace=True)
        s2 = rng.choice(group2, size=n2, replace=True)
        estimates[i] = rank_biserial_mwu(s1, s2)
    return tuple(np.percentile(estimates, [2.5, 97.5]).tolist())


def wilcoxon_rank_biserial(differences: np.ndarray) -> tuple[float, float, float, int]:
    """Matched-pairs rank-biserial r and two-sided p for paired differences.

    Returns (r, statistic_R_plus, p_value, n_nonzero_differences).
    """
    nonzero = differences[differences != 0]
    n = len(nonzero)
    if n == 0:
        return 0.0, 0.0, 1.0, 0
    result = stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
    r_plus = float(result.statistic)
    total = n * (n + 1) / 2.0
    r = (2.0 * r_plus - total) / total
    return r, r_plus, float(result.pvalue), n


def bootstrap_median_diff_ci(differences: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(RNG_SEED)
    n = len(differences)
    medians = np.median(rng.choice(differences, size=(BOOTSTRAP_REPS, n), replace=True), axis=1)
    return tuple(np.percentile(medians, [2.5, 97.5]).tolist())


def permutation_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    rho = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(RNG_SEED)
    count = 0
    for _ in range(BOOTSTRAP_REPS):
        permuted = rng.permutation(x)
        if abs(float(stats.spearmanr(permuted, y).statistic)) >= abs(rho):
            count += 1
    return rho, (count + 1) / (BOOTSTRAP_REPS + 1)


def main() -> None:
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    results = {r["condition"]: r for r in data["results"]}
    proxy_auc = {p["raw_feature"]: p["proxy_strength_auc"] for p in data["proxy_strength"]}

    feature_frame: dict[str, dict[str, dict[str, float]]] = {}
    for condition, payload in results.items():
        rows: dict[str, dict[str, float]] = {}
        for row in payload["global"]["feature_statistics"]:
            rows[row["encoded_feature"]] = {
                "raw_feature": row["raw_feature"],
                "status": row["status"],
                "mean_rank": row["mean_rank"],
                "rank_variance": row["rank_variance"],
                "rank_range": row["rank_range"],
                "stability_score": row["stability_score"],
            }
        feature_frame[condition] = rows

    encoded_features = list(feature_frame[STOCHASTIC_CONDITIONS[0]].keys())
    proxy_encoded = [f for f in encoded_features if feature_frame[STOCHASTIC_CONDITIONS[0]][f]["status"] == PROXY]
    nonproxy_encoded = [f for f in encoded_features if feature_frame[STOCHASTIC_CONDITIONS[0]][f]["status"] != PROXY]

    raw_features = sorted({row["raw_feature"] for row in feature_frame[STOCHASTIC_CONDITIONS[0]].values()})
    raw_status = {row["raw_feature"]: row["status"] for row in feature_frame[STOCHASTIC_CONDITIONS[0]].values()}

    def raw_aggregate(condition: str, metric: str) -> dict[str, float]:
        sums: dict[str, list[float]] = {raw: [] for raw in raw_features}
        for row in feature_frame[condition].values():
            sums[row["raw_feature"]].append(row[metric])
        return {raw: float(np.mean(values)) for raw, values in sums.items()}

    output: dict[str, object] = {
        "source_file": str(RESULTS_PATH),
        "units": {
            "encoded_features": len(encoded_features),
            "proxy_encoded": len(proxy_encoded),
            "nonproxy_encoded": len(nonproxy_encoded),
            "raw_features": len(raw_features),
            "proxy_raw": sum(1 for r in raw_features if raw_status[r] == PROXY),
            "local_applicants": len(results[STOCHASTIC_CONDITIONS[0]]["local"]),
            "seeds_per_stochastic_condition": results[STOCHASTIC_CONDITIONS[0]]["parameters"]["n_seeds"],
        },
        "families": {},
    }

    # ------------------------------------------------------------------
    # F1: encoded-level proxy vs non-proxy (Mann-Whitney U), per condition
    # ------------------------------------------------------------------
    f1: list[dict[str, object]] = []
    for condition in STOCHASTIC_CONDITIONS:
        for metric in ["mean_rank", "rank_range", "stability_score"]:
            proxy_values = np.array([feature_frame[condition][f][metric] for f in proxy_encoded])
            non_values = np.array([feature_frame[condition][f][metric] for f in nonproxy_encoded])
            stat = stats.mannwhitneyu(proxy_values, non_values, alternative="two-sided")
            effect = rank_biserial_mwu(proxy_values, non_values)
            ci_low, ci_high = bootstrap_mwu_ci(proxy_values, non_values)
            f1.append({
                "family": "F1_encoded_proxy_vs_nonproxy",
                "condition": condition,
                "metric": metric,
                "unit": "encoded feature",
                "n_proxy": len(proxy_values),
                "n_nonproxy": len(non_values),
                "median_proxy": float(np.median(proxy_values)),
                "median_nonproxy": float(np.median(non_values)),
                "u_statistic": float(stat.statistic),
                "p_value": float(stat.pvalue),
                "effect_rank_biserial": effect,
                "effect_ci95": [ci_low, ci_high],
                "effect_direction": "positive = proxy group has larger values",
            })
    for t, p_adj in zip(f1, holm([t["p_value"] for t in f1])):  # type: ignore[index]
        t["p_holm"] = p_adj  # type: ignore[index]
    output["families"]["F1"] = f1  # type: ignore[index]

    # ------------------------------------------------------------------
    # F2: raw-level sensitivity (Mann-Whitney U on raw aggregates)
    # ------------------------------------------------------------------
    proxy_raw = [r for r in raw_features if raw_status[r] == PROXY]
    nonproxy_raw = [r for r in raw_features if raw_status[r] != PROXY]
    f2: list[dict[str, object]] = []
    for condition in STOCHASTIC_CONDITIONS:
        for metric in ["mean_rank", "stability_score"]:
            aggregated = raw_aggregate(condition, metric)
            proxy_values = np.array([aggregated[r] for r in proxy_raw])
            non_values = np.array([aggregated[r] for r in nonproxy_raw])
            stat = stats.mannwhitneyu(proxy_values, non_values, alternative="two-sided")
            effect = rank_biserial_mwu(proxy_values, non_values)
            ci_low, ci_high = bootstrap_mwu_ci(proxy_values, non_values)
            f2.append({
                "family": "F2_raw_proxy_vs_nonproxy",
                "condition": condition,
                "metric": metric,
                "unit": "raw feature",
                "n_proxy": len(proxy_values),
                "n_nonproxy": len(non_values),
                "median_proxy": float(np.median(proxy_values)),
                "median_nonproxy": float(np.median(non_values)),
                "u_statistic": float(stat.statistic),
                "p_value": float(stat.pvalue),
                "effect_rank_biserial": effect,
                "effect_ci95": [ci_low, ci_high],
            })
    for t, p_adj in zip(f2, holm([t["p_value"] for t in f2])):  # type: ignore[index]
        t["p_holm"] = p_adj  # type: ignore[index]
    output["families"]["F2"] = f2  # type: ignore[index]

    # ------------------------------------------------------------------
    # F3: association proxy_strength_auc vs raw-aggregated metrics
    # ------------------------------------------------------------------
    auc_vector = np.array([proxy_auc[r] for r in raw_features])
    f3: list[dict[str, object]] = []
    for condition in STOCHASTIC_CONDITIONS:
        for metric in ["mean_rank", "stability_score"]:
            aggregated = np.array([raw_aggregate(condition, metric)[r] for r in raw_features])
            rho, p_value = permutation_spearman(auc_vector, aggregated)
            f3.append({
                "family": "F3_spearman_proxy_auc",
                "condition": condition,
                "metric": metric,
                "unit": "raw feature",
                "n": len(raw_features),
                "spearman_rho": rho,
                "p_permutation": p_value,
                "note": "observational association; not causal evidence",
            })
    for t, p_adj in zip(f3, holm([t["p_permutation"] for t in f3])):  # type: ignore[index]
        t["p_holm"] = p_adj  # type: ignore[index]
    output["families"]["F3"] = f3  # type: ignore[index]

    # ------------------------------------------------------------------
    # F4: hypergeometric enrichment of proxies among top-k mean ranks
    # ------------------------------------------------------------------
    f4: list[dict[str, object]] = []
    universe = len(encoded_features)
    proxy_count = len(proxy_encoded)
    for condition in STOCHASTIC_CONDITIONS:
        ordered = sorted(encoded_features, key=lambda f: feature_frame[condition][f]["mean_rank"])
        for top_k in (10, 20):
            top = ordered[:top_k]
            observed = sum(1 for f in top if feature_frame[condition][f]["status"] == PROXY)
            p_value = float(stats.hypergeom.sf(observed - 1, universe, proxy_count, top_k))
            f4.append({
                "family": "F4_topk_enrichment",
                "condition": condition,
                "top_k": top_k,
                "unit": "encoded feature",
                "universe": universe,
                "proxy_in_universe": proxy_count,
                "proxy_in_topk": observed,
                "expected_if_neutral": proxy_count * top_k / universe,
                "p_value": p_value,
            })
    for t, p_adj in zip(f4, holm([t["p_value"] for t in f4])):  # type: ignore[index]
        t["p_holm"] = p_adj  # type: ignore[index]
    output["families"]["F4"] = f4  # type: ignore[index]

    # ------------------------------------------------------------------
    # F5: paired feature-level comparison across conditions (Wilcoxon)
    # ------------------------------------------------------------------
    f5: list[dict[str, object]] = []
    for left, right in CONDITION_PAIRS:
        for metric in ["mean_rank", "rank_variance", "rank_range"]:
            differences = np.array(
                [feature_frame[right][f][metric] - feature_frame[left][f][metric] for f in encoded_features]
            )
            effect, _, p_value, n_nonzero = wilcoxon_rank_biserial(differences)
            ci_low, ci_high = bootstrap_median_diff_ci(differences)
            f5.append({
                "family": "F5_paired_feature_level",
                "comparison": f"{left} vs {right}",
                "metric": metric,
                "unit": "encoded feature (paired across conditions)",
                "n_pairs": len(encoded_features),
                "n_nonzero_differences": n_nonzero,
                "median_difference_right_minus_left": float(np.median(differences)),
                "median_diff_ci95": [ci_low, ci_high],
                "p_value": p_value,
                "effect_rank_biserial": effect,
                "effect_direction": "positive = right condition larger",
            })
    for t, p_adj in zip(f5, holm([t["p_value"] for t in f5])):  # type: ignore[index]
        t["p_holm"] = p_adj  # type: ignore[index]
    output["families"]["F5"] = f5  # type: ignore[index]

    # ------------------------------------------------------------------
    # F6/F7: applicant-level local stability (paired)
    # ------------------------------------------------------------------
    local_frame = {
        condition: {
            "jaccard": np.array([row["mean_top_k_jaccard"] for row in results[condition]["local"]]),
            "cosine": np.array([row["mean_attribution_cosine"] for row in results[condition]["local"]]),
        }
        for condition in STOCHASTIC_CONDITIONS
    }
    f6: list[dict[str, object]] = []
    for condition in STOCHASTIC_CONDITIONS:
        differences = local_frame[condition]["cosine"] - local_frame[condition]["jaccard"]
        effect, _, p_value, n_nonzero = wilcoxon_rank_biserial(differences)
        ci_low, ci_high = bootstrap_median_diff_ci(differences)
        f6.append({
            "family": "F6_jaccard_vs_cosine",
            "condition": condition,
            "unit": "applicant (paired metrics)",
            "n": len(differences),
            "median_jaccard": float(np.median(local_frame[condition]["jaccard"])),
            "median_cosine": float(np.median(local_frame[condition]["cosine"])),
            "median_difference_cosine_minus_jaccard": float(np.median(differences)),
            "median_diff_ci95": [ci_low, ci_high],
            "p_value": p_value,
            "effect_rank_biserial": effect,
            "n_nonzero_differences": n_nonzero,
        })
    for t, p_adj in zip(f6, holm([t["p_value"] for t in f6])):  # type: ignore[index]
        t["p_holm"] = p_adj  # type: ignore[index]
    output["families"]["F6"] = f6  # type: ignore[index]

    f7: list[dict[str, object]] = []
    for left, right in CONDITION_PAIRS:
        differences = local_frame[right]["jaccard"] - local_frame[left]["jaccard"]
        effect, _, p_value, n_nonzero = wilcoxon_rank_biserial(differences)
        ci_low, ci_high = bootstrap_median_diff_ci(differences)
        f7.append({
            "family": "F7_paired_local_jaccard",
            "comparison": f"{left} vs {right}",
            "unit": "applicant (paired across conditions)",
            "n_pairs": len(differences),
            "mean_jaccard_left": float(local_frame[left]["jaccard"].mean()),
            "mean_jaccard_right": float(local_frame[right]["jaccard"].mean()),
            "median_difference_right_minus_left": float(np.median(differences)),
            "median_diff_ci95": [ci_low, ci_high],
            "p_value": p_value,
            "effect_rank_biserial": effect,
            "n_nonzero_differences": n_nonzero,
        })
    for t, p_adj in zip(f7, holm([t["p_value"] for t in f7])):  # type: ignore[index]
        t["p_holm"] = p_adj  # type: ignore[index]
    output["families"]["F7"] = f7  # type: ignore[index]

    # ------------------------------------------------------------------
    # Descriptive global-vs-local table (no test: too few conditions)
    # ------------------------------------------------------------------
    output["global_vs_local_descriptive"] = [  # type: ignore[index]
        {
            "condition": condition,
            "kendalls_w": results[condition]["global"]["kendalls_w"],
            "mean_local_jaccard": float(local_frame[condition]["jaccard"].mean()),
            "mean_local_cosine": float(local_frame[condition]["cosine"].mean()),
        }
        for condition in STOCHASTIC_CONDITIONS
    ]

    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Statistical analysis written to: {OUTPUT_PATH}")
    families: dict[str, list[dict[str, object]]] = output["families"]  # type: ignore[assignment]
    for family, tests in families.items():
        significant = sum(
            1 for t in tests if t.get("p_holm", t.get("p_value", 1.0)) < 0.05
        )
        print(f"{family}: {len(tests)} tests, {significant} significant after Holm (alpha=0.05)")


if __name__ == "__main__":
    main()

