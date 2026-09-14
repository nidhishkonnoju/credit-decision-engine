"""Recompute and validate local stability metrics from persisted raw per-seed data.

Reads ``evaluation/research_raw_100_seeds_final.npz`` (per-seed rank matrices,
mean |SHAP| matrices, and local attribution vectors written by
``run_research_experiments.py --persist-raw``) and:

1. Recomputes per-applicant top-5 |SHAP| sets, pairwise-seed Jaccard, and
   attribution cosine for every applicant in every condition.
2. Verifies the recomputed values match the aggregate JSON stored alongside
   the run (consistency check).
3. Verifies the aggregate global feature statistics reproduce from the raw
   rank matrices.
4. Runs pre-stated degeneracy diagnostics so that a metric which does not
   vary across applicants is flagged and excluded from applicant-level
   inference rather than reported as evidence of individual-level behaviour.

Degeneracy rule (fixed before inspection): a condition-metric is degenerate
for applicant-level inference if the across-applicant standard deviation of
the per-applicant mean value is < 1e-9 (numerically constant across all
applicants). The mechanism diagnostic - how many seeds have a single
applicant-invariant top-5 set - is reported alongside.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

RAW_PATH = Path("evaluation") / "research_raw_100_seeds_final.npz"
MANIFEST_PATH = Path("evaluation") / "research_raw_manifest_final.json"
AGGREGATE_PATH = Path("evaluation") / "research_results_100_seeds_final.json"
OUTPUT_PATH = Path("evaluation") / "research_local_recomputed_final.json"
TOP_K = 5
DEGENERACY_SD_THRESHOLD = 1e-9


def applicant_metrics(local_shap: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-applicant mean pairwise-seed Jaccard and cosine from raw vectors."""
    n_seeds, n_applicants, _ = local_shap.shape
    abs_shap = np.abs(local_shap)
    # Top-k sets per seed per applicant: (n_seeds, n_applicants, top_k).
    topk = np.argsort(abs_shap, axis=2)[:, :, -top_k:]
    jaccard_means = np.empty(n_applicants)
    cosine_means = np.empty(n_applicants)
    for applicant in range(n_applicants):
        sets = [frozenset(topk[s, applicant].tolist()) for s in range(n_seeds)]
        vectors = local_shap[:, applicant, :]
        jaccards, cosines = [], []
        for left, right in combinations(range(n_seeds), 2):
            union = sets[left] | sets[right]
            jaccards.append(len(sets[left] & sets[right]) / len(union) if union else 1.0)
            denominator = float(np.linalg.norm(vectors[left]) * np.linalg.norm(vectors[right]))
            cosines.append(float(np.dot(vectors[left], vectors[right]) / denominator) if denominator else 1.0)
        jaccard_means[applicant] = float(np.mean(jaccards))
        cosine_means[applicant] = float(np.mean(cosines))
    return jaccard_means, cosine_means


def per_seed_distinct_sets(local_shap: np.ndarray, top_k: int) -> list[int]:
    """For each seed, how many distinct top-k sets exist across applicants."""
    abs_shap = np.abs(local_shap)
    topk = np.argsort(abs_shap, axis=2)[:, :, -top_k:]
    return [
        len({frozenset(topk[s, a].tolist()) for a in range(topk.shape[1])})
        for s in range(topk.shape[0])
    ]



def metric_block(values: np.ndarray) -> dict:
    distinct = sorted({round(float(v), 6) for v in values})
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "sd_across_applicants": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "n_distinct_applicant_means_6dp": len(distinct),
        "degenerate": bool(values.std(ddof=0) < DEGENERACY_SD_THRESHOLD),
    }


def main() -> None:
    raw = np.load(RAW_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    aggregate = json.loads(AGGREGATE_PATH.read_text(encoding="utf-8"))
    agg_by_condition = {r["condition"]: r for r in aggregate["results"]}
    feature_names = manifest["feature_names"]

    output: dict[str, object] = {
        "source_raw": str(RAW_PATH),
        "source_aggregate": str(AGGREGATE_PATH),
        "top_k": TOP_K,
        "degeneracy_rule": f"across-applicant SD < {DEGENERACY_SD_THRESHOLD:g}",
        "conditions": {},
    }

    for prefix, info in manifest["conditions"].items():
        local_shap = raw[f"{prefix}__local_shap_values"]
        ranks = raw[f"{prefix}__global_ranks"]
        mean_abs = raw[f"{prefix}__global_mean_abs_shap"]
        n_seeds, n_applicants, n_features = local_shap.shape
        assert n_features == len(feature_names)
        assert info["arrays"]["local_shap_values"] == [n_seeds, n_applicants, n_features]

        jaccard_means, cosine_means = applicant_metrics(local_shap, TOP_K)
        distinct_sets = per_seed_distinct_sets(local_shap, TOP_K)

        agg_result = agg_by_condition[prefix]
        stored_jaccard = np.array([row["mean_top_k_jaccard"] for row in agg_result["local"]])
        stored_cosine = np.array([row["mean_attribution_cosine"] for row in agg_result["local"]])
        stored_stats = {row["encoded_feature"]: row for row in agg_result["global"]["feature_statistics"]}
        recomputed_mean_rank = ranks.mean(axis=0)
        recomputed_variance = ranks.var(axis=0)
        max_mean_rank_diff = 0.0
        max_variance_diff = 0.0
        for index, name in enumerate(feature_names):
            max_mean_rank_diff = max(max_mean_rank_diff, abs(stored_stats[name]["mean_rank"] - recomputed_mean_rank[index]))
            max_variance_diff = max(max_variance_diff, abs(stored_stats[name]["rank_variance"] - recomputed_variance[index]))
        # Ranks must be reproducible from mean |SHAP|.
        assert np.array_equal(np.argsort(np.argsort(-mean_abs, axis=1), axis=1) + 1, ranks)

        output["conditions"][prefix] = {  # type: ignore[index]
            "n_seeds": n_seeds,
            "n_applicants": n_applicants,
            "n_features": n_features,
            "jaccard": metric_block(jaccard_means),
            "cosine": metric_block(cosine_means),
            "per_seed_distinct_topk_sets": {
                "min": int(min(distinct_sets)),
                "max": int(max(distinct_sets)),
                "n_seeds_with_single_applicant_invariant_set": int(sum(1 for d in distinct_sets if d == 1)),
            },
            "consistency_with_aggregate": {
                "jaccard_max_abs_diff": float(np.max(np.abs(jaccard_means - stored_jaccard))),
                "cosine_max_abs_diff": float(np.max(np.abs(cosine_means - stored_cosine))),
                "global_mean_rank_max_abs_diff": float(max_mean_rank_diff),
                "global_rank_variance_max_abs_diff": float(max_variance_diff),
            },
        }
        print(f"{prefix}: jaccard mean={jaccard_means.mean():.6f} sd={jaccard_means.std():.2e} "
              f"distinct={len({round(v, 6) for v in jaccard_means})} | "
              f"cosine mean={cosine_means.mean():.6f} sd={cosine_means.std():.2e} | "
              f"seeds with single top-5 set: {sum(1 for d in distinct_sets if d == 1)}/{n_seeds}")

    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Recomputed local analysis written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

