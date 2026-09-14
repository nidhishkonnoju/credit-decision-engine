"""Run the pre-registered SHAP reliability research protocol.

Use a small pilot first. The final paper run should use ``--seeds 100`` and
record the package versions, command, and output JSON together.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.preprocessing import build_preprocessed_data
from app.research_experiments import (
    EXPERIMENT_CONDITIONS,
    _feature_status,
    estimate_univariate_proxy_strength,
    run_stability_condition,
)


def _prepare_data(include_proxies: bool) -> dict:
    data = build_preprocessed_data(include_audited_proxies=include_proxies)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SHAP reliability research protocol.")
    parser.add_argument("--seeds", type=int, default=20, help="Use 100 for the final paper run; 20 is a pilot.")
    parser.add_argument("--global-sample", type=int, default=512)
    parser.add_argument("--local-sample", type=int, default=100)
    parser.add_argument("--workers", type=int, default=1, help="Independent seed fits to run concurrently; each fit uses one XGBoost thread.")
    parser.add_argument(
        "--policies", nargs="+", choices=["audited", "proxy_inclusive_diagnostic"],
        default=["proxy_inclusive_diagnostic"],
        help="Feature policies to run. The primary study compares proxy and legitimate fields within the diagnostic policy.",
    )
    parser.add_argument("--determinism-seeds", type=int, default=2, help="Seeds for the sanity check; two identical fits are sufficient.")
    parser.add_argument("--output", type=Path, default=Path("evaluation") / "research_results.json")
    parser.add_argument("--include-determinism-check", action="store_true")
    parser.add_argument(
        "--persist-raw", action="store_true",
        help="Also persist per-seed raw data (global ranks, mean |SHAP|, local attribution vectors) "
        "so stability metrics can be recomputed without refitting models.",
    )
    parser.add_argument("--raw-output", type=Path, default=Path("evaluation") / "research_raw_100_seeds_final.npz")
    parser.add_argument("--raw-manifest", type=Path, default=Path("evaluation") / "research_raw_manifest_final.json")
    args = parser.parse_args()
    if args.seeds < 2:
        raise ValueError("--seeds must be at least 2.")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.determinism_seeds < 2:
        raise ValueError("--determinism-seeds must be at least 2.")
    if args.local_sample > args.global_sample:
        raise ValueError("--local-sample cannot exceed --global-sample because local SHAP values reuse the global sample.")

    policies = {"audited": False, "proxy_inclusive_diagnostic": True}
    all_results = []
    raw_arrays: dict[str, np.ndarray] = {}
    raw_manifest: dict[str, Any] = {
        "top_k": 5,
        "policy": args.policies[0] if len(args.policies) == 1 else args.policies,
        "conditions": {},
        "feature_names": None,
        "applicants": [],
        "note": "Raw arrays are keyed '<policy>__<condition>__<array>' when multiple policies are run.",
    }
    for policy_name in args.policies:
        include_proxies = policies[policy_name]
        data = _prepare_data(include_proxies)
        X_train = data["X_train_processed"]
        X_test = data["X_test_processed"]
        feature_names = data["preprocessor"].get_feature_names_out().tolist()
        positive = float(data["y_train"].sum())
        negative = float(len(data["y_train"]) - positive)
        params = {
            "objective": "binary:logistic", "n_estimators": 300, "max_depth": 5,
            "learning_rate": 0.05, "scale_pos_weight": negative / positive,
        }
        conditions = list(EXPERIMENT_CONDITIONS)
        if args.include_determinism_check:
            conditions.insert(0, "determinism_check")
        local_indices = data["X_test"].index[: args.local_sample].tolist()
        if args.persist_raw:
            raw_manifest["feature_names"] = feature_names
            raw_manifest["applicants"] = [
                {
                    "local_applicant_position": position,
                    "test_row_index": int(test_index),
                    "prospect_id": int(data["df"].loc[test_index, "PROSPECTID"]),
                }
                for position, test_index in enumerate(local_indices)
            ]
        for condition in conditions:
            condition_seeds = args.determinism_seeds if condition == "determinism_check" else args.seeds
            result = run_stability_condition(
                X_train, data["y_train"], X_test[:args.global_sample], X_test[:args.local_sample],
                feature_names, condition, condition_seeds, params,
                n_workers=args.workers,
                return_raw=args.persist_raw,
            )
            result["feature_policy"] = policy_name
            for row in result["global"]["feature_statistics"]:
                row.update(_feature_status(row["encoded_feature"], data["feature_columns"]))
            for local_row, test_index in zip(result["local"], local_indices):
                local_row["test_row_index"] = int(test_index)
                local_row["prospect_id"] = int(data["df"].loc[test_index, "PROSPECTID"])
            if args.persist_raw:
                prefix = condition if len(args.policies) == 1 else f"{policy_name}__{condition}"
                raw_arrays[f"{prefix}__global_mean_abs_shap"] = result["raw"]["global_mean_abs_shap"]
                raw_arrays[f"{prefix}__global_ranks"] = result["raw"]["global_ranks"]
                raw_arrays[f"{prefix}__local_shap_values"] = result["raw"]["local_shap_values"]
                raw_manifest["conditions"][prefix] = {
                    "n_seeds": condition_seeds,
                    "n_local_applicants": len(local_indices),
                    "n_features": len(feature_names),
                    "arrays": {
                        "global_mean_abs_shap": [condition_seeds, len(feature_names)],
                        "global_ranks": [condition_seeds, len(feature_names)],
                        "local_shap_values": [condition_seeds, len(local_indices), len(feature_names)],
                    },
                }
                # Raw numpy arrays are persisted separately (NPZ); the JSON
                # aggregate must remain JSON-serializable.
                result.pop("raw")
            all_results.append(result)
            print(f"Completed {policy_name}: {condition} (W={result['global']['kendalls_w']})")
            del result

    proxy_data = _prepare_data(True)
    output = {
        "protocol_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "research_question": "Can SHAP stability provide misleading evidence of explanation reliability when inputs include audited near-perfect target proxies?",
        "interpretation_guardrail": "Proxy strength is association, not proof of target-construction leakage. The production model remains audited and excludes audited proxy fields.",
        "conditions": EXPERIMENT_CONDITIONS,
        "command_configuration": {
            "n_seeds": args.seeds, "global_sample": args.global_sample,
            "local_sample": args.local_sample, "parallel_seed_workers": args.workers,
            "feature_policies": args.policies, "determinism_seeds": args.determinism_seeds,
        },
        "proxy_strength": estimate_univariate_proxy_strength(proxy_data["X"], proxy_data["y"]),
        "results": all_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Research results written to: {args.output}")

    if args.persist_raw:
        args.raw_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.raw_output, **raw_arrays)
        raw_manifest["raw_output"] = str(args.raw_output)
        raw_manifest["aggregate_output"] = str(args.output)
        raw_manifest["generated_at_utc"] = output["generated_at_utc"]
        args.raw_manifest.write_text(json.dumps(raw_manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Raw per-seed data written to: {args.raw_output}")
        print(f"Raw manifest written to: {args.raw_manifest}")


if __name__ == "__main__":
    main()
