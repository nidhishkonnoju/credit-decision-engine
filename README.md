# Credit Decision Engine

A Python-based credit underwriting decision-support project that trains an explainable XGBoost model, filters protected and label-proxy fields, and produces stability-filtered 5C memo reasons.

## Overview

This project builds a compact credit decision pipeline for a bank-style underwriting workflow:

- Loads internal bank-data and bureau-data tables from the `Dataset` folder.
- Cleans numeric sentinels and removes protected / audit-sensitive fields.
- Trains a cost-aware XGBoost classifier for binary default-risk labeling.
- Selects a threshold using validation F2 performance to prioritize recall for risky applicants.
- Scores a single applicant from JSON input and returns:
  - decision (`APPROVE`, `REVIEW`, or `REJECT`)
  - probability score
  - threshold used
  - top SHAP features driving the decision

The implementation is intentionally lightweight and designed to be easy to run locally for experimentation and model validation.

## Repository structure

- `app/` — model, preprocessing, and inference logic
  - `app/preprocessing.py` — dataset loading, target construction, missing-value handling, feature exclusions
  - `app/modeling.py` — training, metrics, threshold selection, SHAP summarization
  - `app/inference.py` — JSON applicant loading and decision generation
  - `app/persistence.py` — versioned model-artifact save/load helpers
  - `app/cam.py` — 5C mapping and structured stability-filtered CAM generation
- `tests/` — smoke tests for modeling and fairness-related feature exclusions
- `Dataset/` — expected input data files (`External_Cibil_Dataset.xlsx` and `Internal_Bank_Dataset.xlsx`)
- `main.py` — training entry point and CLI runner
- `evaluate_stability_filter.py` — deterministic Phase 4 ablation and coverage evaluation
- `evaluation/` — fixed applicant sample and recorded evaluation results
- `applicant.example.json` — sample applicant payload for local scoring
- `requirements.txt` — project dependencies
- `artifacts/` — locally generated persisted model artifacts (ignored by Git)

## Environment setup

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data requirements

The project expects these files to exist under the `Dataset` directory:

- `External_Cibil_Dataset.xlsx`
- `Internal_Bank_Dataset.xlsx`

The pipeline uses a merged applicant-level dataset keyed by `PROSPECTID` and excludes fields such as `PROSPECTID`, `GENDER`, `MARITALSTATUS`, and known label-proxy features from the model input.

## Running the project

Train the model and print summary metrics:

```powershell
python .\main.py --retrain
```

After the first training run, normal commands load `artifacts/credit_model_v1.joblib` instead of retraining:

```powershell
python .\main.py --input .\applicant.example.json
```

Use `--retrain` whenever the dataset, preprocessing policy, or model configuration changes.

Score a single applicant from JSON input:

```powershell
python .\main.py --input .\applicant.example.json
```

Print the applicant's structured 5C Credit Appraisal Memo:

```powershell
python .\main.py --input .\applicant.example.json --cam
```

The memo contains Character, Capacity, Capital, Collateral, and Conditions sections.
Each section includes only stability-filtered SHAP reasons or an explicit message when
no high-confidence factor is available for that category.

Run the deterministic Phase 4 ablation and coverage evaluation:

```powershell
python .\evaluate_stability_filter.py
```

The first run saves a fixed 50-row test sample to
`evaluation/phase4_sample.json`; later runs reuse those same row indices. Results are
written to `evaluation/phase4_results.json` and include the number of raw explanations
that would have contained an unstable feature, reasons removed by filtering, decision
counts, and empty-section coverage for each 5C category.

The sample applicant file is a flat JSON object with the same feature names used in the trained model.

## Model behavior

The default modeling workflow includes:

- binary target encoding where `P4` is treated as the reject/high-risk label
- median imputation for numeric fields
- most-frequent imputation for categorical fields
- one-hot encoding for categorical features
- XGBoost with class imbalance weighting via `scale_pos_weight`
- threshold tuning on a validation split using F2 score
- local SHAP explanations for the top risk drivers
- structured Character, Capacity, Capital, Collateral, and Conditions memo sections
- 100-seed SHAP rank stability analysis using Kendall's W on a fixed 512-row test prefix
- memo eligibility limited to the top 10 mean-ranked features with rank range at most 3
- persisted model, preprocessing, threshold, and feature-schema contract

The stability procedure trains each seed on identical training data and changes only
`random_state`. It reports overall concordance, concordance for the five highest
mean-ranked features, and concordance for the six features with the greatest rank
variance. This follows the seed-initialization design described by Lin and Wang (2025),
"SHAP Stability in Credit Risk Management," *Risks*, 13(238).

## Validation

Run the project tests with:

```powershell
pytest -q
```

The tests verify metric generation, threshold logic, applicant inference, stability filtering, 5C mapping, persistence, and that protected or label-proxy attributes do not leak into the final model input set.

## Notes

- This project is designed for experimentation and evaluation in a local environment, not a production deployment pipeline.
- The training data should be reviewed before use in a regulated lending workflow, especially around fairness and label leakage risk.
- Feature exclusions are intentionally conservative to reduce the chance of using underwriting labels or protected attributes as direct model inputs.
