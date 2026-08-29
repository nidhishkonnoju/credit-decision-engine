# Credit Decision Engine

A Python-based credit risk scoring project that trains an explainable XGBoost model, filters protected and label-proxy fields, and produces a per-applicant decision with SHAP-based reasons.

## Overview

This project builds a compact credit decision pipeline for a bank-style underwriting workflow:

- Loads internal bank-data and bureau-data tables from the `Dataset` folder.
- Cleans numeric sentinels and removes protected / audit-sensitive fields.
- Trains a cost-aware XGBoost classifier for binary default-risk labeling.
- Selects a threshold using validation F2 performance to prioritize recall for risky applicants.
- Scores a single applicant from JSON input and returns:
  - decision (`APPROVE` or `REJECT`)
  - probability score
  - threshold used
  - top SHAP features driving the decision

The implementation is intentionally lightweight and designed to be easy to run locally for experimentation and model validation.

## Repository structure

- `app/` — model, preprocessing, and inference logic
  - `app/preprocessing.py` — dataset loading, target construction, missing-value handling, feature exclusions
  - `app/modeling.py` — training, metrics, threshold selection, SHAP summarization
  - `app/inference.py` — JSON applicant loading and decision generation
- `tests/` — smoke tests for modeling and fairness-related feature exclusions
- `Dataset/` — expected input data files (`External_Cibil_Dataset.xlsx` and `Internal_Bank_Dataset.xlsx`)
- `main.py` — training entry point and CLI runner
- `applicant.example.json` — sample applicant payload for local scoring
- `requirements.txt` — project dependencies

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
python .\main.py
```

Score a single applicant from JSON input:

```powershell
python .\main.py --input .\applicant.example.json
```

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

## Validation

Run the project tests with:

```powershell
pytest -q
```

The smoke tests verify metric generation, threshold logic, applicant inference, and that protected or label-proxy attributes do not leak into the final model input set.

## Notes

- This project is designed for experimentation and evaluation in a local environment, not a production deployment pipeline.
- The training data should be reviewed before use in a regulated lending workflow, especially around fairness and label leakage risk.
- Feature exclusions are intentionally conservative to reduce the chance of using underwriting labels or protected attributes as direct model inputs.
