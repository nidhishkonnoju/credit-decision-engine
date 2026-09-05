# Audited Model Evaluation Report

## Scope

This project models the existing `Approved_Flag` underwriting tier as a binary risk label, not a realized default event:

- `P4` = 1, reject/high-risk
- `P1`, `P2`, `P3` = 0, approve/low-risk

This model is therefore best described as a classifier for the current underwriting tier, not a validated future-default model. The project is explicit that the label is a bureau/in-house risk assignment rather than a true realized-default outcome.

## Feature policy

The final model uses 48 audited features after merging the external bureau and internal bank tables. The following are excluded before training:

- identifiers and target fields: `PROSPECTID`, `Approved_Flag`
- protected fields: `GENDER`, `MARITALSTATUS`
- audited label-proxy and underwriting-equivalent fields listed in `AUDITED_LABEL_PROXY_COLUMNS` in `app/preprocessing.py`

The exclusion list was not inferred from names alone. It was informed by a leakage audit that demonstrated a shallow tree using suspected proxy features can achieve near-perfect discrimination, which is exactly the pattern we wanted to avoid.

## Evaluation design

1. Merge and clean the two labeled source tables.
2. Create an 80/20 stratified train/test split.
3. Use a validation subset from the training data to select the operating threshold.
4. Train a cost-aware XGBoost model with `scale_pos_weight` based on the natural class ratio.
5. Select the primary operating point on validation data using F2, which prioritizes recall when the cost of approving a risky applicant is material.
6. Evaluate once on the untouched test set at the same threshold used in production.
7. Report 5-fold stratified cross-validation on the training portion using the same primary threshold so the thresholded CV and holdout metrics are comparable.
8. Produce batch CSV scoring by reusing the existing single-applicant scoring path row by row.

## Current results

### Untouched test set

- ROC-AUC: 0.7523
- PR-AUC: 0.3034
- KS: 0.3675
- Recall: 0.7815
- Precision: 0.1953
- F1: 0.3125
- F2: 0.4884
- Accuracy: 0.6063
- Predicted reject rate: 0.4582
- Primary threshold: 0.40

### Five-fold stratified cross-validation

- Primary threshold: 0.40
- Mean ROC-AUC: 0.7513
- Mean PR-AUC: 0.3108
- Mean KS: 0.3712
- Mean recall: 0.7807
- Mean precision: 0.1934
- Mean F1: 0.3099
- Mean F2: 0.4857

### Three-tier operational output

The final operating model now uses a second, higher review threshold above the primary cutoff. This creates a realistic bank-officer queue instead of a binary reject-only presentation.

- Primary decision threshold: 0.40
- Review threshold: 0.6569
- Held-out test distribution:
  - APPROVE: 54.18%
  - REVIEW: 34.14%
  - REJECT: 11.69%

No bucket is empty.

### SHAP stability analysis

The stability analysis trains 100 XGBoost models on the identical training data,
varying only `random_state`. Each model is evaluated on the same fixed 512-row prefix
of the untouched test matrix. Features are ranked by mean absolute SHAP value, and
Kendall's W is calculated from the resulting rank matrix.

- Overall W: 0.9891
- Top-five mean-ranked feature W: 1.0000
- Six highest-rank-variance feature W: 0.4258

A feature is eligible for a customer-facing explanation only when it is among the top
10 features by mean rank and its rank range across the 100 seeds is at most 3. The
stored feature names use the exact encoded names emitted by the fitted preprocessor.

### Stability-filter ablation and 5C coverage

The Phase 4 evaluator uses a fixed 50-applicant sample from the untouched test split.
The sample is saved with both dataframe row indices and `PROSPECTID` values, and is
reused on subsequent runs.

| Measure | Result |
|---|---:|
| Memos with an unstable raw top-five reason | 17 / 50 (34.00%) |
| Raw reasons across the sample | 250 |
| Stability-filtered reasons across the sample | 226 |
| Reasons removed by the stability filter | 24 |
| APPROVE / REVIEW / REJECT | 30 / 16 / 4 |
| Empty Character sections | 4 / 50 (8.00%) |
| Empty Capacity sections | 15 / 50 (30.00%) |
| Empty Capital sections | 28 / 50 (56.00%) |
| Empty Collateral sections | 5 / 50 (10.00%) |
| Empty Conditions sections | 30 / 50 (60.00%) |

The evaluator produced identical output on two consecutive runs. These results show
that the stability filter changes the generated explanations materially while also
revealing that Conditions and Capital are frequently empty in this sample.

## Interpretation

Accuracy is not the key objective because the positive class rate is roughly 11.5%. Recall, precision, F1, PR-AUC, and KS provide more informative evidence on this imbalanced underwriting-task setup.

The important fix in this round of work was methodological consistency: the 5-fold cross-validation metrics are now evaluated at the same primary threshold as the untouched test set. This prevents a false comparison between a thresholded holdout and an unthresholded or mismatched CV summary. The final model is therefore reported with internal consistency across train/validation/test evaluation.

The leakage audit defines the feature policy and changes the raw AUC story from a near-perfect but invalid signal to a reasonable, audit-safe score that reflects the current underwriting tier rather than future default risk. The stability analysis follows Lin and Wang (2025), "SHAP Stability in Credit Risk Management," *Risks*, 13(238): 100 seed-only XGBoost fits, fixed evaluation rows, SHAP rank lists, and Kendall's W.

## Current output and workflow

The terminal workflow supports:

- JSON single-applicant scoring using the audited 48-feature schema
- CSV batch scoring for multiple applicants, reusing the same per-row single-applicant path
- dual-audience CAM generation:
  - internal technical SHAP summary
  - applicant-facing plain-language explanation
- stability-filtered feature reasons using the documented 100-seed SHAP rank rule
- three-tier decisions: APPROVE / REVIEW / REJECT

The CLI prints the selected threshold, the review threshold, the probability, and the human-readable credit appraisal rationale.

The project should be presented as a realistic underwriting-tier classifier with explainability and operational triage, not as validated real-world default prediction or production lending suitability.
