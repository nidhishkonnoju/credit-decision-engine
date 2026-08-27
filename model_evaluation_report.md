# Audited Model Evaluation Report

## Scope

This project predicts the existing `Approved_Flag` underwriting tier as a binary target:

- `P4` = 1, reject/high-risk
- `P1`, `P2`, `P3` = 0, approve/low-risk

This is not a validated future-default model because `Approved_Flag` is an underwriting label rather than an independent realized-default outcome.

## Feature policy

The model uses 48 features after merging the external CIBIL and internal bank tables. The following are excluded:

- identifiers and target fields: `PROSPECTID`, `Approved_Flag`
- protected fields: `GENDER`, `MARITALSTATUS`
- audited label-proxy fields and related bureau aggregates listed in `AUDITED_LABEL_PROXY_COLUMNS` in `app/preprocessing.py`

`Unseen_Dataset.xlsx` is excluded from evaluation. It is the 42-column score-only artifact from a different, unaudited reference pipeline. It overlaps with the current feature set in only 26 columns and is not a valid holdout for this audited 48-feature model.

## Evaluation design

1. Merge and clean the two labelled source tables.
2. Create an 80/20 stratified train/test split.
3. Split the training portion again into model-training and validation portions.
4. Train cost-aware XGBoost using the natural class ratio through `scale_pos_weight`.
5. Select the operating threshold on validation data using F2, which weights recall approximately twice as much as precision. This represents the project assumption that approving a risky applicant is more costly than unnecessarily rejecting a safe applicant.
6. Evaluate once on the untouched test set.
7. Report five-fold stratified cross-validation on the training portion.

## Current results

### Untouched test set

- ROC-AUC: `0.7523`
- PR-AUC: `0.3034`
- KS: `0.3675`
- Recall: `0.7815`
- Precision: `0.1953`
- F1: `0.3125`
- F2: `0.4884`
- Accuracy: `0.6063`
- Predicted reject rate: `0.4582`
- Selected threshold: `0.40`

### Five-fold stratified cross-validation

- Mean ROC-AUC: `0.7513`
- Mean PR-AUC: `0.3108`
- Mean KS: `0.3712`
- Mean recall: `0.6243`
- Mean precision: `0.2319`
- Mean F1: `0.3381`
- Mean F2: `0.4664`

## Interpretation

Accuracy is not the primary metric because the positive class rate is approximately 11.5%. Recall, precision, F1, PR-AUC, and KS provide more useful information for this imbalanced classification task.

The cross-validation results are close to the final holdout ROC-AUC, PR-AUC, and KS results, which provides a basic stability check. The model should still be presented as a mini-project classifier for the existing underwriting tier, not as evidence of real-world default prediction or production lending suitability.

## Current output

The terminal workflow supports JSON input containing the 48 audited model features and prints:

- APPROVE or REJECT
- rejection-risk probability
- selected threshold
- SHAP-based applicant-specific reasons

No CAM generation or production application layer is included in the current scope.
