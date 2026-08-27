---
name: train-lightgbm
description: Executes Step 8 (Credit Risk Evaluation) and Step 7 (Explainability) training pipeline
---
# LightGBM Training & SHAP Skill

## EXECUTION STEPS:
1. **LOAD**: Connect to Databricks via `DatabricksSession` and load the synthetic Indian SME dataset from Delta Lake.
2. **FEATURES**: Extract the 65 features mapped from Step 6 (Unified Feature Matrix), including NetworkX fraud flags and FinBERT sentiment.
3. **SPLIT**: Perform an 80/20 stratified train/test split.
4. **TUNE**: Run Optuna for hyperparameter tuning (target: `num_leaves=63`, `n_estimators=1000`).
5. **TRAIN**: Train the LightGBM classifier to calculate Default Risk Probability.
6. **EXPLAIN**: Wrap the model in `shap.TreeExplainer` to calculate Feature Contribution Analysis.
7. **REGISTER**: Log the model, parameters, and SHAP waterfall plots to MLflow.

## CONSTRAINTS:
Do not use consumer/retail data (like LendingClub). Rely strictly on the synthetic corporate data logic defined in our instructions.