# Stability-Aware Feature Selection for Credit Risk

Can repeated-seed SHAP rank stability pick a compact, reproducible feature subset that keeps predictive performance across model families?

On a merged credit-bureau and bank dataset (51,336 applicants), a **9-feature stable-SHAP subset** (an 89% reduction from 82 features) stays within about 0.001 ROC-AUC of the full feature set for Random Forest, XGBoost and LightGBM. It matches single-model SHAP top-10 selection and clearly beats a mutual-information top-10 baseline.

## Method

1. Train 100 XGBoost models with different seeds and record each model's raw-feature SHAP ranks.
2. Keep features with **mean rank ≤ 10 and rank range ≤ 3** (the stable set, F3).
3. Compare four feature sets (all features, MI top-10, SHAP top-10, stable SHAP) across RF, XGBoost and LightGBM.
4. Quantify differences with a paired bootstrap (5,000 resamples).

Preprocessing, selection and tuning use the development set only. The test set is used for final evaluation. Hyperparameters are frozen and shared across feature sets, so selection effects are not confounded with re-tuning.

## Data

Kaggle: `saurabhbadole/leading-indian-bank-and-cibil-real-world-dataset` (downloaded automatically via `kagglehub`). Bureau and bank files are joined on `PROSPECTID`. The binary target is `Approved_Flag == P4` (11.46% positive). `GENDER` and `MARITALSTATUS` are excluded from predictors. Split: stratified 80/20 (41,068 dev / 10,268 test).

## Dataset-validity diagnostic: `Credit_Score`

`Credit_Score` separates the classes almost perfectly: the rule `Credit_Score <= 658` is exact on the development set, and the score alone reaches 0.9998 test ROC-AUC. With no provenance metadata, this is treated as a validity diagnostic rather than proof of leakage. The primary experiment (`NO_CS`) removes it. The `WITH_CS` branch is kept for transparency.

## Results (NO_CS, held-out test set)

| Model | All 82 features | Stable SHAP (9) | MI top-10 |
|---|---|---|---|
| XGBoost | 0.9787 | 0.9779 | 0.8830 |
| LightGBM | 0.9779 | 0.9772 | 0.8843 |
| Random Forest | 0.9747 | 0.9748 | 0.8768 |

ROC-AUC at a 0.40 decision threshold. Stable SHAP is statistically indistinguishable from SHAP top-10. Against the full feature set, the ROC-AUC loss for XGBoost and LightGBM is very small but detectable in the unadjusted bootstrap (about -0.0007 to -0.0009).

**Stable features:** `enq_L3m`, `Age_Oldest_TL`, `num_times_delinquent`, `time_since_recent_enq`, `time_since_recent_deliquency`, `pct_PL_enq_L6m_of_L12m`, `pct_PL_enq_L6m_of_ever`, `num_std_12mts`, `max_deliq_12mts`

## Limitations

- **Proxy overlap.** 8 of the 9 stable features are in the project's earlier target-proxy audit list. Removing `Credit_Score` does not remove the proxy family, so read the result as compactness and retention, not as selection of provably reliable features.
- **Test-set history.** The test set was inspected during the `Credit_Score` diagnosis, before the `NO_CS` branch was added. Selection and tuning never used it.
- **In-sample SHAP.** SHAP is computed on training rows, so stability describes the fitted models, not out-of-sample explanations.
- **One dataset, one split.** Generalization is untested. Bootstrap intervals are unadjusted for multiple comparisons.

## Reproduce

```bash
pip install numpy pandas scikit-learn xgboost lightgbm shap scipy joblib kagglehub openpyxl
jupyter notebook credit-decision-project.ipynb
```

Run all cells top to bottom (Python 3.12; Kaggle authentication may be needed). All seeds are fixed, and settings such as `N_SEEDS`, `THRESHOLD` and `RUN_TUNING` are in §1. Outputs are written to `phase6_optimized_outputs/` and `phase6_optimization_ab_outputs/`.

## License

Code released under the MIT License (see `LICENSE`). The dataset is subject to its Kaggle terms.
