# AI-Driven Credit Risk Assessment with Automated Credit Appraisal Memo (CAM) Generation

## 1. Overview

Financial institutions rely on credit scoring to minimize risk, but modern Machine Learning (ML) models operate as **black boxes**. High accuracy often comes at the cost of explainability. Bank risk committees need actionable artifacts, not just raw probabilities.

Under RBI's Fair Practices Code (and global equivalents like ECOA), rejected loan applicants are legally entitled to specific, communicable reasons for denial.

**Pipeline:**
```
Applicant Profile (raw applicant profile & credit documents)
        |
Machine Learning (SMOTE + XGBoost)
        |
Explainability (SHAP feature attributions)
        |
Output: Plain-language Credit Appraisal Memo (CAM)
```

The goal is to synthesize predictive performance, class balancing, and explainability into a unified pipeline that outputs an actionable, legally compliant Credit Appraisal Memo — not just a risk score.

---

## 2. The Problem

- **Black-Box Algorithms** — Highly accurate ML models (like XGBoost) are mathematically opaque, making them difficult to audit or trust for high-stakes financial lending decisions.
- **Dataset Imbalance** — Real-world default rates are very low (typically 6–10%). Without balancing techniques, naive models simply approve everyone to achieve artificially high baseline accuracy.
- **Inactionable XAI Outputs** — Existing Explainable AI frameworks produce complex mathematical visualizations (e.g., Force Plots) that loan officers and layperson applicants cannot decipher.
- **Fragmented Systems** — Current industry approaches treat predictive performance, fairness, and explainability as separate, isolated steps rather than synthesizing them into a deployable decision artifact.

---

## 3. Problem Statement

**Design an explainable AI system that evaluates applicant default risk and generates a structured Credit Appraisal Memo (CAM) for bank credit officers and loan applicants to ensure fast, actionable, and legally compliant lending decisions.**

```
Applicant Data (financial & demographic records)
        -->
Explainable AI: XGBoost + SHAP (risk scoring & feature extraction)
        -->
Credit Appraisal Memo (automated risk justification)
```

- **Target users:** Bank risk committees, loan officers, & applicants
- **Input:** Tabular applicant financial data
- **Output:** Credit Appraisal Memo

---

## 4. Related Work / Research Gaps

| Approach | Limitation | What This Project Does Differently |
|---|---|---|
| Stacked Ensemble + SMOTE + SHAP | Explanations remain as raw visual plots | Translate raw SHAP data into a plain-language CAM |
| XGBoost SHAP-stability studies | Mid-importance features show high SHAP instability | Restrict CAM text generation strictly to bootstrap-stable features |
| DT/RF + LIME/SHAP for decision support | Explanations are not tailored to specific stakeholders | Generate dual-audience outputs (internal memo vs. applicant letter) |
| Systematic reviews of performance/fairness/XAI | Absence of unified frameworks that co-optimize all pillars | Synthesize scoring and explainability into a single deployable artifact |
| TabTransformer + weighted loss + SHAP | High computational complexity; specific to tabular data | Use SMOTE for model-agnostic imbalance handling to save compute |

**Research gaps being addressed:**
- **G1 — Impractical XAI outputs:** raw SHAP plots aren't understood by non-technical stakeholders or applicants.
- **G2 — Unstable attributions:** using all SHAP features leads to noisy, unstable explanations for mid-tier features under random initialization.
- **G3 — Imbalance-handling overhead:** deep-learning approaches to class imbalance are expensive; model-agnostic resampling (SMOTE) is cheaper.
- **G4 — Lack of unified frameworks:** predictive performance, fairness, and explainability are usually treated as isolated steps rather than one deployable artifact.

---

## 5. Objectives

1. Collect and pre-process imbalanced historical credit datasets using SMOTE (or a schema-safe variant) to ensure fair representation.
2. Train and benchmark an XGBoost model against an interpretable Logistic Regression baseline (and ideally Random Forest) for default classification.
3. Implement SHAP (Shapley Additive Explanations) to extract global feature rankings and local instance contributions per applicant.
4. Build an automated logic module that filters out unstable mid-tier features and translates only the stable top SHAP outputs into structured text.
5. Generate a dual-audience Credit Appraisal Memo (CAM): one version for internal bank risk committees, one plain-language version for applicants.
6. Evaluate model predictive performance (AUC/F1, ideally also PR-AUC and KS-statistic) and demonstrate end-to-end automated deployment.

---

## 6. Proposed Methodology

```
Applicant Data (tabular input)
        -->
Detect + Preprocess (balancing & scaling)
        -->
ML Pipeline (XGBoost vs. Logistic Regression vs. Random Forest)
        -->
Explainability (SHAP attributions, stability-filtered)
        -->
Automated Output (CAM generation)
```

**Process steps:**
1. Ingest tabular financial and demographic applicant records.
2. Balance the historical default classes and scale/encode features.
3. Train and benchmark models to predict a default probability score.
4. Extract the top-driving risk features globally and locally using SHAP.
5. Feed the stable SHAP outputs into text-generation logic to draft the Credit Appraisal Memo (internal + applicant-facing versions).

**Datasets & tools:**
- **Dataset:** Indian bank + CIBIL-style credit bureau data (tabular, applicant-level).
- **Imbalance handling:** SMOTE / SMOTENC (imbalanced-learn).
- **Models:** XGBoost, Logistic Regression, Random Forest (scikit-learn).
- **Explainability:** SHAP (Shapley Additive Explanations).
- **Environment:** Python notebook (Jupyter/Colab-compatible).

---

## 7. Core Novelty / Contribution

- **Artifact-driven AI:** most credit-risk XAI projects stop at "the model is explainable" via visual plots. The deliverable here is a tangible, actionable document (the CAM) a human can read and act on.
- **Regulatory compliance:** SHAP outputs are explicitly mapped to plain-text rejection reasons, aligning with fair-lending disclosure requirements (RBI Fair Practices Code / ECOA-style "adverse action" notices).
- **Stability filtering:** the text-generation module only cites SHAP features that are stable across repeated bootstrap re-estimation, not noisy mid-tier features.

**Novelty statement:** unlike generic scoring models that treat predictive performance, fairness, and explainability as fragmented steps, this system synthesizes them into a unified, deployable Credit Appraisal Memo satisfying both internal audit needs and consumer rights.

**Optional extensions worth considering (not core scope, but natural additions):**
- Actionable recourse — telling a rejected applicant what would change the outcome (e.g. via counterfactual search or the `dice-ml` library).
- Fairness/bias audit — checking approval-rate disparities across protected/proxy attributes, not just excluding those columns from training.
- Monotonic constraints in XGBoost so features like income can't perversely increase predicted risk.
- Calibrated probabilities (`CalibratedClassifierCV`) so the probability quoted in the CAM is trustworthy.

---

## 8. Planned Model Comparison

| Method | Explainability |
|---|---|
| Logistic Regression | Inherent (coefficients) |
| Random Forest (Ensemble) | Post-hoc (SHAP) |
| XGBoost (Standard) | Post-hoc (SHAP) |
| **XGBoost + CAM (target system)** | **SHAP + CAM generation** |

Each method should be evaluated both on the raw imbalanced data and on the balanced (SMOTE/SMOTENC) data. Metrics: **AUC**, **F1-score**, ideally also **PR-AUC** and **KS-statistic** given the low base rate. The primary success metric beyond raw scores is successfully generating an actionable, legally-defensible CAM from the XGBoost+SHAP output.

---

## 9. Implementation Notes / Known Pitfalls

- **Target definition:** if using an Indian bank + CIBIL bureau dataset with a risk-tier flag (e.g. P1–P4) as the label, verify what that field actually represents — a risk tier assigned at underwriting is not the same as a realized default outcome, and this affects how the problem should be framed.
- **Missing-value sentinels:** bureau-style datasets sometimes encode missing values as `-99999` instead of `NaN` — check for this and impute properly before scaling.
- **Protected attributes:** exclude `gender`, `maritalstatus` (and consider `education`) from model features — including them undermines the fair-lending compliance goal, even if unintentional. Note that removing them alone doesn't guarantee fairness (other features can proxy for them); a disparate-impact check post-training is more defensible.
- **SMOTE ordering:** apply SMOTE (or `SMOTENC`) *before* one-hot encoding categorical features, not after. Plain `SMOTE` on already one-hot-encoded columns generates invalid fractional category values (e.g. 0.6 of a category).
- **Calibration:** probabilities from a model trained on SMOTE-balanced data are not well-calibrated to the natural imbalanced population. Calibrate on a held-out, non-resampled split before quoting probabilities in the CAM.
- **Evaluation robustness:** prefer stratified k-fold cross-validation over a single train/test split, and apply any resampling *inside* each fold to avoid leakage.
- **Current build status:** a basic version already exists — a model that takes structured input and returns a text output, plus a front-end for entering inputs (has some known bugs to fix).

### Current leakage audit

`Approved_Flag` is an underwriting risk tier rather than an independent realized-default outcome. The audit supports that interpretation: a shallow depth-3 decision tree using a small set of suspicious bureau fields achieved held-out AUC `0.99989` and accuracy `0.999805`, with `Credit_Score` receiving all tree importance. After removing `Credit_Score`, `enq_L3m` dominated the next tree (`AUC 0.924`); after removing it, `enq_L6m` dominated (`AUC 0.900`). These are label proxies, not safe predictors for an independent approval model.

The production feature set therefore uses the explicit `AUDITED_LABEL_PROXY_COLUMNS` exclusion list in `app/preprocessing.py`, based on the tree audit and related bureau aggregates rather than name-pattern matching. A model trained after this exclusion reports ROC-AUC `0.7523` on the current holdout. The categorical single-feature screen found no non-target categorical field above AUC `0.85`; `last_prod_enq2` was highest at `0.6613`. These results should be reported as prediction of the existing underwriting tier, not validated future default risk.
