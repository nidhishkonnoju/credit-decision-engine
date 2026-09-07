# Credit Risk Assessment with Stability-Validated Explainability

## 1. Overview

This project addresses a core problem in credit underwriting: a model can look highly predictive in a lab, yet still fail as a usable decision tool if the label is not an independent default outcome and if the output cannot be communicated clearly to a human decision-maker.

The project therefore combines three components into one operational workflow:

1. a leakage-aware underwriting model trained on an audit-safe feature set,
2. SHAP-based local explanations filtered to stable features,
3. a bank-officer-friendly decision layer with APPROVE / REVIEW / REJECT output and a plain-language CAM.

This is not a generic default-risk benchmark. It is a practical underwriter support tool for an existing bureau-plus-bank underwriting tier, designed to be credible, explainable, and operationally usable.

---

## 2. The core problem

The project encountered the most important failure mode in credit risk modeling: the target label itself can contain underwriting information. If the target is an internal approval tier, then the model can learn directly from the underwriting logic rather than from actual borrower default behavior.

The implications are severe:

- model AUC can become unrealistically high even when the feature set is effectively encoding the label assignment process,
- the model learns proxy patterns instead of independent risk drivers,
- the resulting output looks “accurate” but is not defensible as a true credit-risk model.

The project therefore treats leakage auditing as a required modeling step rather than as a post-hoc explanation for a model score.

---

## 3. Leakage audit and target limitation

The project explicitly identifies and removes underwriting leakage before model training.

The audited methodology is straightforward:

- identify suspicious fields that are functionally tied to underwriting or bureau score assignment,
- test whether a shallow model can recover the target label from those fields with extremely high AUC,
- remove those fields as part of the feature policy,
- retrain and re-evaluate the final model on a defensible feature set.

This produces a more defensible underwriting-tier classifier, while not changing the limitation that the target is not a realized default event.

The leakage audit originally surfaced extreme behavior of the form:

- a shallow model using suspicious bureau fields achieved near-perfect discrimination at roughly 0.9998 AUC,
- those fields were not independent risk signals, but policy-embedded labels or proxies.

After the audit-based exclusion of the leakage fields, the final audited model reports:

- ROC-AUC: 0.7503 on the untouched test set,
- 5-fold CV mean ROC-AUC: 0.7518,
- threshold-aligned CV/test evaluation, which is now directly comparable.

These results support reporting the model as an underwriting-tier classifier rather than as a future-default model.

---

## 4. Actual project design

The final system is intentionally narrow and robust:

- train on an 80/20 stratified split,
- select the primary operating threshold on validation data using F2,
- evaluate on the untouched test set at the same threshold,
- compute SHAP values for each applicant,
- keep only features that pass the 100-seed SHAP rank stability rule,
- translate the top stable reasons into a plain-language CAM,
- present the final decision as APPROVE / REVIEW / REJECT for real-world bank-officer use.

The project therefore does not build a separate scoring pipeline for CSVs or CAM output. It reuses the same per-row applicant logic in a loop for batch scoring.

---

## 5. Feature engineering and fairness policy

The feature set is built from merged bureau and internal bank data with the following policy:

- remove protected or sensitive attributes such as `GENDER` and `MARITALSTATUS`,
- remove identifiers such as `PROSPECTID`,
- remove the target label itself (`Approved_Flag`),
- remove known label-proxy and underwriting-equivalent fields via `AUDITED_LABEL_PROXY_COLUMNS`,
- apply missing-value normalization for sentinels such as `-99999`, blank strings, and `NA`.

This is a governance-first feature policy, not merely a statistical cleanup. It reflects the fact that fairness and explainability are not optional add-ons in underwriting.

---

## 6. Model and threshold logic

The core model is an XGBoost classifier trained with class weighting to reflect the original positive rate.

The project uses a two-step threshold policy:

1. primary operating threshold at 0.40 for the standard reject vs approve decision,
2. upper review threshold at approximately 0.6547, derived from validation behavior,

with the review band defined as:

- below 0.40: APPROVE
- 0.40 to 0.6547: REVIEW
- above 0.6547: REJECT

This design is more realistic for a human workflow than a binary cutoff alone.

The threshold search uses validation data, not the test set, and the final CV summary uses the same operating threshold so metric comparisons are methodologically honest.

---

## 7. Stability-filtered SHAP explanations

The model does not dump all top SHAP impacts into a final memo. Instead, it filters the explanations by stability.

The current logic follows Lin and Wang (2025), "SHAP Stability in Credit Risk Management," *Risks*, 13(238):

- train 100 models on identical training data while varying only `random_state`,
- rank features by mean absolute SHAP value on a fixed 512-row test sample,
- calculate Kendall's W overall, for the top five mean-ranked features, and for the six highest-rank-variance features,
- allow a feature into the memo only when it is in the top 10 by mean rank and its rank range is at most 3.

This matters because mid-tier features can appear highly ranked in a single run yet be unstable across random-seed fits. For bank communication, unstable reasons are worse than no reason at all.

---

## 8. CAM design: dual audience output

The final output is a dual-audience credit appraisal memo.

### 8.1 Internal technical summary

The internal version keeps the original SHAP features, contributions, and decision metadata so the underwriting team can audit the local explanation and verify the risk logic.

### 8.2 Applicant-facing plain-language summary

The applicant-facing version converts encoded or technical feature names into human-readable descriptions such as:

- age of the oldest credit account,
- monthly income,
- percentage of your total balances currently in use,
- share of your exposure that is unsecured.

The applicant summary avoids raw encoded names such as `numeric__Age_Newest_TL` and instead emits natural-language reasons such as:

- “Your age of the newest credit account increases your risk of rejection.”

This is important for operational clarity and for any explanation process that needs to be readable by a non-technical stakeholder.

---

## 9. Batch scoring and operational support

The system supports batch CSV scoring by reusing the single-applicant path in a loop rather than inventing a second downstream model.

The batch routine:

- loads the CSV,
- validates that all required columns are present,
- normalizes common CSV noise such as thousands separators, blank strings, `NA`, and `-99999`,
- reuses `handle_missing_sentinels` and `predict_applicant` for each row,
- returns a list of per-row decisions.

This means all scoring logic remains centralized and consistent with the audited single-row path.

---

## 10. Final measured outcomes

The current audited model reports the following on the untouched test set:

- ROC-AUC: 0.7503
- PR-AUC: 0.3049
- KS: 0.3657
- recall: 0.7670
- precision: 0.1915
- F1: 0.3065
- F2: 0.4791
- threshold: 0.40

The same threshold is used in the 5-fold CV summary, which preserves comparability.

The three-tier decision distribution on the held-out set is:

- APPROVE: 54.14%
- REVIEW: 34.33%
- REJECT: 11.53%

This is much closer to an actual officer-facing workflow than a raw binary reject rate.

---

## 11. What this project is and is not

This project is:

- a transparent underwriting-tier classifier,
- a leakage-aware modeling workflow,
- an explainable decision support system with SHAP + CAM,
- a bank-officer triage aid with review bandwidth.

This project is not:

- a validated real-world default-risk model,
- a claim that the current underwriting label is a future default outcome,
- a production-ready lending engine without human governance.

The defensible description is therefore: the project implements a leakage-aware, explainable underwriting decision-support workflow and reports its target definition and predictive limitations explicitly.
