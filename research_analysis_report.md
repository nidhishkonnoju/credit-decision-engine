# Research Analysis Report — SHAP Stability Under Near-Perfect Target Proxies

**Generated:** 2026-09-14 · **Protocol version:** 1 · **Status:** analysis of committed 100-seed results (no model refitting)

---

## 1. Research question

> Can SHAP stability provide misleading evidence of explanation reliability when a
> credit-risk model contains target-construction leakage / near-perfect target proxies?

### Research gap

SHAP value explanations are widely used for interpretability in regulated credit
risk. Recent work shows that SHAP *stability* (rank-concordance of feature
importance across seeds) is often treated as evidence that an explanation is
trustworthy. However, a stability criterion is a *ranking-consistency* measure,
not a *validity* measure: it cannot distinguish a genuinely important feature
from a near-duplicate of the label that is artificially stable.

### Hypothesis

Primary (H1): in this dataset and experimental setting, an audited near-perfect
target proxy (`Credit_Score`, proxy AUC = 1.000) would appear as highly stable
and highly ranked under the SHAP stability protocol, such that a stability-only
filter would not flag it as problematic.

Secondary (H2): this behaviour persists across all three XGBoost stochastic
conditions (bundled baseline, row sampling, feature sampling).

Scope guard (from `research_protocol.md`): the study targets the existing
`Approved_Flag` underwriting tier (P4 = reject/high-risk), does not claim default
prediction, and the proxy-inclusive feature policy is research-only.

## 2. Current experimental design (as implemented, verified by code inspection)

**Code paths.** Runner: `run_research_experiments.py`. Core: `app/research_experiments.py`.
Preprocessing/target: `app/preprocessing.py`. Kendall's W and the production
stability module: `app/modeling.py`.

- Data: 51,336 merged bureau+bank applicants (`External_Cibil_Dataset.xlsx` +
  `Internal_Bank_Dataset.xlsx`, merged on `PROSPECTID`); binary target
  `target = 1` iff `Approved_Flag == "P4"` (`define_binary_target`).
- Split: 80/20 stratified, `random_state=42`, fitted once per feature policy
  (`build_preprocessed_data(include_audited_proxies=True)` in the runner).
  Preprocessing: numeric median imputation + missing indicators + standard
  scaling; categorical most-frequent imputation + one-hot. Fit on train, applied
  to test.
- Feature policy: `proxy_inclusive_diagnostic` (the audited 35 near-perfect
  target proxies are intentionally re-included; research-only).
- Conditions (`EXPERIMENT_CONDITIONS` / `DETERMINISM_CHECK`):

| Condition | Outer train sample | `subsample` | `colsample_bytree` | Seeds |
| --- | --- | ---: | ---: | ---: |
| `bundled_baseline` | per-seed 80% stratified resample (`random_state=seed`) | 0.8 | 0.8 | 100 |
| `row_sampling` | fixed full train | 0.8 | 1.0 | 100 |
| `feature_sampling` | fixed full train | 1.0 | 0.8 | 100 |
| `determinism_check` | fixed full train | 1.0 | 1.0 | 2 |

  All other hyperparameters fixed: `n_estimators=300, max_depth=5,
  learning_rate=0.05, scale_pos_weight=neg/pos, n_jobs=1`. Seeds are
  `range(100)` in every stochastic condition; `n_jobs=1` per fit.
- SHAP evaluation: `shap.TreeExplainer(model).shap_values(X_test[:512])`
  (identical rows in every condition; local sample = first 100 rows of the same
  pass). Global ranking per seed = descending rank of mean |SHAP| per encoded
  feature. Kendall's W over the 100×122 rank matrix
  (`kendalls_w` in `app/modeling.py`).
- Local metrics per applicant (`_local_stability`): mean over all 4,950 seed
  pairs of (a) Jaccard overlap of the top-5 |SHAP| feature sets, (b) cosine
  similarity of the full attribution vectors.
- Proxy strength (`estimate_univariate_proxy_strength`): 3-fold out-of-fold AUC
  of a shallow single-field tree (depth 3, `min_samples_leaf=30`), symmetrised
  `max(AUC, 1−AUC)`, computed on the **full** dataset (train+test rows) —
  a diagnostic association score, not a modelling input.
- Committed results: `evaluation/research_results_100_seeds.json` /
  `evaluation/research_results_100_seeds_final.json` (byte-identical modulo
  `generated_at_utc`; verified against `output/e2e_20260913_1521/`).
- Raw per-seed data: `evaluation/research_raw_100_seeds_final.npz` (100 seeds ×
  122 features per condition; 100 seeds × 100 applicants × 122 features for
  local vectors). Manifest: `evaluation/research_raw_manifest_final.json`.

## 3. Exact data / observational units

| Unit | Definition | n | Paired across conditions? |
| --- | --- | --- | --- |
| Condition-level | Kendall's W (one aggregate per condition) | 3 (+1 determinism) | descriptive only — never a test sample |
| Encoded-feature level | one of the 122 encoded columns; 61 descend from the 35 audited proxy fields, 61 from 48 audited model fields | 122 | yes (identical column sets, verified) |
| Raw-feature level | one of the 83 raw source fields (35 proxy / 48 audited) | 83 | yes |
| Applicant level | the fixed first 100 test rows (`PROSPECTID` recorded per row) | 100 | yes |

## 4. Proxy / leakage definitions actually supported by the repository

- The 35 fields in `AUDITED_LABEL_PROXY_COLUMNS` (`app/preprocessing.py`) are
  labelled **"audited near-perfect target proxy"** in code
  (`_feature_status`, `app/research_experiments.py`), with the explicit note:
  *"Excluded from production scoring; provenance has not been encoded as a
  leakage claim."*
- Supporting evidence (`project_context_technical.md` §3): the audit identified
  fields "functionally tied to underwriting or bureau score assignment" and a
  shallow model recovered the target from them at ≈0.9998 AUC. The proxy AUCs
  in the committed JSON (e.g. `Credit_Score = 1.000`, `enq_L3m = 0.87326`)
  are **association** evidence only.
- **`Credit_Score` is NOT documented as part of target construction anywhere in
  the repository.** The correct classification is: *audited near-perfect target
  proxy without proven target construction*. The stronger phrase
  "target-construction leakage" is not supported by repository evidence and
  should not be used for individual fields.
- Documentation inconsistency found: `project_paper_documentation.md`
  (Contribution 3) says "39 label-proxy columns"; the code policy contains 35.
  The paper text should be reconciled with the code (not silently changed here).


## 5. Existing descriptive results

Source: `evaluation/research_results_100_seeds_final.json` (byte-identical
to the original committed `evaluation/research_results_100_seeds.json` modulo
`generated_at_utc`; verified). Raw per-seed data:
`evaluation/research_raw_100_seeds_final.npz` (see §15).

**Global (condition-level, descriptive).**

| Condition | Kendall's W | Credit_Score mean rank (range) | Proxies in top-10 | Proxies in top-20 (expected 10) |
| --- | ---: | ---: | ---: | ---: |
| determinism_check | 1.0 | 1 (0) | — | — |
| bundled_baseline | 0.894822 | 1 (0) | 9/10 | 16/20 |
| row_sampling | 0.985470 | 1 (0) | 9/10 | 14/20 |
| feature_sampling | 0.934871 | 1 (0) | 9/10 | 16/20 |

`numeric__Credit_Score` is ranked first with rank range 0 and rank variance 0 in
**all four** conditions. Top proxy AUCs: `Credit_Score` 1.000, `enq_L3m`
0.873, `enq_L6m` 0.854, `enq_L12m` 0.817, `time_since_recent_enq` 0.787
(`proxy_strength` array of the same file).

**Local (applicant-level, n=100).**

| Condition | mean top-5 Jaccard (sd) | distinct per-applicant means | mean attribution cosine | min cosine |
| --- | ---: | ---: | ---: | ---: |
| determinism_check | 1.0000 (0) | 1 | 1.000000 | 1.000000 |
| bundled_baseline | 0.8115 (0.107) | 89 | 0.999774 | 0.999345 |
| row_sampling | 0.6813 (0.000) | **1** | 0.999999 | 0.999999 |
| feature_sampling | 0.8453 (0.089) | 91 | 0.999777 | 0.999552 |

**Anomaly detected and investigated (row_sampling degeneracy).** In
`row_sampling`, all 100 applicants share an *identical* mean top-5 Jaccard
(0.681332; one distinct value, vs 89/91 distinct values in the other
conditions). A focused 3-seed diagnostic using the exact research code path
(`build_preprocessed_data(include_audited_proxies=True)`, `XGBClassifier` with
the condition spec, `shap.TreeExplainer` on the same 100 applicants) showed that
in `row_sampling` the per-seed top-5 |SHAP| feature **set is identical for all
100 applicants** (1 distinct set per seed):

- seed 0 shared set: `first_prod_enq2_{CC, ConsumerLoan, HL, others}` +
  `Credit_Score`
- seed 1 = seed 2: `first_prod_enq2_{CC, ConsumerLoan, HL}` + `Credit_Score` +
  `time_since_recent_enq`

Control (`feature_sampling`): 26/25/18 distinct top-5 sets per seed —
applicant-dependent, as expected. Mechanism: with `colsample_bytree=1.0` the
dominant `Credit_Score` signal plus the one-hot block structure of
`first_prod_enq2`/`last_prod_enq2` produce a stable *global* top-5 pool; within
a seed every applicant's five largest |SHAP| values fall inside the same set, so
the per-applicant "local" Jaccard collapses to the seed-to-seed overlap of one
global 5-set. **Consequence:** the row_sampling local Jaccard (0.6813) is a
global statistic, not per-applicant fragility, and cross-condition local
comparisons involving it must be interpreted with that caveat.

**Full 100-seed verification.** Raw per-seed rank matrices and local attribution
vectors were then persisted to `evaluation/research_raw_100_seeds_final.npz`
(122 features × 100 seeds × 100 applicants per condition; manifest at
`evaluation/research_raw_manifest_final.json`). `recompute_local_stability.py`
re-derived all local metrics from these raw outputs and confirmed:

- `row_sampling` Jaccard: mean = 0.681332, **SD across applicants = 1.11e-16**
  (numerically constant; `degenerate: true`), with **all 100 seeds** having a
  single applicant-invariant top-5 set — not just 3 seeds. Per-applicant means
  are identical to 6 decimal places (1 distinct value).
- `bundled_baseline` Jaccard: mean = 0.811530, SD = 0.107, 89 distinct
  per-applicant values — genuinely per-applicant.
- `feature_sampling` Jaccard: mean = 0.845336, SD = 0.089, 91 distinct
  per-applicant values — genuinely per-applicant.
- All recomputed means match the committed aggregate values to < 1e-6
  (max abs diff ≈ 5e-7 for local; 0.0 for global ranks).
- Determinism check: Jaccard = 1.0 with 0 SD across both fits (2 seeds).

**Status of row_sampling local Jaccard:** retained as a diagnostic aggregate
only (it measures seed-to-seed global-set volatility, not per-applicant
fragility). It is **not** used as evidence of individual-level instability in
any statistical test or claim. The only conditions supporting applicant-level
local inference are `bundled_baseline` and `feature_sampling`.

## 6. Statistical analysis plan (pre-specified before testing)

Implemented verbatim in `analyze_research_results.py`; outputs
`evaluation/research_statistical_analysis.json`. No test was chosen after
inspecting significance. Kendall's W is never used as a test sample. All tests
two-sided, α=0.05, Holm–Bonferroni within each family; effect sizes with 95%
bootstrap percentile CIs (10,000 resamples, seed 2026).

- **F1 (RQ1, encoded level):** Mann–Whitney U, proxy (n=61) vs non-proxy
  (n=61), on `mean_rank`, `rank_range`, `stability_score`, per condition
  (9 tests). Effect: rank-biserial r (+ = proxy larger).
- **F2 (RQ1 sensitivity, raw level):** Mann–Whitney U on raw-aggregated
  `mean_rank` / `stability_score` (35 vs 48; 6 tests).
- **F3 (RQ1 association):** Spearman ρ between raw proxy AUC (n=83) and
  raw-aggregated `mean_rank` / `stability_score`, permutation p (6 tests).
  Observational association only.
- **F4 (RQ1 enrichment):** hypergeometric test for proxy over-representation in
  the top-10 / top-20 mean-ranked features (6 tests).
- **F5 (RQ2, paired feature level):** Wilcoxon signed-rank across condition
  pairs on `mean_rank`, `rank_variance`, `rank_range` (n=122 pairs; 9 tests).
- **F6 (RQ3, applicant level):** Wilcoxon signed-rank, `cosine` vs `jaccard`
  within condition (n=100; 3 tests).
- **F7 (RQ3, applicant level):** Wilcoxon signed-rank, `jaccard` across
  condition pairs (n=100; 3 tests).
- Global↔local correspondence across conditions: only 3–4 condition-level
  points → **descriptive only, no test** (no p-value manufactured).

## 7. Results of statistical tests

Source: `evaluation/research_statistical_analysis.json` (generated by
`analyze_research_results.py` from the committed 100-seed JSON).

**F1 — encoded level, proxy vs non-proxy (9 tests, 0 survive Holm).**

| Condition | Metric | p (raw) | p (Holm) | r | median proxy / non-proxy |
| --- | --- | ---: | ---: | ---: | --- |
| bundled | mean_rank | 0.568 | 1.0 | −0.060 | 60.09 / 59.85 |
| bundled | rank_range | 0.064 | 0.385 | −0.195 | 40 / 42 |
| bundled | stability_score | 0.0058 | 0.052 | +0.290 | 0.019 / 0.008 |
| row | mean_rank | 0.649 | 1.0 | −0.048 | 51.00 / 62.02 |
| row | rank_range | 0.350 | 1.0 | −0.091 | 0 / 1 |
| row | stability_score | 0.320 | 1.0 | +0.097 | 1.000 / 0.981 |
| feature | mean_rank | 0.568 | 1.0 | −0.060 | 63.96 / 57.84 |
| feature | rank_range | 0.051 | 0.358 | −0.205 | 11 / 35 |
| feature | stability_score | 0.0132 | 0.105 | +0.260 | 0.190 / 0.018 |

**F2 — raw level, proxy vs non-proxy (6 tests, 2 survive Holm).**

| Condition | Metric | p (Holm) | r | 95% CI of r | median proxy / non-proxy |
| --- | --- | ---: | ---: | --- | --- |
| bundled | stability_score | **1.37e-5** | +0.611 | [0.401, 0.789] | 0.258 / 0.009 |
| row | stability_score | 0.326 | −0.126 | [−0.384, 0.136] | 0.799 / 0.933 |
| feature | stability_score | **1.07e-3** | +0.479 | [0.239, 0.696] | 0.405 / 0.015 |
| bundled / row / feature | mean_rank | 0.080 / 0.080 / 0.080 | ≈−0.29 | ≈[−0.53, −0.03] | proxies better ranked |

**F3 — Spearman ρ, proxy AUC vs raw metrics (6 tests, 5 survive Holm).**

| Condition | Metric | ρ | p (Holm) |
| --- | --- | ---: | ---: |
| bundled | mean_rank | −0.315 | **0.018** |
| bundled | stability_score | +0.476 | **0.0006** |
| row | mean_rank | +0.137 | 0.213 |
| row | stability_score | **−0.249** | **0.045** (sign flip) |
| feature | mean_rank | −0.297 | **0.018** |
| feature | stability_score | +0.439 | **0.0006** |

**F4 — proxy enrichment in top-k (6 tests, 6 survive Holm).** Top-10:
9/10 proxy in every condition (expected 5.0; Holm p = 0.033). Top-20:
16/20 bundled (0.018), 14/20 row (0.043), 16/20 feature (0.018).

**F5 — paired across conditions, encoded features n=122 (9 tests).**
`mean_rank`: no significant shift (Holm p = 1.0; |r| ≤ 0.07) — the global
importance ordering is preserved across stochastic settings.
`rank_variance` / `rank_range`: highly significant everywhere (Holm p ≤ 1.8e-9):

| Comparison | median Δ variance [95% CI] | median Δ range [95% CI] | r (range) |
| --- | --- | --- | ---: |
| bundled → row | −66.25 [−104.2, −41.1] | −37.5 [−42.0, −32.0] | −0.822 |
| bundled → feature | −8.88 [−31.0, −1.7] | −2.0 [−4.0, −1.0] | −0.732 |
| row → feature | +27.13 [+6.4, +49.8] | +21.0 [+12.5, +32.0] | −0.767 |

Row sampling is the most globally stable condition, bundled the least.

**F6 — per-applicant cosine vs Jaccard within condition (3 tests, 3 survive
Holm, all p_holm < 1e-16, r ≤ −0.989).** Median cosine − Jaccard: bundled
+0.237 [0.192, 0.252]; row +0.319 [0.319, 0.319] (degenerate — see §5);
feature +0.174 [0.150, 0.204]. Attribution-vector cosine ≈ 1 coexists with
much lower top-5 set overlap.

**F7 — per-applicant Jaccard across condition pairs (3 tests, 3 survive
Holm).** bundled → row: −0.081 [−0.126, −0.067]; bundled → feature: +0.029
[0.018, 0.042]; row → feature: +0.145 [0.115, 0.169]. (row_sampling values are
degenerate — see §5.)

## 8. Effect sizes and confidence intervals (summary)

- Proxy-vs-non-proxy raw-level stability gap: r = +0.61 (bundled),
  +0.48 (feature), −0.13 (row, n.s.) — large-to-moderate where present, absent
  under row sampling.
- Proxy AUC association: |ρ| ≈ 0.25–0.48 (moderate), with a **sign flip** under
  row sampling (stability: +0.48/+0.44 vs −0.25).
- Top-k enrichment: 1.8×–1.8× expected proxy counts (9/10 vs 5/10; 16/20,
  14/20, 16/20 vs 10/20).
- Condition effects on rank variability: median rank-range differences of
  2–37.5 rank positions (all CIs exclude 0).
- Local metric gap: cosine−Jaccard median 0.17–0.32 (CIs exclude 0).


## 9. Global vs local stability findings

- **They decouple.** The condition with the *highest* global concordance
  (row_sampling, W = 0.9855) has the *lowest* mean local top-5 Jaccard (0.6813),
  while the condition with the lowest W (bundled, 0.8948) has a higher local
  Jaccard (0.8115). With only 3–4 condition-level points this is reported
  descriptively — no significance test is possible or claimed.
- **Cosine and Jaccard disagree** (F6): attribution vectors are nearly identical
  (cosine ≈ 0.9998–1.0) while top-5 reason *sets* differ (Jaccard ≈ 0.68–0.85).
  A single dominant feature (`Credit_Score`) makes full-vector cosine high even
  when the lower-ranked reason set reshuffles — cosine overstates local
  reliability.
- **Mechanistic caveat:** in row_sampling the per-applicant Jaccard is
  degenerate (identical for all 100 applicants; per-seed top-5 sets are
  applicant-independent — §5). Its local number measures seed-to-seed overlap of
  one *global* set, not applicant-level fragility. The RQ3 local comparison is
  therefore cleanest between `bundled` and `feature_sampling` (both show genuine
  per-applicant variation: 89/91 distinct means), where local Jaccard still
  differs significantly (median +0.029, CI [0.018, 0.042]) while W differs by
  ~0.04.

## 10. Robustness across stochastic conditions

- **Global stability of proxies:** proxies are significantly more stable than
  non-proxies at the raw level in `bundled_baseline` and `feature_sampling`
  (F2), the proxy-AUC association has the same sign in both (F3: ρ_stab ≈
  +0.44–0.48), and top-10/top-20 enrichment holds in every condition (F4,
  Holm p ≤ 0.043). The encoded-level comparison shows the same direction but
  does not survive Holm correction (F1) — the signal is concentrated in a
  subset of strong proxies rather than spread across all 61 encoded columns.
- **Not fully robust:** under `row_sampling` the proxy-stability association
  flips sign (F2 r = −0.13 n.s.; F3 ρ = −0.249, Holm p = 0.045). A plausible
  mechanism is scale compression: with `colsample_bytree = 1.0`, 61 of 122
  encoded columns have near-zero rank variance (median stability ≈ 0.80–1.0),
  so between-feature stability contrasts shrink and can reverse. This should be
  reported as a boundary condition, not hidden.
- **Importance is robust:** mean ranks barely move across conditions (F5,
  Holm p = 1.0), so the top-ranked proxies (`Credit_Score`, `enq_L3m`,
  `num_times_delinquent`, …) remain the top-ranked proxies everywhere.

## 11. What the evidence supports

In this dataset and experimental setting (51,336 applicants, one XGBoost
configuration, 100 seeds × 3 stochastic conditions + determinism check):

1. A feature with out-of-fold single-field AUC = 1.000 (`Credit_Score`) is
   ranked #1 with zero rank variance in every stochastic condition, including
   the bundled treatment that reproduces the base paper's seed-only protocol.
2. Audited near-perfect target proxies are significantly over-represented among
   the most stable, most highly ranked features in all three conditions
   (hypergeometric, Holm-corrected), and at the raw-feature level are
   significantly more stable than non-proxies in two of three conditions
   (Wilcoxon–Mann–Whitney, Holm-corrected).
3. Aggregate Kendall's W of 0.89–0.99 would certify the ranking as highly
   concordant, yet the top of that ranking is dominated by audited proxies.
   Global stability therefore provides **no warning** about the problematic
   features in this setting — consistent with the research hypothesis that
   stability alone can be misleading evidence of explanation reliability.
4. Global and local stability measures can decouple (§9), and a high-cosine /
   moderate-Jaccard dissociation exists within every condition.

## 12. What the evidence does NOT support

- It does **not** prove that any individual field (including `Credit_Score`)
  participates in constructing the label; the repository documents association
  (AUC), not target-construction provenance.
- It does **not** show that SHAP stability is *generally* unreliable — only
  that it fails to flag audited proxies here.
- It does **not** establish causality: proxy AUC → importance/stability
  associations (F3) are observational, within one fitted model family.
- It does **not** support a significance test on Kendall's W across conditions
  (one aggregate value per condition) or on the global-vs-local relationship
  (n = 3–4 conditions).
- It does **not** support interpreting row_sampling's local Jaccard as
  applicant-level fragility (degenerate — §5/§9).
- It does **not** speak to production: the production artifact, thresholds, and
  feature policy were untouched (research code never persists models; the
  2026-09-13 e2e run reproduced the artifact byte-for-byte).

## 13. Threats to validity / limitations

1. **Proxy AUC computed on the full dataset** (train+test rows;
   `run_research_experiments.py` calls `estimate_univariate_proxy_strength` on
   `data["X"], data["y"]`). Acceptable for a diagnostic label, but the AUCs are
   not out-of-sample estimates and should not be quoted as generalisation
   performance.
2. **Bundled condition confounds** outer 80% row resampling with
   `colsample_bytree = 0.8` (both change together vs row_sampling). This is the
   intended reproduction of the seed-bundled treatment, but bundled-vs-row
   contrasts are not single-factor comparisons.
3. **Encoded-level units are correlated**: 61 encoded proxy columns derive from
   35 raw fields (one-hot dummies and missing indicators share a source), so F1
   is conservative; F2/F3 (raw level, n=83) are the better-powered analyses, and
   raw aggregation gives one-hot blocks equal weight to single numerics.
4. **Dependence among the 122 features** (raw fields vs their missing
   indicators) means per-feature tests are not fully independent; Holm
   correction controls family-wise error but not dependence.
5. **Single dataset, single model family, one evaluation prefix** (512 rows;
   100 applicants) — findings are setting-specific.
6. **Row-sampling degeneracy** limits RQ3 conclusions for that condition (§5).
7. **Documentation drift**: paper text says "39 label-proxy columns" vs 35 in
   code (§4); must be reconciled before submission.
8. Per-seed rank matrices and per-applicant per-seed local attribution vectors are
   now persisted in `evaluation/research_raw_100_seeds_final.npz` (122
   features × 100 seeds × 100 applicants per stochastic condition; manifest at
   `evaluation/research_raw_manifest_final.json`). `recompute_local_stability.py`
   re-derives all local metrics from these raw outputs for full transparency and
   degeneracy validation. Bootstrap confidence intervals for Kendall's W remain
   pre-registered as not pursued: with n=3–4 conditions, a bootstrap over
   conditions is not statistically defensible, and per-condition bootstrap over
   seeds does not yield a sampling distribution for W itself.

## 14. Recommended next research step (IMPLEMENTED)

Persist per-seed artifacts in the research pipeline. **This step is complete.**
`run_research_experiments.py --persist-raw` writes `rank_matrix` (100 seeds × 122
encoded features) and `local_shap_values` (100 seeds × 100 applicants × 122
features) per stochastic condition to `evaluation/research_raw_100_seeds_final.npz`.
The 100-seed protocol was re-run once with this flag (same seeds 0–99, same
evaluation sample, same preprocessing). `recompute_local_stability.py` re-derives
all local metrics from the raw outputs and confirms aggregate consistency to
< 1e-6. This changed only what is *stored*, not the experimental design.

## 15. Reproducibility information

| What | Value / file |
| --- | --- |
| Research runner | `run_research_experiments.py` (`--seeds 100 --workers 4 --include-determinism-check --persist-raw`) |
| Statistical analysis | `analyze_research_results.py` (reads `evaluation/research_results_100_seeds.json`) |
| Local recompute & validation | `recompute_local_stability.py` (reads `evaluation/research_raw_100_seeds_final.npz`) |
| Final results (aggregate) | `evaluation/research_results_100_seeds_final.json` |
| Raw per-seed data | `evaluation/research_raw_100_seeds_final.npz` (manifest: `evaluation/research_raw_manifest_final.json`) |
| Recomputed local analysis | `evaluation/research_local_recomputed_final.json` |
| Statistical tests output | `evaluation/research_statistical_analysis.json` |
| Determinism check W | 1.0 (subsample=1.0, colsample_bytree=1.0) |
| Production artifact SHA-256 | `1e34d83b3d5b3c5033812150a18ae78b73d2b4328438c2bd286945e337d8bd9f` (unchanged) |
| Key package versions | xgboost 3.4.1, shap 0.52.0, scikit-learn 1.9.0, numpy 2.4.3, scipy 1.18.1 |
| Seeds | 0–99 (range(100)), plus 2 for determinism check |
| Evaluation sample | X_test[:512] for SHAP evaluation; first 100 rows for local analysis |

## 16. Final research status

**RESEARCH COMPLETE: YES**

All criteria for freezing the research design are met:

- **Existing research questions answered:** H1 (proxies appear stable & important)
  is supported; H2 (robust across conditions) is supported for importance and top-k
  enrichment, with the row_sampling stability sign-flip documented as a boundary
  condition; the global↔local decoupling is documented.
- **Raw outputs persisted:** per-seed rank matrices and per-applicant attribution
  vectors are in `evaluation/research_raw_100_seeds_final.npz`.
- **Local metrics validated:** row_sampling degeneracy confirmed over all 100 seeds
  via `recompute_local_stability.py`; the degenerate metric is flagged and excluded
  from applicant-level inference.
- **Statistical analysis defensible:** 7 test families, Holm-Bonferroni within each
  family, correct observational units (feature-level / applicant-level / paired),
  no test on aggregate Kendall's W, no test on global↔local (n=3–4).
- **Conclusions scoped correctly:** "in this dataset and experimental setting"
  throughout; no generalisation, causality, or production claims.
- **Tests pass:** `tests/test_research_experiments.py` (4 passed); full suite
  (27 passed). Production artifact hash-verified unchanged.
- **Documentation matches implementation.**

**Next phase:** optimisation, cleanup, figure generation, paper writing.
No additional experiments are required.


