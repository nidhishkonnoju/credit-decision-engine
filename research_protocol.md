# SHAP Reliability Research Protocol

## Research question

Can SHAP stability provide misleading evidence of explanation reliability when a credit underwriting-tier model contains audited near-perfect target proxies?

This protocol studies the existing `Approved_Flag` underwriting tier. It does not
claim to predict future default and does not use a proxy-inclusive model for any
decision-support output.

## Claim discipline

`Credit_Score` and the fields listed in `AUDITED_LABEL_PROXY_COLUMNS` are called
**audited near-perfect target proxies** in code. The project has strong empirical
evidence of association (for example, `Credit_Score` has an out-of-fold shallow
single-field AUC near 1.0), but association alone does not prove target-construction
leakage. Upgrade that wording only if the target's construction is independently
documented.

## Feature policies

| Policy | Purpose | Allowed in production scoring |
| --- | --- | --- |
| `audited` | Existing model feature policy; removes protected fields and audited proxies. | Yes |
| `proxy_inclusive_diagnostic` | Research-only comparison to test whether a proxy can look stable. | No |

## Pre-specified conditions

The same fixed held-out evaluation applicants are explained in every condition.
All models use one CPU thread to make the deterministic check meaningful.

| Condition | Outer training sample | `subsample` | `colsample_bytree` | Role |
| --- | --- | ---: | ---: | --- |
| `bundled_baseline` | Different stratified 80% samples | 0.8 | 0.8 | Reproduces the bundled seed-style treatment. |
| `row_sampling` | Fixed | 0.8 | 1.0 | Row-sampling robustness condition. |
| `feature_sampling` | Fixed | 1.0 | 0.8 | Feature-sampling robustness condition. |
| `determinism_check` | Fixed | 1.0 | 1.0 | Sanity check only; expected global concordance is 1.0. |

The deterministic check is not an empirical result: a lower value signals an
implementation or numerical reproducibility issue.

## Outcomes and units of analysis

- **Feature-level global stability:** mean SHAP rank, rank variance, rank range,
  and a derived `1 / (1 + rank_variance)` score. Kendall's W is reported as a
  descriptive aggregate across feature rankings.
- **Applicant-level local stability:** pairwise top-5 SHAP-reason Jaccard overlap
  and attribution cosine similarity for the same applicant across model fits.
- **Proxy strength:** three-fold out-of-fold AUC of a shallow, single-feature tree,
  symmetrised to the interval 0.5–1.0. This is association, not causal provenance.

The final analysis should compare paired per-feature and per-applicant distributions
between conditions. It should not run a significance test on one aggregate Kendall's
W value per condition.

## Running the study

Start with a pilot:

```powershell
python .\run_research_experiments.py --seeds 20 --workers 4 --include-determinism-check
```

For the final recorded run, use 100 seeds and retain the JSON output with the exact
command and package versions:

```powershell
python .\run_research_experiments.py --seeds 100 --workers 4 --include-determinism-check --output .\evaluation\research_results_100_seeds.json
```

The JSON output is deliberately machine-readable so figures and statistical tests
can be generated without re-running the models.

The default primary run uses `proxy_inclusive_diagnostic`: the comparison of
audited proxies and ordinary features happens within that same fitted model. Add
`--policies audited proxy_inclusive_diagnostic` only for the optional feature-policy
sensitivity replication. The deterministic control uses two fits by default because
its expected result is exact agreement, not a distributional estimate.

## Relationship to prior work

Lin and Wang (2025) motivates the seed-based SHAP-ranking baseline. The protocol
does not claim that SHAP stability, resampling, or local fragility are unexplored.
It asks the narrower question of whether a stability criterion can fail to flag a
problematic proxy in this specified credit-risk/XGBoost setting. Recent work on
gradient-boosting mechanisms such as first-mover bias should be cited as related,
distinct mechanism-focused evidence rather than presented as a contradiction.
