# Implementation Plan: Stability-Validated 5C Credit Appraisal Memo

**Work through phases in order — do not start a phase until the previous one's "Definition of Done" checks all pass.**

## Context for the agent

Repo: `credit-decision-engine`. Existing pipeline: `app/preprocessing.py` (leakage-audited feature set) → `app/modeling.py` (XGBoost + threshold tuning + a crude 5-run bootstrap SHAP "stability" check) → `app/inference.py` (single-applicant and batch CSV scoring) → `app/cam.py` (early-stage decision summary).

Base paper for this work: **Lin & Wang (2025), "SHAP Stability in Credit Risk Management," Risks 13(238)** — in `Research Papers/risks-13-00238.pdf`. Do not substitute a different stability method without checking this brief first.

**Their exact method** (replicate this precisely, it is load-bearing for the whole project):
- Fix the train/test split once.
- Train **100 XGBoost models** on the *same* training data, varying only `random_state` per model (not bootstrap-resampled rows — this is a seed-initialization study, not a resampling study).
- For each model, rank all features by mean absolute SHAP value → this gives 100 rank lists (an m×n matrix, m=100 models, n=features).
- Compute **Kendall's Coefficient of Concordance (Kendall's W)** across those 100 rank lists — overall, and separately for the top-5 features and for a mid-importance band.
- Their finding: W≈0.93 for top features (highly stable), W≈0.34 for mid-importance features (unstable) — i.e., stability is not uniform, and only stable features should appear in customer-facing or regulatory explanations.

Novelty this project is adding on top of that paper: **an automated Credit Appraisal Memo (CAM) generator that only uses stability-validated features, maps them onto the 5 C's of credit (Character, Capacity, Capital, Collateral, Conditions), and measures what the stability filter actually changes in the generated memo** — something Lin & Wang recommend but never build.

Dataset stays as-is (`Dataset/External_Cibil_Dataset.xlsx` + `Internal_Bank_Dataset.xlsx`). Do not switch datasets.

---

## Phase 0 — Stabilize the baseline before adding anything new

**Objective:** Make sure the existing repo is actually correct and reproducible, so nothing built on top of it inherits a silent bug.

**Tasks:**
1. Fix `_normalize_csv_values` in `app/inference.py`: it currently checks `dtype == "object"`, which never matches on pandas ≥3.0 where string columns default to `str`/`string` dtype. Update the check to catch both (e.g. `pd.api.types.is_string_dtype(normalized[column])` or check for `{"object", "string", "str"}`).
2. Pin `requirements.txt` to exact versions (`pandas==X.Y.Z`, not `>=`) using whatever versions are currently installed and verified working.
3. Add model persistence: after `fit_credit_model` runs once, serialize the fitted preprocessor + model + threshold (e.g. via `joblib.dump`) to a versioned artifact path, and add a `--retrain` flag to `main.py` so normal runs load the saved artifact instead of retraining from scratch every time.
4. Remove dead/unused files that don't match the current scope (`Unseen_Dataset.xlsx` if truly unreferenced, leftover `.github/copilot-instructions.md` / `microservice.prompt.md` scaffolding from the abandoned earlier architecture, if still present).

**Definition of Done / how to recheck:**
- Write a new test that builds a CSV with a comma-formatted number (e.g. `NETMONTHLYINCOME = "50,000"`) and asserts `batch_score_csv` succeeds and produces the numerically correct result. This test must fail on the old code and pass on the fix — confirm both.
- Run `python -m pytest -q` — all tests pass, including the new one.
- Run `main.py` twice in a row on the same input; second run should load the saved model (verify via a log line or timing — second run should be dramatically faster, no retraining).
- `pip install -r requirements.txt` in a clean virtualenv installs the exact pinned versions with no resolver warnings.

---

## Phase 1 — Rebuild the stability module to match Lin & Wang exactly

**Objective:** Replace the current 5-run bootstrap majority-vote stability check with the paper's actual method, so any later claim of "stability-validated" is defensible.

**Tasks:**
1. In `app/modeling.py`, implement `compute_seed_stability(X_train, y_train, X_test, base_params, n_seeds=100)`:
   - Trains `n_seeds` XGBoost models on the identical `X_train`/`y_train`, varying only `random_state`.
   - For each model, computes SHAP values on a fixed evaluation sample (e.g. the test set, or a fixed subsample if compute is a concern — decide and document which, since this affects reproducibility) and ranks features by mean |SHAP|.
   - Assembles the resulting rank matrix (100 rows × n_features).
2. Implement `kendalls_w(rank_matrix)` from scratch (or use a vetted library implementation) following the formula in the paper (Section 3, around their Kendall's W definition) — verify your implementation against their reported numbers style (W ranges 0 to 1, 1 = perfect agreement).
3. Compute and store: overall W across all features, W restricted to the top-5 features by mean rank, and W for the next band of features (mirror their top-5 vs. mid-importance split, adapted to this dataset's feature count).
4. Define an explicit **stability threshold** for "usable in a customer-facing memo" — e.g., a feature qualifies if it is in the top-K by mean rank *and* its individual rank variance/rank-range across the 100 models falls under a stated cutoff. Document the exact rule; don't leave it implicit.
5. Keep the old 5-run function only if you want a documented ablation ("naive stability check vs. proper stability check") — otherwise remove it so there isn't a second, weaker method living alongside the real one.

**Definition of Done / how to recheck:**
- Run the new stability computation and print: overall Kendall's W, top-5 W, mid-band W. Sanity check: top-5 W should be noticeably higher than mid-band W (mirroring the paper's qualitative pattern of stable-important vs. unstable-moderate features) — if it isn't, investigate before proceeding, since building the memo generator on a broken stability signal defeats the purpose.
- Confirm `n_seeds=100` actually varies only `random_state` and not the train/test split — add an assertion or a short test proving `X_train`/`y_train` are identical objects/values across all 100 fits.
- Time the full 100-model run once and record it — if it's impractically slow for iteration, cache the rank matrix to disk after computing it once, and document that caching in the README.
- Write a unit test with a synthetic dataset containing one dominant feature and several noise features; assert the dominant feature gets a high stability score and noise features get low ones. This is your regression test for "the stability method actually detects instability."

---

## Phase 2 — Formalize the feature → 5 C's mapping

**Objective:** Turn the existing informal `FEATURE_LABEL_MAP` (human-readable labels) into a citable classification: every model feature assigned to exactly one of Character, Capacity, Capital, Collateral, Conditions — with a stated rationale, since this mapping is itself a contribution being evaluated in the paper.

**Tasks:**
1. Create a new mapping structure, e.g. `FEATURE_TO_5C: dict[str, str]` in `app/cam.py`, covering every feature that survives the leakage audit (not just the ones currently in `FEATURE_LABEL_MAP`).
2. For each feature, write a one-line justification in a comment or a companion doc (e.g., "`Time_With_Curr_Empr` → Capacity: tenure length is a standard proxy for income stability" ). This justification list is something you'll lift almost directly into the paper's methodology section, so write it in full sentences now.
3. Decide and document what happens to features that don't cleanly fit one C (there will be some) — e.g., a documented "Conditions" catch-all, or an explicit secondary-category rule. Don't leave any feature silently unmapped.
4. Add a validation function that fails loudly (raises, or a startup check) if any feature used by the model is missing from the mapping — this prevents silent drift if the feature set changes later.

**Definition of Done / how to recheck:**
- Every feature in the final model's feature list has an entry in `FEATURE_TO_5C` — write a test asserting `set(model_features) <= set(FEATURE_TO_5C.keys())`.
- Manually review the mapping against the written justifications with a second person if possible (classmate, advisor) — this is a qualitative judgment call and is exactly the kind of thing a paper reviewer will scrutinize, so get a sanity check before it's load-bearing.
- Confirm the mapping file has no feature assigned to more than one category (unless you've explicitly designed for that and documented why).

---

## Phase 3 — Build the CAM generator

**Objective:** Generate a structured, officer-facing Credit Appraisal Memo per applicant, organized by the 5 C's, using only stability-validated SHAP reasons.

**Tasks:**
1. In `app/cam.py`, implement `generate_cam(applicant, model_result, stability_result)` that:
   - Computes SHAP values for this specific applicant.
   - Filters to only features that passed the Phase 1 stability threshold.
   - Groups the surviving reasons by their Phase 2 5C category.
   - Produces a structured output (a dict or dataclass first — Character/Capacity/Capital/Collateral/Conditions sections, each listing the applicant's relevant stable factors and their direction of impact) that can be rendered as plain text or a template-filled document.
   - Handles the edge case where a C-category has **zero** stable reasons for this applicant — decide and document what the memo says here (e.g., "No high-confidence factors identified for Conditions" rather than silently omitting the section) — this edge case is worth reporting as a finding, not hiding.
2. Wire this into `main.py` as a new output mode (e.g., `--cam` flag) alongside the existing decision output.
3. Keep the decision (approve/review/reject) and threshold logic separate from the memo — the memo explains, it doesn't re-decide.

**Definition of Done / how to recheck:**
- Generate memos for at least 3 applicants spanning approve, review, and reject outcomes; manually read all three — do the 5C sections make sense given the applicant's actual data? Check by hand against the raw feature values.
- Generate a memo for an applicant deliberately constructed (or found) to have very few stable features overall — confirm the empty-category case renders sensibly rather than crashing or producing an empty/confusing section.
- Write a test asserting every reason that appears in a generated memo also appears in that applicant's stability-filtered SHAP output (i.e., no reason in the memo is one that should have been filtered out) — this is the single most important correctness check in the whole project, since it's the thing the novelty claim depends on.

---

## Phase 4 — The evaluation (this is what makes it a paper, not just a feature)

**Objective:** Produce a number, not just a demo. Show what the stability filter actually changes.

**Tasks — pick at least one, ideally two:**
1. **Ablation:** Generate memos both with and without the stability filter (i.e., using raw top-K SHAP vs. stability-filtered SHAP) for the same set of applicants (aim for enough applicants to say something quantitative — e.g., 50–100 sampled from the test set). Count and report: how many memos would have included at least one feature that Phase 1 flagged as unstable, if the filter hadn't been applied. This is your headline result.
2. **Coverage:** Across the same applicant sample, report how often each of the 5 C's ends up with zero stable reasons. If some C is chronically empty, that's a real, reportable finding about a limitation of applying this taxonomy to this dataset.
3. **(Optional, adds real weight) Small human eval:** Have 5–10 people (classmates, an advisor, anyone with basic finance literacy) read a handful of anonymized memos and rate clarity/plausibility on a simple scale. Even n=5-10 is a legitimate result section if reported honestly as a small pilot, not a definitive claim.

**Definition of Done / how to recheck:**
- You have at least one table or chart with actual numbers (not just "the memos looked reasonable") that you could paste directly into a paper's Results section.
- The ablation, if done, is run on a fixed, saved applicant sample (not regenerated randomly each run) so the numbers are reproducible — save the sample IDs to a file and reuse them.
- Re-run the evaluation script twice and confirm identical output — if it's not deterministic, fix the seeding before reporting any number.

---

## Phase 5 — Documentation and paper-prep cleanup

**Objective:** Make sure the repo's own documentation would survive a reviewer reading it next to the code.

**Tasks:**
1. Rewrite `project_context_technical.md` to remove self-congratulatory language ("strongest claim," "methodological contribution") and replace with plain statements of what was done, what was found, and what the limitations are.
2. Add an explicit citation to Lin & Wang (2025) at every point in the code/docs where their method is being used (the stability computation, the design rationale for the memo filter) — one sentence each is enough, but it must be there.
3. Update `README.md` to describe the CAM feature, the stability methodology, and how to reproduce the Phase 4 evaluation numbers from a clean clone.
4. Draft a short internal "novelty ledger": one paragraph stating precisely what is new here relative to Lin & Wang (2025) — this paragraph becomes the core of your paper's Introduction, so write it now while the reasoning is fresh.

**Definition of Done / how to recheck:**
- Hand the README to someone who hasn't seen the project and ask them to reproduce the Phase 4 result from a clean clone, following only the README — if they get stuck, the README isn't done.
- Grep the repo for promotional phrases ("strongest," "significant contribution," "novel" used without a specific referent) and justify or remove each instance.

---

## Phase 6 — Final reproducibility QA (do this right before writing the paper)

**Objective:** Nothing in the paper should describe a result that a stranger cloning the repo can't reproduce.

**Tasks:**
1. Clone the repo fresh into a clean environment (or have a teammate do it).
2. Follow only the README, top to bottom: install, run tests, run the model, generate a CAM, run the Phase 4 evaluation.
3. Confirm every number you plan to put in the paper matches what this fresh run produces.

**Definition of Done / how to recheck:**
- `pytest` passes with zero modification to any file.
- The Phase 4 ablation number matches (exactly, or within a stated tolerance if there's any non-determinism you've documented) what's written in your draft results section.
- No file in the repo references infrastructure, datasets, or tools that aren't actually used (final check against the Phase 0 cleanup).
