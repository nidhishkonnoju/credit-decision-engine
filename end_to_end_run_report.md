# End-to-End Run Report — Credit Decision Engine

**Run date:** 2026-09-13 · **Start:** 15:21:54 IST · **Finish:** 15:36:46 IST · **Total wall time:** ~14 min 52 s
**Machine OS:** Windows 11 (win32) · **Working directory:** `d:\IITH` · **Result:** ✅ all 12 steps exit code 0

This report documents a full, fresh end-to-end execution of the project: environment validation,
the complete test suite, a production retrain with 100-seed SHAP stability analysis, single-JSON
scoring, 5C CAM generation with PDF export, CSV batch scoring, the Phase 4 stability-filter
ablation, and the pre-registered 100-seed SHAP reliability research protocol. Every command ran
unattended with logs captured to `output\e2e_20260913_1521\`.

## 1. Executive summary

| Area | Outcome |
| --- | --- |
| Dependency sanity (`pip check`) | ✅ No broken requirements |
| Test suite (`pytest`) | ✅ **26 passed**, 3 warnings, 144.80 s |
| Production retrain (`main.py --retrain`) | ✅ 51,336-row dataset, 48 model features, test ROC-AUC 0.7523, 100-seed stability W 0.9898 |
| Artifact reproducibility | ✅ Retrained artifact is **bit-for-bit identical** to the pre-existing one (same SHA-256) |
| Single-applicant JSON scoring | ✅ `REVIEW`, rejection risk 0.5002 |
| 5C CAM + PDF export | ✅ Structured memo produced; 2 valid PDFs generated (`%PDF-` verified) |
| CSV batch scoring (6 applicants) | ✅ APPROVE 2 · REVIEW 2 · REJECT 2, monotonic with risk profiles |
| Phase 4 stability-filter evaluation | ✅ Stability filter removed 27 unstable reasons across 50 fixed applicants |
| Research protocol (100 seeds) | ✅ Determinism check W = 1.0; W values **exactly reproduce** the committed results |

## 2. Environment

- **Python:** 3.12.4 (project virtualenv `d:\IITH\.venv`)
- **Dependency audit:** `python -m pip check` → *"No broken requirements found"*
- **Key package versions** (from `pip freeze`, full list in `output\e2e_20260913_1521\environment.log`):

| Package | Version | Package | Version |
| --- | --- | --- | --- |
| pandas | 3.0.1 | xgboost | 3.4.1 |
| numpy | 2.4.3 | shap | 0.52.0 |
| scikit-learn | 1.9.0 | joblib | 1.5.3 |
| openpyxl | 3.1.5 | reportlab | 5.0.1 |
| pytest | 9.1.1 | numba / llvmlite | 0.67.0 / 0.49.0 |
| scipy | 1.18.1 | matplotlib | 3.11.1 |

- **Input data verified present** (SHA-256 recorded in manifest):
  - `Dataset\External_Cibil_Dataset.xlsx` — `9873e0fe…`
  - `Dataset\Internal_Bank_Dataset.xlsx` — `788695e0…`
  - `applicant.example.json` — `fd720afa…`
  - `sample_applicants.csv` — `c6f437e3…`

## 3. How the run was orchestrated

The project is a local CLI pipeline (no server), so the end-to-end run was driven by a dedicated
runner script that executes each documented command with the venv Python, captures stdout+stderr
per step, records exit codes and durations, and checksums inputs/outputs:

- Runner: `output\e2e_20260913_1521\run_e2e.py`
- Machine-readable record: `output\e2e_20260913_1521\manifest.json`
- Per-step logs: `output\e2e_20260913_1521\<step>.log`
- Safety: the pre-existing `artifacts\credit_model_v1.joblib` was backed up to
  `output\e2e_20260913_1521\credit_model_before_run.joblib` before the retrain. Original files in
  `evaluation\` were left untouched — fresh Phase 4 / research outputs were redirected into the
  run folder using the scripts' own `--sample/--output` flags.

## 4. Step-by-step results

| # | Step | Command (venv Python) | Exit | Duration |
| -- | --- | --- | --: | --: |
| 1 | environment | `python -m pip freeze` | 0 | 1.05 s |
| 2 | dependency_check | `python -m pip check` | 0 | 0.86 s |
| 3 | tests | `python -m pytest tests -q --durations=10` | 0 | 149.06 s |
| 4 | train | `python main.py --retrain` | 0 | 318.14 s |
| 5 | load | `python main.py` | 0 | 3.92 s |
| 6 | json_score | `python main.py --input applicant.example.json` | 0 | 3.90 s |
| 7 | json_cam_pdf | `python main.py --input applicant.example.json --cam --pdf output\e2e_20260913_1521\applicant_cam.pdf` | 0 | 4.24 s |
| 8 | batch | `python main.py --batch sample_applicants.csv` | 0 | 4.25 s |
| 9 | csv_row | `python main.py --batch sample_applicants.csv --row 0` | 0 | 3.85 s |
| 10 | csv_row_cam_pdf | `python main.py --batch sample_applicants.csv --row 0 --pdf output\e2e_20260913_1521\csv_applicant_cam.pdf` | 0 | 4.19 s |
| 11 | phase4 | `python evaluate_stability_filter.py --sample output\e2e_20260913_1521\phase4_sample.json --output output\e2e_20260913_1521\phase4_results.json` | 0 | 33.45 s |
| 12 | research | `python run_research_experiments.py --seeds 100 --workers 4 --include-determinism-check --output output\e2e_20260913_1521\research_results_100_seeds.json` | 0 | 364.70 s |

## 5. Test suite

`pytest -q` over `tests/` — **26 passed, 3 warnings, 144.80 s (0:02:24)**.

- The suite covers metric generation, threshold logic (F2 search + review band), applicant
  inference, CSV batch normalization, stability filtering, 5C mapping, persistence round-trip,
  PDF smoke tests, and fairness checks that protected/label-proxy attributes never reach the
  model input set. Research-protocol unit tests (proxy strength, determinism control) also pass.
- The session fixture trains a compact validation model (`stability_n_seeds=5`,
  `cross_validation_folds=2`) to keep regression checks practical — production defaults remain
  100 seeds / 5 folds.
- Slowest items (setup dominated by dataset load + compact training):
  46.72 s and 42.67 s setups (PDF smoke / CAM naming tests), then 26.01 s
  (`test_every_raw_model_feature_has_exactly_one_5c_category`) and 25.26 s
  (`test_protected_fields_do_not_appear_in_final_feature_set`).
- The only 3 warnings are `PendingDeprecationWarning`s from the **shap** package internals
  (`shap/plots/colors/_colors.py`) — upstream and harmless.

## 6. Production training (`main.py --retrain`)

### 6.1 Preprocessing summary (from `train.log`)

| Metric | Value |
| --- | --- |
| Merged dataset shape | **51,336 rows × 88 columns** (bank ⋈ bureau on `PROSPECTID`) |
| Positive rate (P4 = reject label) | 11.46 % |
| Train / test positive rate | 11.46 % / 11.45 % (stratified split) |
| Model feature columns | **48** |
| Excluded protected/label-proxy fields | **39** (4 protected + 35 audited proxies) |
| Train / test split | 41,068 × 48 / 10,268 × 48 |

Sentinel `-99999` replaced with NaN; numeric median imputation with missing-indicator flags +
standard scaling; categorical most-frequent imputation + one-hot encoding.

### 6.2 Held-out test metrics (threshold 0.40 selected on validation F2)

| Metric | Value | Metric | Value |
| --- | ---: | --- | ---: |
| ROC-AUC | **0.7523** | Recall | **0.7730** |
| PR-AUC | 0.3045 | Precision | 0.1932 |
| KS statistic | 0.3648 | F1 | 0.3092 |
| Accuracy | 0.6044 | F2 | 0.4831 |
| Predicted reject rate | 0.4581 | Threshold | 0.40 |

The deliberately low precision with high recall reflects the cost-aware design (F2 threshold
search prioritizes catching risky applicants; a risky approval is assumed costlier than a
rejected safe applicant).

### 6.3 Cross-validation (5 stratified folds at the operating threshold 0.40)

mean ROC-AUC 0.7504 · mean PR-AUC 0.3077 · mean KS 0.3652 · mean recall 0.7745 ·
mean precision 0.1921 · mean F1 0.3079 · mean F2 0.4822 — consistent with the test split
(no sign of split overfitting).

### 6.4 SHAP seed-stability audit (100 seeds, fixed 512-row evaluation prefix)

| Statistic | Value |
| --- | ---: |
| Overall Kendall's W | **0.9898** |
| Top-5 features Kendall's W | **1.0** (perfectly concordant) |
| Diversity band (6 most rank-volatile features) W | 0.6679 |
| Memo-eligible stable features | **10** (top-10 mean-ranked with rank range ≤ 3) |

### 6.5 Artifact & bit-for-bit reproducibility

- Artifact written: `artifacts\credit_model_v1.joblib` (786,194 bytes).
- **SHA-256 of the freshly retrained artifact:
  `1e34d83b3d5b3c5033812150a18ae78b73d2b4328438c2bd286945e337d8bd9f`**
- SHA-256 of the pre-run backup (an artifact trained earlier the same day): **identical**.
- ⇒ Given the same dataset files and seeds, training is **fully deterministic end-to-end**
  (XGBoost `n_jobs=1` + fixed `random_state` values). The sample decision printed on the first
  test row was `REVIEW` at probability 0.5025 (between cutoffs 0.40 / 0.65).

## 7. Applicant scoring

### 7.1 Single-applicant JSON (`applicant.example.json`)

| Field | Value |
| --- | --- |
| Decision | **REVIEW** |
| Estimated rejection risk | **0.5002** |
| Decision threshold | 0.40 |
| Review threshold | 0.6542 |

Top SHAP reasons (stability-filtered): age of the oldest credit account **increases** risk,
monthly income **increases** risk, time with current employer **reduces** risk, percentage of
total balances in use **increases** risk, time since most recent payment **reduces** risk.

### 7.2 Structured 5C CAM + PDF

`--cam --pdf` produced the 5C memo and a valid PDF (`%PDF-` magic verified, 6,231 bytes):

| Category | Reasons |
| --- | --- |
| Character | Age_Oldest_TL ↑ · time_since_recent_payment ↓ |
| Capacity | NETMONTHLYINCOME ↑ · Time_With_Curr_Empr ↓ |
| Capital | pct_currentBal_all_TL ↑ |
| Collateral | *No high-confidence factors identified* |
| Conditions | *No high-confidence factors identified* |

### 7.3 CSV batch (`sample_applicants.csv`, 6 profiles)

Decision counts: **APPROVE 2 · REVIEW 2 · REJECT 2** — decisions track the engineered risk
profiles exactly, from "Prime Low-Risk Borrower" (0.1688) to "Distressed Subprime Default" (0.8424):

| Applicant | Profile | Decision | Rejection risk |
| --- | --- | --- | ---: |
| APP-1001 | Prime Low-Risk Borrower | APPROVE | 0.1688 |
| APP-1002 | Salaried Professional | APPROVE | 0.2711 |
| APP-1003 | Borderline Moderate-Risk | REVIEW | 0.5025 |
| APP-1004 | Thin-File Underwrite-Review | REVIEW | 0.5774 |
| APP-1005 | High-Risk Overleveraged | REJECT | 0.6632 |
| APP-1006 | Distressed Subprime Default | REJECT | 0.8424 |

### 7.4 Single-row CSV selection (`--row 0`) + PDF

`APP-1001 (Prime Low-Risk Borrower)` → **APPROVE @ 0.1688**, with all five reasons
risk-reducing (monthly income ↓, consumer credit limit ↓, unsecured credit limit ↓, current
balance utilization ↓, age of newest account ↓). CAM filled Character/Capacity/Capital/
Collateral sections, Conditions empty. PDF saved (6,169 bytes, `%PDF-` verified).

## 8. Phase 4 stability-filter evaluation (`evaluate_stability_filter.py`)

Ran on the fixed 50-row test sample (`phase4_sample.json`, indices + PROSPECTID validated
against the current split) with the freshly retrained artifact. Results written to
`output\e2e_20260913_1521\phase4_results.json` (the committed `evaluation\phase4_results.json`
was left untouched).

| Metric | Value |
| --- | ---: |
| Fixed sample size | 50 applicants |
| Memos that would have contained an unstable raw reason | **18 / 50 (36.00 %)** |
| Raw reason features across sample | 250 |
| Features surviving the stability filter | 223 |
| **Reasons removed by the stability filter** | **27** |
| Decision counts | APPROVE 28 · REVIEW 18 · REJECT 4 |
| Empty 5C sections | Capacity 17 · Capital 16 · Character 1 · Collateral 13 · Conditions 29 |

Interpretation: without filtering, over a third of customer memos would surface at least one
seed-unstable SHAP feature; the filter trims those while leaving 4/5 categories usually populated
(Conditions is empty for 58 % of the sample — expected, since few features map there).

## 9. Research protocol run (`run_research_experiments.py`, 100 seeds)

Pre-registered question: *can SHAP stability provide misleading evidence of explanation
reliability when inputs include audited near-perfect target proxies?* The run used the
research-only `proxy_inclusive_diagnostic` feature policy (production scoring is unaffected),
100 seeds, 4 parallel seed workers, 512-row global sample, 100-applicant local sample.

### 9.1 Univariate proxy strength (out-of-fold single-field AUC, symmetrised)

| Raw feature | Proxy AUC | Audit status |
| --- | ---: | --- |
| `Credit_Score` | **1.000** | audited near-perfect target proxy |
| `enq_L3m` | 0.8733 | audited near-perfect target proxy |
| `enq_L6m` | 0.8541 | audited near-perfect target proxy |
| `enq_L12m` | 0.8167 | audited near-perfect target proxy |
| `time_since_recent_enq` | 0.7867 | audited near-perfect target proxy |
| `tot_enq` | 0.7671 | audited near-perfect target proxy |
| `PL_enq_L6m` | 0.7445 | audited near-perfect target proxy |
| `pct_PL_enq_L6m_of_ever` | 0.7403 | audited near-perfect target proxy |

### 9.2 Condition results (100 seeds each; determinism control 2 fits)

| Condition | subsample / colsample | Outer resample | Global Kendall's W | Local mean top-k Jaccard | Local mean attribution cosine |
| --- | --- | --- | ---: | ---: | ---: |
| determinism_check | 1.0 / 1.0 | no | **1.000000** | 1.0000 | 1.0000 |
| bundled_baseline | 0.8 / 0.8 | yes | 0.894822 | 0.8115 | 0.9998 |
| row_sampling | 0.8 / 1.0 | no | 0.985470 | 0.6813 | 1.0000 |
| feature_sampling | 1.0 / 0.8 | no | 0.934871 | 0.8453 | 0.9998 |

Key observation consistent with the protocol's warning: `Credit_Score` — the near-perfect
proxy — ranks **first** in the top-5 mean-ranked features of *every* condition, i.e. a
seed-stability criterion alone does **not** flag the problematic proxy. The determinism check
returned exactly W = 1.0 as required (any lower value would indicate a reproducibility bug).

### 9.3 Reproducibility against the committed run

Kendall's W from this fresh run matches the committed `evaluation\research_results_100_seeds.json`
**exactly** for all four conditions (1.0 / 0.894822 / 0.98547 / 0.934871) — the research
pipeline is deterministic given the same data and seeds.

## 10. Files generated by this run

All artifacts are in `output\e2e_20260913_1521\` (the `output\` directory is Git-ignored):

| File | Size | Purpose |
| --- | ---: | --- |
| `manifest.json` | 5.2 KB | Machine-readable record: commands, exit codes, durations, hashes |
| `run_e2e.py` | 3.8 KB | The runner script (can be reused for future runs) |
| `environment.log` / `dependency_check.log` | <1 KB | `pip freeze` / `pip check` output |
| `tests.log` | 2.5 KB | Full pytest output incl. durations + warnings |
| `train.log` / `load.log` | ~2.2 KB | Retrain and artifact-load outputs |
| `json_score.log` / `json_cam_pdf.log` | ~3 KB | JSON applicant scoring + CAM/PDF outputs |
| `batch.log` / `csv_row.log` / `csv_row_cam_pdf.log` | ~5 KB | Batch and single-row CSV scoring outputs |
| `phase4.log` / `phase4_results.json` | 2.6 KB | Phase 4 stability-filter evaluation |
| `research.log` / `research_results_100_seeds.json` | 309 KB | Research protocol results |
| `applicant_cam.pdf` / `csv_applicant_cam.pdf` | 6.2 / 6.2 KB | Generated Credit Appraisal Memo PDFs (verified `%PDF-`) |
| `credit_model_before_run.joblib` | 786 KB | Backup of the pre-run model artifact |
| `runner.log` / `runner_error.log` | <1 KB | Runner stdout/stderr (stderr empty) |

Repository state impact: only `artifacts\credit_model_v1.joblib` was regenerated (byte-identical
to before), the two fresh evaluation JSONs were written to the run folder, and this report plus
the run folder were added. No tracked source files were modified.

## 11. Observations & limitations

1. **Determinism is a strength here.** Retraining reproduced the artifact bit-for-bit and the
   100-seed research run matched the committed W values exactly — reproducible model governance
   evidence for a paper or audit trail.
2. **Precision is low (0.1932) by design.** The F2-driven threshold search on an 11.5 % positive
   rate trades precision for recall (0.773). If the business wants fewer false rejections, the
   threshold policy (not just the model) would need revisiting.
3. **Test-suite runtime (~2.5 min) is dominated by data loading + compact retraining** in the
   session fixture; it is the main loop overhead for iterative development.
4. **3 pytest warnings** originate from `shap`'s colormap internals (upstream deprecation);
   no action needed in this repo.
5. **`Dataset\Unseen_Dataset.xlsx` exists but is not consumed** by the documented pipeline
   (`load_credit_data` reads only the two CIBIL/bank files); it appears intended for a separate
   holdout workflow.
6. **CAM section coverage is asymmetric** (Conditions empty for 58 % of the Phase 4 sample)
   because only 48 audited features map to the 5C taxonomy and few land in Conditions.
7. **Research vs production separation held:** the research run's proxy-inclusive diagnostic
   policy never touched the production artifact or any scoring path.

## 12. How to reproduce this run

```powershell
cd d:\IITH
.\.venv\Scripts\Activate.ps1
python -m pip check
pytest -q                                  # ~2.5 min
python main.py --retrain                   # ~5.5 min (100 stability seeds + 5-fold CV)
python main.py                             # artifact load + sample decision
python main.py --input .\applicant.example.json
python main.py --input .\applicant.example.json --cam --pdf .\output\cam.pdf
python main.py --batch .\sample_applicants.csv
python main.py --batch .\sample_applicants.csv --row 0
python evaluate_stability_filter.py        # ~35 s
python run_research_experiments.py --seeds 100 --workers 4 --include-determinism-check   # ~6 min
```

Or replay the entire instrumented run in one command (logs to a fresh folder under `output\`):

```powershell
python .\output\e2e_20260913_1521\run_e2e.py
```

*Report generated automatically from the captured logs and JSON outputs of the 2026-09-13 run.*

