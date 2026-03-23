# MIMIC-IV Dataset Design Notes (Tabular-GIA)

This document explains the design choices behind the MIMIC-derived datasets used in `tabular_gia`.

## Why we built this ourselves
- We chose to implement preprocessing in this repo (instead of relying on external preprocessed artifacts) to keep full control over:
  - temporal validity of features,
  - threat-model assumptions,
  - reproducibility and debugging.
- This avoids ambiguity about where leakage might have been introduced.

## Source data and row definition
- Source: MIMIC-IV v3.1 tables under `hosp/` (and optionally `icu/`, though current builder is admission-focused).
- Current dataset row unit: **one hospital admission** (`hadm_id`).
- A patient (`subject_id`) can appear multiple times across rows (multiple admissions). IDs are deidentified in MIMIC.

## Prediction point
- Single prediction point for all generated tasks: **at hospital admission time** (`admittime`).
- Design rule: features must be available at or before this point.

## Tasks and targets
- Binary:
  - Target: `hospital_expire_flag` (in-hospital mortality indicator).
- Multiclass:
  - Target: `mortality_horizon` with classes:
    - `alive`
    - `died_0_48h`
    - `died_48h_7d`
    - `died_over_7d`
    - `died_unknown_time`
- Regression:
  - Target: `hospital_los_hours` = `dischtime - admittime` in hours.

## Tiered feature sets

### Tier 1 (admission baseline)
- Admission-time and demographic/context features:
  - `age_at_admit_est`, `admit_hour`, `admit_weekday`,
  - `had_ed`, `ed_los_hours`, `ed_to_admit_hours`,
  - `gender`, `admission_type`, `admission_location`, `insurance`,
  - `language`, `marital_status`, `race`, `anchor_year_group`,
  - plus history counters (`prior_admission_count`, `prior_expire_count`).

### Tier 2 (+ diagnosis history)
- Adds prior diagnosis history features from `diagnoses_icd.csv.gz`:
  - `prior_diag_top_*` indicator features for top-K diagnosis concepts,
  - `prior_diagnosis_event_count`.

### Tier 3 (+ medication/procedure history)
- Adds prior medication and procedure history:
  - `prior_med_top_*` indicator features from `prescriptions.csv.gz`,
  - `prior_medication_event_count`,
  - `prior_procedure_event_count` from `procedures_icd.csv.gz`.

## What "prior" means
- `prior_*` features are computed from **earlier admissions of the same subject only**.
- For each admission row:
  - value `1` in a `prior_*_top_*` feature means "this concept was seen in at least one earlier admission",
  - value `0` means "not seen previously".
- This is done to preserve admission-time validity and avoid same-stay future leakage.

## What "top-K" means
- For a one-to-many table (diagnoses or prescriptions):
  - count concept frequencies globally,
  - keep only the K most frequent concepts,
  - create stable indicator columns only for those concepts.
- Purpose:
  - keeps dimensionality manageable,
  - reduces extreme sparsity/noise from long-tail rare codes.

## ED-related fields
- `ED` stands for **Emergency Department**.
- `had_ed`: whether ED registration/discharge timestamps exist.
- `ed_los_hours`: time spent in ED (`edouttime - edregtime`).
- `ed_to_admit_hours`: delay from ED registration to hospital admission (`admittime - edregtime`).

## Leakage and temporal safeguards
- Current-admission event tables are not used directly as same-row one-hot indicators.
- One-to-many event features are transformed to per-admission and then shifted to prior-history features.
- For non-binary tasks, labels/time columns (for example `deathtime`, `dischtime`) are not included as input features.
- Strict mode is now the default in the builder:
  - only admissions with strictly earlier `admittime` contribute to `prior_*`,
  - subject-disjoint train/val/test split artifacts are written by default.
  - top-K diagnosis/medication concept sets are fitted on train split only by default.
  - legacy behavior can be restored with:
    - `--allow-same-time-prior`
    - `--allow-subject-overlap-split`
    - `--allow-global-topk`

## YAML schema choice
- Generated YAML files explicitly list:
  - `numerical_columns`,
  - `categorical_columns`.
- Dataloader now respects these lists when present, instead of fully inferring types.
- This avoids re-treating binary history indicators as categoricals and re-one-hot expanding them.
- Optional strict split generation writes train/val/test CSVs and stores:
  - `has_val_split`
  - `has_test_split`
  in the YAML for explicit external split loading.

## Known caveats
- `prior_expire_count` is expected to be very sparse in real data.
  - Most admissions have `0`; non-zero values may exist due to data edge cases.
- `age_at_admit_est` uses deidentified anchor-year logic and is an estimate.
- Some targets (for example LOS) are inherently post-admission outcomes; this is acceptable because they are targets, not features.

## Rebuild commands
- From repo root:

```bash
python tabular_gia/data/build_mimic_iv_hospital_mortality.py --tier tier1 --task binary
python tabular_gia/data/build_mimic_iv_hospital_mortality.py --tier tier2 --task binary
python tabular_gia/data/build_mimic_iv_hospital_mortality.py --tier tier3 --task binary

python tabular_gia/data/build_mimic_iv_hospital_mortality.py --tier tier2 --task multiclass
python tabular_gia/data/build_mimic_iv_hospital_mortality.py --tier tier2 --task regression

# Legacy opt-out example (not recommended)
python tabular_gia/data/build_mimic_iv_hospital_mortality.py --tier tier3 --task binary --allow-same-time-prior --allow-subject-overlap-split --allow-global-topk
```

- Quick smoke build:

```bash
python tabular_gia/data/build_mimic_iv_hospital_mortality.py --tier tier3 --task binary --max-rows 50000
```
