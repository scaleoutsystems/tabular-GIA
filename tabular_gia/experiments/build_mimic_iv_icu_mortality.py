"""Build flat tabular MIMIC-IV mortality cohorts for tabular_gia experiments.

Supported cohorts:
  - icu_stay: one row per ICU stay (default)
  - admission: one row per hospital admission (larger cohort)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _read_csv_gz(path: Path, usecols: list[str], parse_dates: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        compression="gzip",
        usecols=usecols,
        parse_dates=parse_dates or [],
        low_memory=False,
    )


def _normalize_categorical(df: pd.DataFrame, columns: list[str], missing_token: str) -> None:
    for col in columns:
        df[col] = df[col].astype("string").fillna(missing_token)
        df[col] = df[col].replace({"": missing_token})


def _build_icu_stay_dataset(
    mimic_root: Path,
    out_dir: Path,
    max_rows: int | None = None,
) -> tuple[Path, Path]:
    hosp = mimic_root / "hosp"
    icu = mimic_root / "icu"

    icu_cols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "first_careunit",
        "intime",
    ]
    admissions_cols = [
        "subject_id",
        "hadm_id",
        "admittime",
        "edregtime",
        "edouttime",
        "admission_type",
        "admission_location",
        "insurance",
        "language",
        "marital_status",
        "race",
        "hospital_expire_flag",
    ]
    patients_cols = [
        "subject_id",
        "gender",
        "anchor_age",
        "anchor_year",
        "anchor_year_group",
    ]

    icustays = _read_csv_gz(icu / "icustays.csv.gz", usecols=icu_cols, parse_dates=["intime"])
    admissions = _read_csv_gz(hosp / "admissions.csv.gz", usecols=admissions_cols, parse_dates=["admittime"])
    admissions["edregtime"] = pd.to_datetime(admissions["edregtime"], errors="coerce")
    admissions["edouttime"] = pd.to_datetime(admissions["edouttime"], errors="coerce")
    patients = _read_csv_gz(hosp / "patients.csv.gz", usecols=patients_cols)

    cohort = icustays.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
    cohort = cohort.merge(patients, on="subject_id", how="left")

    # Keep only rows with valid binary target.
    cohort["hospital_expire_flag"] = pd.to_numeric(cohort["hospital_expire_flag"], errors="coerce")
    cohort = cohort[cohort["hospital_expire_flag"].isin([0, 1])].copy()
    cohort["hospital_expire_flag"] = cohort["hospital_expire_flag"].astype(int)

    # Feature engineering with admission-time and ICU-intime context.
    cohort["admit_to_icu_hours"] = (
        (cohort["intime"] - cohort["admittime"]).dt.total_seconds() / 3600.0
    )
    cohort["admit_hour"] = cohort["admittime"].dt.hour.astype("Int64")
    cohort["admit_weekday"] = cohort["admittime"].dt.dayofweek.astype("Int64")
    cohort["had_ed"] = (cohort["edregtime"].notna() & cohort["edouttime"].notna()).astype("Int64")
    cohort["ed_los_hours"] = (cohort["edouttime"] - cohort["edregtime"]).dt.total_seconds() / 3600.0
    cohort["ed_to_admit_hours"] = (cohort["admittime"] - cohort["edregtime"]).dt.total_seconds() / 3600.0
    cohort["anchor_year_offset"] = cohort["admittime"].dt.year - pd.to_numeric(
        cohort["anchor_year"], errors="coerce"
    )
    cohort["age_at_admit_est"] = pd.to_numeric(cohort["anchor_age"], errors="coerce") + cohort["anchor_year_offset"]
    cohort["age_at_admit_est"] = cohort["age_at_admit_est"].clip(lower=0, upper=120)

    # Columns chosen to avoid direct identifiers and obvious future leakage columns.
    keep_cols = [
        "hospital_expire_flag",
        "gender",
        "age_at_admit_est",
        "admit_to_icu_hours",
        "admit_hour",
        "admit_weekday",
        "had_ed",
        "ed_los_hours",
        "ed_to_admit_hours",
        "admission_type",
        "admission_location",
        "insurance",
        "language",
        "marital_status",
        "race",
        "first_careunit",
        "anchor_year_group",
    ]
    cohort = cohort[keep_cols].copy()

    # Replace inf with nan from malformed time differences.
    cohort.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_cols = [
        "age_at_admit_est",
        "admit_to_icu_hours",
        "admit_hour",
        "admit_weekday",
        "ed_los_hours",
        "ed_to_admit_hours",
    ]
    categorical_cols = [
        "gender",
        "had_ed",
        "admission_type",
        "admission_location",
        "insurance",
        "language",
        "marital_status",
        "race",
        "first_careunit",
        "anchor_year_group",
    ]

    missing_token = "UNKNOWN"
    _normalize_categorical(cohort, categorical_cols, missing_token=missing_token)

    if max_rows is not None and max_rows > 0:
        cohort = cohort.head(max_rows).copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "mimic_iv_icu_mortality.csv"
    out_yaml = out_dir / "mimic_iv_icu_mortality.yaml"

    cohort.to_csv(out_csv, index=False)

    meta = {
        "target": "hospital_expire_flag",
        "missing_values": missing_token,
        "numerical_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    return out_csv, out_yaml


def _build_admission_dataset(
    mimic_root: Path,
    out_dir: Path,
    strict_admission_time: bool = False,
    max_rows: int | None = None,
) -> tuple[Path, Path]:
    hosp = mimic_root / "hosp"
    icu = mimic_root / "icu"

    admissions_cols = [
        "subject_id",
        "hadm_id",
        "admittime",
        "edregtime",
        "edouttime",
        "admission_type",
        "admission_location",
        "insurance",
        "language",
        "marital_status",
        "race",
        "hospital_expire_flag",
    ]
    patients_cols = [
        "subject_id",
        "gender",
        "anchor_age",
        "anchor_year",
        "anchor_year_group",
    ]
    icu_cols = [
        "hadm_id",
        "first_careunit",
        "intime",
    ]

    admissions = _read_csv_gz(hosp / "admissions.csv.gz", usecols=admissions_cols, parse_dates=["admittime"])
    admissions["edregtime"] = pd.to_datetime(admissions["edregtime"], errors="coerce")
    admissions["edouttime"] = pd.to_datetime(admissions["edouttime"], errors="coerce")
    patients = _read_csv_gz(hosp / "patients.csv.gz", usecols=patients_cols)
    icustays = _read_csv_gz(icu / "icustays.csv.gz", usecols=icu_cols, parse_dates=["intime"])

    # earliest ICU stay per admission if present
    first_icu = (
        icustays.sort_values(["hadm_id", "intime"])
        .drop_duplicates(subset=["hadm_id"], keep="first")
        .rename(columns={"intime": "first_icu_intime", "first_careunit": "first_icu_careunit"})
    )

    cohort = admissions.merge(patients, on="subject_id", how="left")
    cohort = cohort.merge(first_icu[["hadm_id", "first_icu_intime", "first_icu_careunit"]], on="hadm_id", how="left")

    cohort["hospital_expire_flag"] = pd.to_numeric(cohort["hospital_expire_flag"], errors="coerce")
    cohort = cohort[cohort["hospital_expire_flag"].isin([0, 1])].copy()
    cohort["hospital_expire_flag"] = cohort["hospital_expire_flag"].astype(int)

    cohort["admit_hour"] = cohort["admittime"].dt.hour.astype("Int64")
    cohort["admit_weekday"] = cohort["admittime"].dt.dayofweek.astype("Int64")
    cohort["had_ed"] = (cohort["edregtime"].notna() & cohort["edouttime"].notna()).astype("Int64")
    cohort["ed_los_hours"] = (cohort["edouttime"] - cohort["edregtime"]).dt.total_seconds() / 3600.0
    cohort["ed_to_admit_hours"] = (cohort["admittime"] - cohort["edregtime"]).dt.total_seconds() / 3600.0
    cohort["has_icu"] = cohort["first_icu_intime"].notna().astype("Int64")
    cohort["admit_to_first_icu_hours"] = (
        (cohort["first_icu_intime"] - cohort["admittime"]).dt.total_seconds() / 3600.0
    )
    cohort["anchor_year_offset"] = cohort["admittime"].dt.year - pd.to_numeric(
        cohort["anchor_year"], errors="coerce"
    )
    cohort["age_at_admit_est"] = pd.to_numeric(cohort["anchor_age"], errors="coerce") + cohort["anchor_year_offset"]
    cohort["age_at_admit_est"] = cohort["age_at_admit_est"].clip(lower=0, upper=120)

    keep_cols = [
        "hospital_expire_flag",
        "gender",
        "age_at_admit_est",
        "admit_hour",
        "admit_weekday",
        "had_ed",
        "ed_los_hours",
        "ed_to_admit_hours",
        "admission_type",
        "admission_location",
        "insurance",
        "language",
        "marital_status",
        "race",
        "anchor_year_group",
    ]
    if not strict_admission_time:
        keep_cols.extend(
            [
                "has_icu",
                "admit_to_first_icu_hours",
                "first_icu_careunit",
            ]
        )
    cohort = cohort[keep_cols].copy()
    cohort.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_cols = [
        "age_at_admit_est",
        "admit_hour",
        "admit_weekday",
        "ed_los_hours",
        "ed_to_admit_hours",
    ]
    if not strict_admission_time:
        numeric_cols.append("admit_to_first_icu_hours")
    categorical_cols = [
        "gender",
        "had_ed",
        "admission_type",
        "admission_location",
        "insurance",
        "language",
        "marital_status",
        "race",
        "anchor_year_group",
    ]
    if not strict_admission_time:
        categorical_cols.extend(["has_icu", "first_icu_careunit"])

    missing_token = "UNKNOWN"
    _normalize_categorical(cohort, categorical_cols, missing_token=missing_token)

    if max_rows is not None and max_rows > 0:
        cohort = cohort.head(max_rows).copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "mimic_iv_admission_mortality.csv"
    out_yaml = out_dir / "mimic_iv_admission_mortality.yaml"

    cohort.to_csv(out_csv, index=False)

    meta = {
        "target": "hospital_expire_flag",
        "missing_values": missing_token,
        "numerical_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    return out_csv, out_yaml


def build_dataset(
    mimic_root: Path,
    out_dir: Path,
    cohort: str = "icu_stay",
    strict_admission_time: bool = False,
    max_rows: int | None = None,
) -> tuple[Path, Path]:
    if cohort == "icu_stay":
        return _build_icu_stay_dataset(mimic_root=mimic_root, out_dir=out_dir, max_rows=max_rows)
    if cohort == "admission":
        return _build_admission_dataset(
            mimic_root=mimic_root,
            out_dir=out_dir,
            strict_admission_time=strict_admission_time,
            max_rows=max_rows,
        )
    raise ValueError(f"Unknown cohort '{cohort}'. Use 'icu_stay' or 'admission'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MIMIC-IV flat ICU mortality dataset")
    parser.add_argument(
        "--mimic-root",
        default="tabular_gia/data/mimic-iv-3.1 unextracted/mimic-iv-3.1",
        help="Root directory containing MIMIC-IV hosp/ and icu/ folders",
    )
    parser.add_argument(
        "--out-dir",
        default="tabular_gia/data/binary/mimic_iv_icu_mortality",
        help="Output directory for CSV and YAML",
    )
    parser.add_argument(
        "--cohort",
        choices=["icu_stay", "admission"],
        default="icu_stay",
        help="Cohort type to build: ICU-stay (default) or larger admission-level cohort",
    )
    parser.add_argument(
        "--strict-admission-time",
        action="store_true",
        help=(
            "When used with --cohort admission, drop post-admission ICU availability/timing features "
            "(has_icu, first_icu_careunit, admit_to_first_icu_hours)."
        ),
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional row cap for quick tests (0 = all rows)",
    )
    args = parser.parse_args()

    mimic_root = Path(args.mimic_root)
    out_dir = Path(args.out_dir)
    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else None

    out_csv, out_yaml = build_dataset(
        mimic_root=mimic_root,
        out_dir=out_dir,
        cohort=args.cohort,
        strict_admission_time=args.strict_admission_time,
        max_rows=max_rows,
    )
    print(f"Wrote dataset CSV:  {out_csv}")
    print(f"Wrote dataset YAML: {out_yaml}")


if __name__ == "__main__":
    main()
