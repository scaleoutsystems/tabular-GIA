"""Build admission-time MIMIC-IV tabular datasets with progressive feature tiers.

Goals:
- Keep a single prediction point: admission-time.
- Support three tasks from the same base cohort:
  * binary: in-hospital mortality (hospital_expire_flag)
  * multiclass: mortality horizon classes
  * regression: hospital length-of-stay in hours
- Add complexity in controlled tiers:
  * tier1: strict admission-time baseline features
  * tier2: + prior-diagnosis history (top-K + counts)
  * tier3: + prior-medication history (top-K + counts) + prior procedure counts

All one-to-many features are converted to per-admission features using hadm_id, then shifted
into prior-history features per subject to preserve admission-time availability.

How to run (from repo root):
    # Binary tier1
    python tabular_gia/data/build_mimic_iv_icu_mortality.py --tier tier1 --task binary

    # Binary tier2 and tier3 (100 top diagnoses/medications by default)
    python tabular_gia/data/build_mimic_iv_icu_mortality.py --tier tier2 --task binary
    python tabular_gia/data/build_mimic_iv_icu_mortality.py --tier tier3 --task binary

    # Multiclass / regression variants
    python tabular_gia/data/build_mimic_iv_icu_mortality.py --tier tier2 --task multiclass
    python tabular_gia/data/build_mimic_iv_icu_mortality.py --tier tier2 --task regression

    # Quick smoke test on fewer rows
    python tabular_gia/data/build_mimic_iv_icu_mortality.py --tier tier3 --task binary --max-rows 50000

Notes:
- ED = Emergency Department.
- The script expects compressed MIMIC tables (*.csv.gz) under:
  tabular_gia/data/mimic-iv-3.1 unextracted/mimic-iv-3.1/{hosp,icu}
"""

from __future__ import annotations

import argparse
import re
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
        if col not in df.columns:
            continue
        df[col] = df[col].astype("string").fillna(missing_token)
        df[col] = df[col].replace({"": missing_token})


def _safe_slug(text: str, max_len: int = 50) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()
    if not slug:
        slug = "unknown"
    return slug[:max_len]


def _build_topk_value_map(values: pd.Series, top_k: int, prefix: str) -> dict[str, str]:
    top_values = values.value_counts().head(top_k).index.tolist()
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for value in top_values:
        base = f"{prefix}_{_safe_slug(value)}"
        candidate = base
        i = 2
        while candidate in used:
            candidate = f"{base}_{i}"
            i += 1
        used.add(candidate)
        mapping[value] = candidate
    return mapping


def _add_prior_topk_presence(
    df: pd.DataFrame,
    *,
    table_path: Path,
    hadm_col: str,
    value_col: str,
    prefix: str,
    top_k: int,
    extra_value_col: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    usecols = [hadm_col, value_col] + ([extra_value_col] if extra_value_col else [])
    events = _read_csv_gz(table_path, usecols=usecols)

    if extra_value_col:
        value_series = (
            events[extra_value_col].astype("string").fillna("UNK")
            + "_"
            + events[value_col].astype("string").fillna("UNK")
        )
    else:
        value_series = events[value_col].astype("string").fillna("UNK")

    value_series = value_series.str.strip()
    mask = value_series.notna() & (value_series != "")
    events = events.loc[mask, [hadm_col]].copy()
    events["_value"] = value_series.loc[mask]

    value_map = _build_topk_value_map(events["_value"], top_k=top_k, prefix=prefix)
    if not value_map:
        return df, []

    events = events[events["_value"].isin(value_map.keys())].copy()
    events.drop_duplicates(subset=[hadm_col, "_value"], inplace=True)
    events["_col"] = events["_value"].map(value_map)
    events["_present"] = 1

    wide = (
        events.pivot_table(index=hadm_col, columns="_col", values="_present", aggfunc="max", fill_value=0)
        .reset_index()
    )

    current_cols = [c for c in wide.columns if c != hadm_col]
    df = df.merge(wide, on=hadm_col, how="left")
    for c in current_cols:
        df[c] = df[c].fillna(0).astype(np.int8)

    prior_matrix = df.groupby("subject_id", sort=False)[current_cols].cumsum() - df[current_cols]
    prior_cols: list[str] = []
    for c in current_cols:
        pc = f"prior_{c}"
        df[pc] = (prior_matrix[c] > 0).astype(np.int8)
        prior_cols.append(pc)

    # Remove same-admission indicators to avoid using future/within-admission events directly.
    df.drop(columns=current_cols, inplace=True)
    return df, prior_cols


def _add_prior_count_feature(
    df: pd.DataFrame,
    *,
    table_path: Path,
    hadm_col: str,
    out_col: str,
) -> pd.DataFrame:
    events = _read_csv_gz(table_path, usecols=[hadm_col])
    cur_col = f"_{out_col}_current"
    counts = events.groupby(hadm_col).size().rename(cur_col).reset_index()

    df = df.merge(counts, on=hadm_col, how="left")
    df[cur_col] = df[cur_col].fillna(0).astype(np.int32)

    prior_counts = df.groupby("subject_id", sort=False)[cur_col].cumsum() - df[cur_col]
    df[out_col] = prior_counts.astype(np.int32)
    df.drop(columns=[cur_col], inplace=True)
    return df


def _build_base_admissions(mimic_root: Path) -> pd.DataFrame:
    hosp = mimic_root / "hosp"

    admissions_cols = [
        "subject_id",
        "hadm_id",
        "admittime",
        "dischtime",
        "deathtime",
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

    admissions = _read_csv_gz(
        hosp / "admissions.csv.gz",
        usecols=admissions_cols,
        parse_dates=["admittime", "dischtime", "deathtime", "edregtime", "edouttime"],
    )
    patients = _read_csv_gz(hosp / "patients.csv.gz", usecols=patients_cols)

    df = admissions.merge(patients, on="subject_id", how="left")

    df["hospital_expire_flag"] = pd.to_numeric(df["hospital_expire_flag"], errors="coerce")
    df = df[df["hospital_expire_flag"].isin([0, 1])].copy()
    df["hospital_expire_flag"] = df["hospital_expire_flag"].astype(int)

    df["admit_hour"] = df["admittime"].dt.hour.astype("Int64")
    df["admit_weekday"] = df["admittime"].dt.dayofweek.astype("Int64")

    # ED = Emergency Department
    df["had_ed"] = (df["edregtime"].notna() & df["edouttime"].notna()).astype("Int64")
    df["ed_los_hours"] = (df["edouttime"] - df["edregtime"]).dt.total_seconds() / 3600.0
    df["ed_to_admit_hours"] = (df["admittime"] - df["edregtime"]).dt.total_seconds() / 3600.0

    df["anchor_year_offset"] = df["admittime"].dt.year - pd.to_numeric(df["anchor_year"], errors="coerce")
    df["age_at_admit_est"] = pd.to_numeric(df["anchor_age"], errors="coerce") + df["anchor_year_offset"]
    df["age_at_admit_est"] = df["age_at_admit_est"].clip(lower=0, upper=120)

    # Sort once and keep this ordering for all cumulative prior-history features.
    df.sort_values(["subject_id", "admittime", "hadm_id"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["prior_admission_count"] = df.groupby("subject_id", sort=False).cumcount().astype(np.int32)
    # Prior outcomes from earlier admissions of the same subject only.
    # This does NOT mean "patient died before treatment" for current admission.
    df["prior_expire_count"] = (
        df.groupby("subject_id", sort=False)["hospital_expire_flag"].cumsum() - df["hospital_expire_flag"]
    ).astype(np.int32)

    return df


def _derive_target(df: pd.DataFrame, task: str) -> tuple[pd.DataFrame, str]:
    if task == "binary":
        return df, "hospital_expire_flag"

    if task == "multiclass":
        delta_h = (df["deathtime"] - df["admittime"]).dt.total_seconds() / 3600.0
        target = np.where(
            df["deathtime"].notna() & (delta_h <= 48),
            "died_0_48h",
            np.where(
                df["deathtime"].notna() & (delta_h <= 168),
                "died_48h_7d",
                np.where(
                    df["deathtime"].notna() & (delta_h > 168),
                    "died_over_7d",
                    np.where(df["hospital_expire_flag"].eq(1), "died_unknown_time", "alive"),
                ),
            ),
        )
        out = df.copy()
        out["mortality_horizon"] = target
        return out, "mortality_horizon"

    if task == "regression":
        out = df.copy()
        out["hospital_los_hours"] = (out["dischtime"] - out["admittime"]).dt.total_seconds() / 3600.0
        out = out[out["hospital_los_hours"].notna()].copy()
        out = out[out["hospital_los_hours"] >= 0].copy()
        return out, "hospital_los_hours"

    raise ValueError(f"Unknown task '{task}'.")


def _default_out_dir(task: str, tier: str) -> Path:
    if task == "binary":
        root = Path("tabular_gia/data/binary")
    elif task == "multiclass":
        root = Path("tabular_gia/data/multiclass")
    elif task == "regression":
        root = Path("tabular_gia/data/regression")
    else:
        raise ValueError(task)
    return root / f"mimic_admission_{tier}_{task}"


def build_dataset(
    *,
    mimic_root: Path,
    out_dir: Path,
    tier: str,
    task: str,
    top_k_diagnoses: int = 100,
    top_k_medications: int = 100,
    max_rows: int | None = None,
) -> tuple[Path, Path]:
    hosp = mimic_root / "hosp"

    df = _build_base_admissions(mimic_root)

    numeric_cols = [
        "age_at_admit_est",
        "admit_hour",
        "admit_weekday",
        "had_ed",
        "ed_los_hours",
        "ed_to_admit_hours",
        "prior_admission_count",
        "prior_expire_count",
    ]
    categorical_cols = [
        "gender",
        "admission_type",
        "admission_location",
        "insurance",
        "language",
        "marital_status",
        "race",
        "anchor_year_group",
    ]

    if tier in ("tier2", "tier3"):
        df, prior_diag_cols = _add_prior_topk_presence(
            df,
            table_path=hosp / "diagnoses_icd.csv.gz",
            hadm_col="hadm_id",
            value_col="icd_code",
            extra_value_col="icd_version",
            prefix="diag_top",
            top_k=top_k_diagnoses,
        )
        df = _add_prior_count_feature(
            df,
            table_path=hosp / "diagnoses_icd.csv.gz",
            hadm_col="hadm_id",
            out_col="prior_diagnosis_event_count",
        )
        numeric_cols.extend(prior_diag_cols)
        numeric_cols.append("prior_diagnosis_event_count")

    if tier == "tier3":
        df, prior_med_cols = _add_prior_topk_presence(
            df,
            table_path=hosp / "prescriptions.csv.gz",
            hadm_col="hadm_id",
            value_col="drug",
            prefix="med_top",
            top_k=top_k_medications,
        )
        df = _add_prior_count_feature(
            df,
            table_path=hosp / "prescriptions.csv.gz",
            hadm_col="hadm_id",
            out_col="prior_medication_event_count",
        )
        df = _add_prior_count_feature(
            df,
            table_path=hosp / "procedures_icd.csv.gz",
            hadm_col="hadm_id",
            out_col="prior_procedure_event_count",
        )
        numeric_cols.extend(prior_med_cols)
        numeric_cols.extend(["prior_medication_event_count", "prior_procedure_event_count"])

    df, target_col = _derive_target(df, task=task)

    # Drop labels that should never be used as input features for non-binary tasks.
    blocked_features = {"hospital_expire_flag", "deathtime", "dischtime", "admittime", "anchor_age", "anchor_year"}
    if task == "binary":
        blocked_features.remove("hospital_expire_flag")

    # Keep only expected feature columns + target.
    feature_cols = [c for c in (numeric_cols + categorical_cols) if c in df.columns and c not in blocked_features]
    df = df[[target_col] + feature_cols].copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    missing_token = "UNKNOWN"
    _normalize_categorical(df, categorical_cols, missing_token=missing_token)

    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows).copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"mimic_admission_{tier}_{task}.csv"
    out_yaml = out_dir / f"mimic_admission_{tier}_{task}.yaml"

    df.to_csv(out_csv, index=False)

    meta = {
        "target": target_col,
        "missing_values": missing_token,
        "numerical_columns": [c for c in numeric_cols if c in df.columns],
        "categorical_columns": [c for c in categorical_cols if c in df.columns],
    }
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)

    return out_csv, out_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Build admission-time MIMIC-IV tabular datasets")
    parser.add_argument(
        "--mimic-root",
        default="tabular_gia/data/mimic-iv-3.1 unextracted/mimic-iv-3.1",
        help="Root directory containing MIMIC-IV hosp/ and icu/ folders",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory; default is inferred from task+tier",
    )
    parser.add_argument(
        "--tier",
        choices=["tier1", "tier2", "tier3"],
        default="tier1",
        help="Feature complexity tier",
    )
    parser.add_argument(
        "--task",
        choices=["binary", "multiclass", "regression"],
        default="binary",
        help="Prediction task",
    )
    parser.add_argument(
        "--top-k-diagnoses",
        type=int,
        default=100,
        help="Top-K diagnosis concepts for tier2/tier3 prior-history features",
    )
    parser.add_argument(
        "--top-k-medications",
        type=int,
        default=100,
        help="Top-K medication concepts for tier3 prior-history features",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional row cap for quick tests (0 = all rows)",
    )
    args = parser.parse_args()

    mimic_root = Path(args.mimic_root)
    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(args.task, args.tier)
    max_rows = args.max_rows if args.max_rows and args.max_rows > 0 else None

    out_csv, out_yaml = build_dataset(
        mimic_root=mimic_root,
        out_dir=out_dir,
        tier=args.tier,
        task=args.task,
        top_k_diagnoses=args.top_k_diagnoses,
        top_k_medications=args.top_k_medications,
        max_rows=max_rows,
    )

    print(f"Wrote dataset CSV:  {out_csv}")
    print(f"Wrote dataset YAML: {out_yaml}")


if __name__ == "__main__":
    main()
