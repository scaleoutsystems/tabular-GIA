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
    # Default build: tier3 binary with a deterministic 30k-row sample
    python tabular_gia/data/build_mimic_iv_hospital_mortality.py

    # Other binary variants
    python tabular_gia/data/build_mimic_iv_hospital_mortality.py --tier tier1 --task binary
    python tabular_gia/data/build_mimic_iv_hospital_mortality.py --tier tier2 --task binary
    python tabular_gia/data/build_mimic_iv_hospital_mortality.py --tier tier3 --task binary

    # Multiclass / regression variants
    python tabular_gia/data/build_mimic_iv_hospital_mortality.py --tier tier2 --task multiclass
    python tabular_gia/data/build_mimic_iv_hospital_mortality.py --tier tier2 --task regression

    # Explicit row-cap override
    python tabular_gia/data/build_mimic_iv_hospital_mortality.py --tier tier3 --task binary --max-rows 50000

Notes:
- ED = Emergency Department.
- The script expects compressed MIMIC tables (*.csv.gz) under:
  tabular_gia/data/mimic-iv-3.1 unextracted/mimic-iv-3.1/{hosp,icu}
- Default output is a single full-cohort CSV intended for dataloader-side splitting.
- Use optional flags for alternate behavior:
  --write-subject-disjoint-splits
  --allow-same-time-prior
  --allow-global-topk
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


def _prior_counts_strictly_earlier(
    df: pd.DataFrame,
    value_cols: list[str],
    *,
    subject_col: str = "subject_id",
    time_col: str = "admittime",
) -> pd.DataFrame:
    """Compute prior counts using only rows with strictly earlier timestamps.

    Rows sharing the same (subject_id, admittime) do not contribute to each other's priors.
    """
    key_cols = [subject_col, time_col]
    tmp = df[key_cols + value_cols].copy()
    tmp[value_cols] = tmp[value_cols].fillna(0)
    for c in value_cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0)

    by_time = (
        tmp.groupby(key_cols, sort=False, dropna=False)[value_cols]
        .sum()
        .reset_index()
    )
    by_time_prior = by_time.copy()
    by_time_prior[value_cols] = (
        by_time.groupby(subject_col, sort=False)[value_cols].cumsum() - by_time[value_cols]
    )

    # Keep original row order.
    prior = tmp[key_cols].merge(by_time_prior, on=key_cols, how="left", sort=False)[value_cols]
    prior.index = df.index
    return prior.fillna(0)


def _load_event_values(
    *,
    table_path: Path,
    hadm_col: str,
    value_col: str,
    extra_value_col: str | None = None,
) -> pd.DataFrame:
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
    out = events.loc[mask, [hadm_col]].copy()
    out["_value"] = value_series.loc[mask]
    return out


def _fit_topk_value_map_from_table(
    *,
    table_path: Path,
    hadm_col: str,
    value_col: str,
    prefix: str,
    top_k: int,
    extra_value_col: str | None = None,
    fit_hadm_ids: set[int] | None = None,
) -> dict[str, str]:
    events = _load_event_values(
        table_path=table_path,
        hadm_col=hadm_col,
        value_col=value_col,
        extra_value_col=extra_value_col,
    )
    if fit_hadm_ids is not None:
        events = events[events[hadm_col].isin(fit_hadm_ids)].copy()
    return _build_topk_value_map(events["_value"], top_k=top_k, prefix=prefix)


def _add_prior_topk_presence(
    df: pd.DataFrame,
    *,
    table_path: Path,
    hadm_col: str,
    value_col: str,
    prefix: str,
    top_k: int,
    extra_value_col: str | None = None,
    strict_prior_by_time: bool = False,
    value_map: dict[str, str] | None = None,
    fit_hadm_ids: set[int] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    events = _load_event_values(
        table_path=table_path,
        hadm_col=hadm_col,
        value_col=value_col,
        extra_value_col=extra_value_col,
    )

    if value_map is None:
        fit_events = events if fit_hadm_ids is None else events[events[hadm_col].isin(fit_hadm_ids)].copy()
        value_map = _build_topk_value_map(fit_events["_value"], top_k=top_k, prefix=prefix)
    if not value_map:
        return df, []

    mapped_cols = list(value_map.values())
    events = events[events["_value"].isin(value_map.keys())].copy()
    events.drop_duplicates(subset=[hadm_col, "_value"], inplace=True)
    events["_col"] = events["_value"].map(value_map)
    events["_present"] = 1

    if events.empty:
        wide = pd.DataFrame(columns=[hadm_col] + mapped_cols)
    else:
        wide = (
            events.pivot_table(index=hadm_col, columns="_col", values="_present", aggfunc="max", fill_value=0)
            .reindex(columns=mapped_cols, fill_value=0)
            .reset_index()
        )

    current_cols = mapped_cols
    df = df.merge(wide, on=hadm_col, how="left")
    for c in current_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0).astype(np.int8)

    if strict_prior_by_time:
        prior_matrix = _prior_counts_strictly_earlier(df, current_cols)
    else:
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
    strict_prior_by_time: bool = False,
) -> pd.DataFrame:
    events = _read_csv_gz(table_path, usecols=[hadm_col])
    cur_col = f"_{out_col}_current"
    counts = events.groupby(hadm_col).size().rename(cur_col).reset_index()

    df = df.merge(counts, on=hadm_col, how="left")
    df[cur_col] = df[cur_col].fillna(0).astype(np.int32)

    if strict_prior_by_time:
        prior_counts = _prior_counts_strictly_earlier(df, [cur_col])[cur_col]
    else:
        prior_counts = df.groupby("subject_id", sort=False)[cur_col].cumsum() - df[cur_col]
    df[out_col] = prior_counts.astype(np.int32)
    df.drop(columns=[cur_col], inplace=True)
    return df


def _build_base_admissions(mimic_root: Path, strict_prior_by_time: bool = False) -> pd.DataFrame:
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

    if strict_prior_by_time:
        df["_row_count"] = 1
        df["prior_admission_count"] = _prior_counts_strictly_earlier(df, ["_row_count"])["_row_count"].astype(np.int32)
        df.drop(columns=["_row_count"], inplace=True)
        # Prior outcomes from earlier admissions of the same subject only.
        # This does NOT mean "patient died before treatment" for current admission.
        df["prior_expire_count"] = _prior_counts_strictly_earlier(df, ["hospital_expire_flag"])[
            "hospital_expire_flag"
        ].astype(np.int32)
    else:
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


def _cap_rows_deterministic(
    df: pd.DataFrame,
    *,
    max_rows: int,
    seed: int,
    task: str,
) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df

    if task == "binary" and "hospital_expire_flag" in df.columns:
        target = pd.to_numeric(df["hospital_expire_flag"], errors="coerce")
        if target.isin([0, 1]).all():
            class_counts = target.value_counts().sort_index()
            raw_alloc = (class_counts / float(len(df))) * int(max_rows)
            alloc = raw_alloc.astype(int)
            remainder = int(max_rows) - int(alloc.sum())
            if remainder > 0:
                fractional = (raw_alloc - alloc).sort_values(ascending=False)
                for cls in fractional.index.tolist()[:remainder]:
                    alloc.loc[cls] += 1
            alloc = alloc.clip(upper=class_counts)
            shortfall = int(max_rows) - int(alloc.sum())
            if shortfall > 0:
                spare = (class_counts - alloc).sort_values(ascending=False)
                for cls in spare.index.tolist():
                    if shortfall <= 0:
                        break
                    add = int(min(shortfall, spare.loc[cls]))
                    alloc.loc[cls] += add
                    shortfall -= add

            sampled_parts = []
            for cls in class_counts.index.tolist():
                take = int(alloc.loc[cls])
                if take <= 0:
                    continue
                cls_df = df.loc[target.eq(cls)]
                sampled_parts.append(cls_df.sample(n=take, random_state=seed))
            return pd.concat(sampled_parts, axis=0).sort_index().reset_index(drop=True)

    return df.sample(n=int(max_rows), random_state=seed).sort_index().reset_index(drop=True)


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


def _split_subject_disjoint(
    df: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (0 < val_frac < 1 and 0 < test_frac < 1 and (val_frac + test_frac) < 1):
        raise ValueError(
            f"Invalid split fractions: val_frac={val_frac}, test_frac={test_frac}. "
            "Need 0 < val,test < 1 and val+test < 1."
        )

    subjects = pd.Series(df[subject_col].dropna().unique())
    n_subjects = int(len(subjects))
    if n_subjects < 3:
        raise ValueError(f"Need at least 3 unique subjects for subject-disjoint split, got {n_subjects}.")

    rng = np.random.default_rng(seed)
    subject_values = subjects.to_numpy(copy=True)
    rng.shuffle(subject_values)

    n_test = max(1, int(np.floor(n_subjects * test_frac)))
    n_val = max(1, int(np.floor(n_subjects * val_frac)))
    if n_test + n_val >= n_subjects:
        n_test = max(1, min(n_subjects - 2, n_test))
        n_val = max(1, min(n_subjects - 1 - n_test, n_val))

    test_subjects = set(subject_values[:n_test].tolist())
    val_subjects = set(subject_values[n_test : n_test + n_val].tolist())
    train_subjects = set(subject_values[n_test + n_val :].tolist())

    train_df = df[df[subject_col].isin(train_subjects)].copy()
    val_df = df[df[subject_col].isin(val_subjects)].copy()
    test_df = df[df[subject_col].isin(test_subjects)].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError(
            f"Subject-disjoint split produced empty split: "
            f"train={len(train_df)} val={len(val_df)} test={len(test_df)}"
        )
    return train_df, val_df, test_df


def build_dataset(
    *,
    mimic_root: Path,
    out_dir: Path,
    tier: str,
    task: str,
    top_k_diagnoses: int = 100,
    top_k_medications: int = 100,
    max_rows: int | None = None,
    sample_seed: int = 42,
    strict_prior_by_time: bool = False,
    subject_disjoint_split: bool = False,
    split_seed: int = 42,
    split_val_frac: float = 0.15,
    split_test_frac: float = 0.15,
    topk_from_train_split: bool = True,
) -> tuple[Path, Path]:
    hosp = mimic_root / "hosp"

    df = _build_base_admissions(mimic_root, strict_prior_by_time=strict_prior_by_time)
    if max_rows is not None and max_rows > 0:
        df = _cap_rows_deterministic(
            df,
            max_rows=int(max_rows),
            seed=int(sample_seed),
            task=task,
        ).copy()

    base_numeric_cols = [
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
    missing_token = "UNKNOWN"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"mimic_admission_{tier}_{task}.csv"
    out_yaml = out_dir / f"mimic_admission_{tier}_{task}.yaml"

    if subject_disjoint_split:
        train_df, val_df, test_df = _split_subject_disjoint(
            df,
            subject_col="subject_id",
            val_frac=float(split_val_frac),
            test_frac=float(split_test_frac),
            seed=int(split_seed),
        )
        # Safety check: strict subject disjointness.
        train_subj = set(train_df["subject_id"].dropna().tolist())
        val_subj = set(val_df["subject_id"].dropna().tolist())
        test_subj = set(test_df["subject_id"].dropna().tolist())
        if train_subj.intersection(val_subj) or train_subj.intersection(test_subj) or val_subj.intersection(test_subj):
            raise RuntimeError("Subject-disjoint split violated: overlap detected across train/val/test.")

        # Fit concept maps on train split only by default, then apply to all splits.
        fit_hadm_ids = set(train_df["hadm_id"].tolist()) if topk_from_train_split else None
        diag_value_map = None
        med_value_map = None
        if tier in ("tier2", "tier3"):
            diag_value_map = _fit_topk_value_map_from_table(
                table_path=hosp / "diagnoses_icd.csv.gz",
                hadm_col="hadm_id",
                value_col="icd_code",
                extra_value_col="icd_version",
                prefix="diag_top",
                top_k=top_k_diagnoses,
                fit_hadm_ids=fit_hadm_ids,
            )
        if tier == "tier3":
            med_value_map = _fit_topk_value_map_from_table(
                table_path=hosp / "prescriptions.csv.gz",
                hadm_col="hadm_id",
                value_col="drug",
                prefix="med_top",
                top_k=top_k_medications,
                fit_hadm_ids=fit_hadm_ids,
            )

        splits_in = {"train": train_df, "val": val_df, "test": test_df}
        splits_out: dict[str, pd.DataFrame] = {}
        ref_model_cols: list[str] | None = None
        ref_numeric_cols: list[str] | None = None
        target_col: str | None = None
        blocked_features = {"hospital_expire_flag", "deathtime", "dischtime", "admittime", "anchor_age", "anchor_year"}
        if task == "binary":
            blocked_features.remove("hospital_expire_flag")

        for split_name, split_df in splits_in.items():
            cur = split_df.copy()
            numeric_cols = list(base_numeric_cols)

            if tier in ("tier2", "tier3"):
                cur, prior_diag_cols = _add_prior_topk_presence(
                    cur,
                    table_path=hosp / "diagnoses_icd.csv.gz",
                    hadm_col="hadm_id",
                    value_col="icd_code",
                    extra_value_col="icd_version",
                    prefix="diag_top",
                    top_k=top_k_diagnoses,
                    strict_prior_by_time=strict_prior_by_time,
                    value_map=diag_value_map,
                )
                cur = _add_prior_count_feature(
                    cur,
                    table_path=hosp / "diagnoses_icd.csv.gz",
                    hadm_col="hadm_id",
                    out_col="prior_diagnosis_event_count",
                    strict_prior_by_time=strict_prior_by_time,
                )
                numeric_cols.extend(prior_diag_cols)
                numeric_cols.append("prior_diagnosis_event_count")

            if tier == "tier3":
                cur, prior_med_cols = _add_prior_topk_presence(
                    cur,
                    table_path=hosp / "prescriptions.csv.gz",
                    hadm_col="hadm_id",
                    value_col="drug",
                    prefix="med_top",
                    top_k=top_k_medications,
                    strict_prior_by_time=strict_prior_by_time,
                    value_map=med_value_map,
                )
                cur = _add_prior_count_feature(
                    cur,
                    table_path=hosp / "prescriptions.csv.gz",
                    hadm_col="hadm_id",
                    out_col="prior_medication_event_count",
                    strict_prior_by_time=strict_prior_by_time,
                )
                cur = _add_prior_count_feature(
                    cur,
                    table_path=hosp / "procedures_icd.csv.gz",
                    hadm_col="hadm_id",
                    out_col="prior_procedure_event_count",
                    strict_prior_by_time=strict_prior_by_time,
                )
                numeric_cols.extend(prior_med_cols)
                numeric_cols.extend(["prior_medication_event_count", "prior_procedure_event_count"])

            cur, cur_target = _derive_target(cur, task=task)
            feature_cols = [c for c in (numeric_cols + categorical_cols) if c in cur.columns and c not in blocked_features]
            model_cols = [cur_target] + feature_cols
            cur = cur[model_cols].copy()
            cur.replace([np.inf, -np.inf], np.nan, inplace=True)
            _normalize_categorical(cur, categorical_cols, missing_token=missing_token)

            if ref_model_cols is None:
                ref_model_cols = model_cols
                ref_numeric_cols = [c for c in numeric_cols if c in model_cols]
                target_col = cur_target
            else:
                assert ref_model_cols is not None
                # Align split schema with train split schema.
                for col in ref_model_cols:
                    if col in cur.columns:
                        continue
                    cur[col] = missing_token if col in categorical_cols else 0
                cur = cur[ref_model_cols].copy()

            splits_out[split_name] = cur

        assert ref_model_cols is not None
        assert ref_numeric_cols is not None
        assert target_col is not None

        split_prefix = f"mimic_admission_{tier}_{task}"
        out_csv = out_dir / f"{split_prefix}.train.csv"
        out_val = out_dir / f"{split_prefix}.val.csv"
        out_test = out_dir / f"{split_prefix}.test.csv"
        splits_out["train"].to_csv(out_csv, index=False)
        splits_out["val"].to_csv(out_val, index=False)
        splits_out["test"].to_csv(out_test, index=False)

        meta = {
            "dataset_family": "mimic",
            "target": target_col,
            "task": task,
            "no_header": False,
            "missing_values": missing_token,
            "numerical_columns": [c for c in ref_numeric_cols if c in ref_model_cols],
            "categorical_columns": [c for c in categorical_cols if c in ref_model_cols],
        }
        if task == "binary":
            meta["binary_pos_weight"] = 13
        meta["has_val_split"] = out_val.name
        meta["has_test_split"] = out_test.name
    else:
        numeric_cols = list(base_numeric_cols)
        if tier in ("tier2", "tier3"):
            df, prior_diag_cols = _add_prior_topk_presence(
                df,
                table_path=hosp / "diagnoses_icd.csv.gz",
                hadm_col="hadm_id",
                value_col="icd_code",
                extra_value_col="icd_version",
                prefix="diag_top",
                top_k=top_k_diagnoses,
                strict_prior_by_time=strict_prior_by_time,
            )
            df = _add_prior_count_feature(
                df,
                table_path=hosp / "diagnoses_icd.csv.gz",
                hadm_col="hadm_id",
                out_col="prior_diagnosis_event_count",
                strict_prior_by_time=strict_prior_by_time,
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
                strict_prior_by_time=strict_prior_by_time,
            )
            df = _add_prior_count_feature(
                df,
                table_path=hosp / "prescriptions.csv.gz",
                hadm_col="hadm_id",
                out_col="prior_medication_event_count",
                strict_prior_by_time=strict_prior_by_time,
            )
            df = _add_prior_count_feature(
                df,
                table_path=hosp / "procedures_icd.csv.gz",
                hadm_col="hadm_id",
                out_col="prior_procedure_event_count",
                strict_prior_by_time=strict_prior_by_time,
            )
            numeric_cols.extend(prior_med_cols)
            numeric_cols.extend(["prior_medication_event_count", "prior_procedure_event_count"])

        df, target_col = _derive_target(df, task=task)
        blocked_features = {"hospital_expire_flag", "deathtime", "dischtime", "admittime", "anchor_age", "anchor_year"}
        if task == "binary":
            blocked_features.remove("hospital_expire_flag")
        feature_cols = [c for c in (numeric_cols + categorical_cols) if c in df.columns and c not in blocked_features]
        model_cols = [target_col, "subject_id"] + feature_cols
        df = df[model_cols].copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        _normalize_categorical(df, categorical_cols, missing_token=missing_token)

        df.to_csv(out_csv, index=False)
        meta = {
            "dataset_family": "mimic",
            "target": target_col,
            "task": task,
            "no_header": False,
            "missing_values": missing_token,
            "numerical_columns": [c for c in numeric_cols if c in model_cols],
            "categorical_columns": [c for c in categorical_cols if c in model_cols],
            "group_split_column": "subject_id",
            "exclude_cols": ["subject_id"],
            "has_val_split": None,
            "has_test_split": None,
        }
        if task == "binary":
            meta["binary_pos_weight"] = 13

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
        default="tier3",
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
        default=30000,
        help="Deterministic row cap (default 30000, 0 = all rows)",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used for deterministic row sampling when max_rows is set.",
    )
    parser.add_argument(
        "--allow-same-time-prior",
        action="store_true",
        help=(
            "Legacy behavior: allow same subject+admittime rows to contribute to each other's prior "
            "(disables strict prior-by-time)."
        ),
    )
    parser.add_argument(
        "--write-subject-disjoint-splits",
        action="store_true",
        help="Write subject-disjoint train/val/test split artifacts instead of a single full-cohort CSV.",
    )
    parser.add_argument(
        "--allow-subject-overlap-split",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-global-topk",
        action="store_true",
        help=(
            "Legacy behavior: fit top-K diagnosis/medication concepts using full dataset "
            "instead of train split only."
        ),
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for subject-disjoint split.",
    )
    parser.add_argument(
        "--split-val-frac",
        type=float,
        default=0.15,
        help="Validation subject fraction when strict subject-disjoint split (default) is enabled.",
    )
    parser.add_argument(
        "--split-test-frac",
        type=float,
        default=0.15,
        help="Test subject fraction when strict subject-disjoint split (default) is enabled.",
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
        sample_seed=int(args.sample_seed),
        strict_prior_by_time=(not bool(args.allow_same_time_prior)),
        subject_disjoint_split=(bool(args.write_subject_disjoint_splits) and not bool(args.allow_subject_overlap_split)),
        split_seed=int(args.split_seed),
        split_val_frac=float(args.split_val_frac),
        split_test_frac=float(args.split_test_frac),
        topk_from_train_split=(not bool(args.allow_global_topk)),
    )

    print(f"Wrote dataset CSV:  {out_csv}")
    print(f"Wrote dataset YAML: {out_yaml}")


if __name__ == "__main__":
    main()
