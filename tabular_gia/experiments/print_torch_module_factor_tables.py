from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

STRUCTURAL_FACTORS: list[tuple[str, str, list[Any]]] = [
    ("d_hidden", "Hidden Width", [32, 64, 128]),
    ("n_hidden_layers", "Hidden Layers", [1, 2, 3]),
]

MODULE_FACTORS: list[tuple[str, str, list[Any]]] = [
    ("norm", "Normalization", ["batchnorm", "layernorm"]),
    ("activation", "Activation", ["relu", "gelu"]),
    ("dropout", "Dropout", [0.0, 0.1]),
]

DATASET_CLEAN_NAMES = {
    "adult": "Adult",
    "california": "California Housing",
    "california_housing": "California Housing",
    "pandemic": "Pandemic Movement Office",
    "mimic": "MIMIC",
}

FACTOR_LEVEL_LABELS = {
    "norm": {
        "batchnorm": "BatchNorm",
        "layernorm": "LayerNorm",
    },
    "activation": {
        "relu": "ReLU",
        "gelu": "GELU",
    },
}

METRIC_ALIASES = {
    "reconstruction": "tableak_acc",
    "recon": "tableak_acc",
    "tableak": "tableak_acc",
}

UTILITY_METRIC_ALIASES = {
    "val_roc_auc": "val_roc_auc",
    "roc_auc": "val_roc_auc",
    "auc": "val_roc_auc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print LaTeX factor-sensitivity tables for the FedSGD torch-modules sweep "
            "using final attacked checkpoint reconstruction accuracy."
        )
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to a torch-modules experiment directory containing per-run aggregated stats.",
    )
    parser.add_argument(
        "--protocol-subdir",
        default="fedsgd",
        help="Protocol subdirectory inside the experiment directory. Default: fedsgd",
    )
    parser.add_argument(
        "--metric",
        default="tableak_acc",
        help="Metric to print from rounds_summary_stats.csv. Default: tableak_acc",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Number of decimals to print in each cell. Default: 3",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset name override for the caption.",
    )
    return parser.parse_args()


def _resolve_metric_name(metric_name: str) -> str:
    return METRIC_ALIASES.get(metric_name.strip().lower(), metric_name.strip())


def _resolve_utility_metric_name(metric_name: str) -> str:
    return UTILITY_METRIC_ALIASES.get(metric_name.strip().lower(), metric_name.strip())


def _dataset_name_from_dir(experiment_dir: Path) -> str:
    name = experiment_dir.name
    for candidate in ("adult", "california_housing", "california", "pandemic", "mimic"):
        if candidate in name:
            return candidate
    return name


def _clean_dataset_name(dataset_name: str) -> str:
    return DATASET_CLEAN_NAMES.get(dataset_name, dataset_name)


def _format_factor_level(factor_name: str, value: Any) -> str:
    if factor_name in FACTOR_LEVEL_LABELS:
        return FACTOR_LEVEL_LABELS[factor_name].get(str(value), str(value))
    if factor_name == "dropout":
        return f"{float(value):.1f}"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _load_run_metadata(run_dir: Path) -> dict[str, Any] | None:
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    dataset_path = str(cfg.get("dataset", {}).get("dataset_path", "unknown"))
    dataset_name = Path(dataset_path).stem
    model_cfg = cfg.get("model", {})
    return {
        "run_id": run_dir.name,
        "dataset_name": dataset_name,
        "batch_size": int(cfg["dataset"]["batch_size"]),
        "d_hidden": int(model_cfg["d_hidden"]),
        "n_hidden_layers": int(model_cfg["n_hidden_layers"]),
        "norm": str(model_cfg["norm"]),
        "dropout": float(model_cfg["dropout"]),
        "activation": str(model_cfg["activation"]),
    }


def _collect_final_rows(experiment_dir: Path, protocol_subdir: str, metric_name: str) -> pd.DataFrame:
    protocol_dir = experiment_dir / protocol_subdir
    if not protocol_dir.exists():
        raise FileNotFoundError(f"Protocol directory not found: {protocol_dir}")

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(protocol_dir.glob("run_*")):
        metadata = _load_run_metadata(run_dir)
        if metadata is None:
            continue

        stats_path = run_dir / "aggregated" / "rounds_summary_stats.csv"
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing rounds summary stats file: {stats_path}")

        stats = pd.read_csv(stats_path)
        if stats.empty:
            raise ValueError(f"Rounds summary stats file is empty: {stats_path}")

        mean_col = f"{metric_name}_mean"
        if mean_col not in stats.columns:
            raise ValueError(f"Metric '{metric_name}' not found in {stats_path}. Expected column '{mean_col}'.")

        final_row = stats.sort_values("round").iloc[-1]
        rows.append(
            {
                **metadata,
                "metric_value": float(final_row[mean_col]),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No usable run rows found under {protocol_dir}")
    return pd.DataFrame(rows)


def _collect_best_utility_rows(experiment_dir: Path, protocol_subdir: str, metric_name: str) -> pd.DataFrame:
    protocol_dir = experiment_dir / protocol_subdir
    if not protocol_dir.exists():
        raise FileNotFoundError(f"Protocol directory not found: {protocol_dir}")

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(protocol_dir.glob("run_*")):
        metadata = _load_run_metadata(run_dir)
        if metadata is None:
            continue

        stats_path = run_dir / "aggregated" / "fl_stats.csv"
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing FL stats file: {stats_path}")

        stats = pd.read_csv(stats_path)
        if stats.empty:
            raise ValueError(f"FL stats file is empty: {stats_path}")

        mean_col = f"{metric_name}_mean"
        std_col = f"{metric_name}_std"
        if mean_col not in stats.columns:
            raise ValueError(f"Metric '{metric_name}' not found in {stats_path}. Expected column '{mean_col}'.")

        phase_rows = stats[stats["phase"] == "checkpoint"].copy()
        if phase_rows.empty:
            phase_rows = stats.copy()

        best_row = phase_rows.sort_values(mean_col, ascending=False).iloc[0]
        rows.append(
            {
                **metadata,
                "utility_mean": float(best_row[mean_col]),
                "utility_std": float(best_row[std_col]) if std_col in best_row and pd.notna(best_row[std_col]) else 0.0,
                "utility_round": int(best_row["round"]),
            }
        )

    if not rows:
        raise FileNotFoundError(f"No usable utility rows found under {protocol_dir}")
    return pd.DataFrame(rows)


def _aggregate_factor_rows(final_rows: pd.DataFrame, factors: list[tuple[str, str, list[Any]]]) -> pd.DataFrame:
    aggregated_frames: list[pd.DataFrame] = []
    for factor_name, _, _ in factors:
        grouped = (
            final_rows.groupby(["dataset_name", "batch_size", factor_name], dropna=False)
            .agg(
                metric_mean=("metric_value", "mean"),
                metric_std=("metric_value", "std"),
                n_configs=("run_id", "count"),
            )
            .reset_index()
            .rename(columns={factor_name: "factor_level"})
        )
        grouped["factor_name"] = factor_name
        grouped["metric_std"] = grouped["metric_std"].fillna(0.0)
        aggregated_frames.append(grouped)
    return pd.concat(aggregated_frames, axis=0, ignore_index=True)


def _build_factor_frame(
    factor_rows: pd.DataFrame,
    factors: list[tuple[str, str, list[Any]]],
) -> pd.DataFrame:
    batches = sorted(int(value) for value in factor_rows["batch_size"].unique().tolist())
    ordered_columns: list[tuple[str, str, str]] = []
    for factor_name, factor_title, levels in factors:
        for level in levels:
            ordered_columns.append((factor_name, factor_title, _format_factor_level(factor_name, level)))

    data: dict[int, dict[tuple[str, str, str], str]] = {batch_size: {} for batch_size in batches}
    for factor_name, factor_title, levels in factors:
        factor_slice = factor_rows[factor_rows["factor_name"] == factor_name].copy()
        for batch_size in batches:
            batch_slice = factor_slice[factor_slice["batch_size"] == batch_size].copy()
            level_map = batch_slice.set_index("factor_level")[["metric_mean", "metric_std"]].to_dict(orient="index")
            for level in levels:
                entry = level_map.get(level)
                col_key = (factor_name, factor_title, _format_factor_level(factor_name, level))
                if entry is None:
                    data[batch_size][col_key] = "--"
                else:
                    mean_value = float(entry["metric_mean"])
                    std_value = float(entry["metric_std"])
                    data[batch_size][col_key] = (mean_value, std_value)

    frame = pd.DataFrame.from_dict(data, orient="index")
    frame = frame.reindex(index=batches)
    frame = frame.reindex(columns=pd.MultiIndex.from_tuples(ordered_columns))
    return frame


def _build_module_combo_extremes(
    final_rows: pd.DataFrame,
    *,
    batch_size: int,
    top_k: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    batch_rows = final_rows[final_rows["batch_size"] == batch_size].copy()
    grouped = (
        batch_rows.groupby(["norm", "activation", "dropout"], dropna=False)
        .agg(
            metric_mean=("metric_value", "mean"),
            metric_std=("metric_value", "std"),
        )
        .reset_index()
        .sort_values("metric_mean")
        .reset_index(drop=True)
    )
    grouped["metric_std"] = grouped["metric_std"].fillna(0.0)
    lowest = grouped.head(top_k).copy().reset_index(drop=True)
    highest = grouped.tail(top_k).copy().sort_values("metric_mean", ascending=False).reset_index(drop=True)
    return lowest, highest


def _format_cell(value: object, decimals: int) -> str:
    if value == "--" or value is None or (isinstance(value, float) and pd.isna(value)):
        return "--"
    mean_value, std_value = value
    return f"{float(mean_value):.{decimals}f} $\\pm$ {float(std_value):.{decimals}f}"


def format_latex_table(
    frame: pd.DataFrame,
    *,
    dataset_name: str,
    factor_group_name: str,
    label_suffix: str,
    decimals: int,
) -> str:
    if not isinstance(frame.columns, pd.MultiIndex):
        raise ValueError("Expected a MultiIndex frame with factor group columns.")

    factor_titles = [str(value) for value in frame.columns.get_level_values(1).unique().tolist()]
    top_levels = list(frame.columns)
    col_spec = "l" + "c" * len(top_levels)
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(
        f"\\caption{{Final reconstruction accuracy for {factor_group_name.lower()} on {dataset_name}. "
        f"Each cell reports mean $\\pm$ standard deviation across the remaining configurations at the final attacked checkpoint.}}"
    )
    lines.append(f"\\label{{tab:torch-modules-{label_suffix}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")

    header_top = "\\multicolumn{1}{l}{}"
    for factor_title in factor_titles:
        n_cols = sum(1 for _, title, _ in top_levels if title == factor_title)
        header_top += f" & \\multicolumn{{{n_cols}}}{{c}}{{{factor_title}}}"
    header_top += " \\\\"
    lines.append(header_top)

    header = "Batch size"
    for _, _, level_label in top_levels:
        header += f" & {level_label}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\hline")

    for batch_size in frame.index.tolist():
        row = frame.loc[batch_size]
        values = [_format_cell(row[col], decimals) for col in top_levels]
        lines.append(f"{int(batch_size)} & " + " & ".join(values) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def format_module_combo_extremes_table(
    lowest_by_batch: dict[int, pd.DataFrame],
    highest_by_batch: dict[int, pd.DataFrame],
    *,
    dataset_name: str,
    label_suffix: str,
    decimals: int,
) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(
        f"\\caption{{Lowest- and highest-reconstruction module combinations on {dataset_name} for batch sizes 8 and 32. "
        f"Reconstruction accuracy is averaged over width and depth and reported at the final attacked checkpoint.}}"
    )
    lines.append(f"\\label{{tab:torch-modules-{label_suffix}}}")
    lines.append("\\begin{tabular}{ccccc}")
    lines.append("\\hline")
    lines.append("Rank & Normalization & Activation & Dropout & Recon. Acc. \\\\")
    lines.append("\\hline")
    ordered_batches = sorted(lowest_by_batch)
    for batch_size in ordered_batches:
        lines.append(f"\\multicolumn{{5}}{{c}}{{Batch size {batch_size}: Lowest reconstruction accuracy}} \\\\")
        lines.append("\\hline")
        for idx, row in enumerate(lowest_by_batch[batch_size].itertuples(index=False), start=1):
            lines.append(
                f"{idx} & {_format_factor_level('norm', row.norm)} & "
                f"{_format_factor_level('activation', row.activation)} & "
                f"{_format_factor_level('dropout', row.dropout)} & "
                f"{float(row.metric_mean):.{decimals}f} $\\pm$ {float(row.metric_std):.{decimals}f} \\\\"
            )
        lines.append("\\hline")
        lines.append(f"\\multicolumn{{5}}{{c}}{{Batch size {batch_size}: Highest reconstruction accuracy}} \\\\")
        lines.append("\\hline")
        for idx, row in enumerate(highest_by_batch[batch_size].itertuples(index=False), start=1):
            lines.append(
                f"{idx} & {_format_factor_level('norm', row.norm)} & "
                f"{_format_factor_level('activation', row.activation)} & "
                f"{_format_factor_level('dropout', row.dropout)} & "
                f"{float(row.metric_mean):.{decimals}f} $\\pm$ {float(row.metric_std):.{decimals}f} \\\\"
            )
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _build_top_raw_configs(
    utility_rows: pd.DataFrame,
    *,
    batch_size: int,
    top_k: int = 5,
) -> pd.DataFrame:
    batch_rows = utility_rows[utility_rows["batch_size"] == batch_size].copy()
    ordered = batch_rows.sort_values(["utility_mean", "utility_std"], ascending=[False, True]).reset_index(drop=True)
    return ordered.head(top_k).copy().reset_index(drop=True)


def format_top_raw_utility_configs_table(
    top_by_batch: dict[int, pd.DataFrame],
    *,
    dataset_name: str,
    utility_metric_label: str,
    label_suffix: str,
    decimals: int,
) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(
        f"\\caption{{Top raw architectural configurations on {dataset_name} ranked by best validation {utility_metric_label} for batch sizes 8 and 32. "
        f"Scores are reported as mean $\\pm$ standard deviation across seeds at the best validation checkpoint for each configuration.}}"
    )
    lines.append(f"\\label{{tab:torch-modules-{label_suffix}}}")
    lines.append("\\begin{tabular}{ccccccc}")
    lines.append("\\hline")
    lines.append("Rank & Width & Layers & Normalization & Activation & Dropout & Val ROC-AUC \\\\")
    lines.append("\\hline")
    for batch_size in sorted(top_by_batch):
        lines.append(f"\\multicolumn{{7}}{{c}}{{Batch size {batch_size}}} \\\\")
        lines.append("\\hline")
        for idx, row in enumerate(top_by_batch[batch_size].itertuples(index=False), start=1):
            lines.append(
                f"{idx} & {int(row.d_hidden)} & {int(row.n_hidden_layers)} & "
                f"{_format_factor_level('norm', row.norm)} & "
                f"{_format_factor_level('activation', row.activation)} & "
                f"{_format_factor_level('dropout', row.dropout)} & "
                f"{float(row.utility_mean):.{decimals}f} $\\pm$ {float(row.utility_std):.{decimals}f} \\\\"
            )
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    metric_name = _resolve_metric_name(args.metric)
    utility_metric_name = _resolve_utility_metric_name("val_roc_auc")
    dataset_name = _clean_dataset_name(args.dataset_name or _dataset_name_from_dir(experiment_dir))

    final_rows = _collect_final_rows(experiment_dir=experiment_dir, protocol_subdir=args.protocol_subdir, metric_name=metric_name)
    utility_rows = _collect_best_utility_rows(
        experiment_dir=experiment_dir,
        protocol_subdir=args.protocol_subdir,
        metric_name=utility_metric_name,
    )
    factor_rows = _aggregate_factor_rows(final_rows, STRUCTURAL_FACTORS + MODULE_FACTORS)

    structural_frame = _build_factor_frame(
        factor_rows[factor_rows["dataset_name"] == final_rows["dataset_name"].iloc[0]].copy(),
        STRUCTURAL_FACTORS,
    )
    module_frame = _build_factor_frame(
        factor_rows[factor_rows["dataset_name"] == final_rows["dataset_name"].iloc[0]].copy(),
        MODULE_FACTORS,
    )
    lowest_by_batch: dict[int, pd.DataFrame] = {}
    highest_by_batch: dict[int, pd.DataFrame] = {}
    top_utility_by_batch: dict[int, pd.DataFrame] = {}
    for batch_size in (8, 32):
        lowest_by_batch[batch_size], highest_by_batch[batch_size] = _build_module_combo_extremes(
            final_rows,
            batch_size=batch_size,
            top_k=3,
        )
        top_utility_by_batch[batch_size] = _build_top_raw_configs(
            utility_rows,
            batch_size=batch_size,
            top_k=5,
        )

    dataset_slug = (args.dataset_name or _dataset_name_from_dir(experiment_dir)).replace("_", "-").lower()
    print(
        format_latex_table(
            structural_frame,
            dataset_name=dataset_name,
            factor_group_name="structural architectural factors",
            label_suffix=f"{dataset_slug}-structural",
            decimals=args.decimals,
        )
    )
    print()
    print(
        format_latex_table(
            module_frame,
            dataset_name=dataset_name,
            factor_group_name="module-level architectural factors",
            label_suffix=f"{dataset_slug}-modules",
            decimals=args.decimals,
        )
    )
    print()
    print(
        format_module_combo_extremes_table(
            lowest_by_batch,
            highest_by_batch,
            dataset_name=dataset_name,
            label_suffix=f"{dataset_slug}-module-extremes",
            decimals=args.decimals,
        )
    )
    print()
    print(
        format_top_raw_utility_configs_table(
            top_utility_by_batch,
            dataset_name=dataset_name,
            utility_metric_label="ROC-AUC",
            label_suffix=f"{dataset_slug}-top-utility-configs",
            decimals=args.decimals,
        )
    )


if __name__ == "__main__":
    main()
