from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

MODEL_CLEAN_NAMES = {
    "fttransformer": "FT-Transformer",
    "resnet": "ResNet",
    "small": "Small MLP",
}


DATASET_CLEAN_NAMES = {
    "adult": "Adult",
    "california": "California Housing",
    "california_housing": "California Housing",
    "mimic": "MIMIC-IV",
    "pandemic": "Private multiclass",
}


DATASET_LABEL_SLUGS = {
    "the private multiclass benchmark": "private-multiclass-benchmark",
    "Private multiclass": "private-multiclass",
}


METRIC_ALIASES = {
    "reconstruction": "tableak_acc",
    "recon": "tableak_acc",
    "tableak": "tableak_acc",
    "exact_match": "emr",
    "exact_match_rate": "emr",
    "strict_emr": "emr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a LaTeX batch-size table from an aggregated batch-size experiment directory."
    )
    parser.add_argument(
        "experiment_dirs",
        type=Path,
        nargs="+",
        help="Path to an experiment directory containing sweep_results.csv and sweep_runs.json.",
    )
    parser.add_argument(
        "--metric",
        default="tableak_acc",
        help="Metric column from rounds_summary_stats.csv to print. Default: tableak_acc",
    )
    parser.add_argument(
        "--emr-table",
        action="store_true",
        help="Print the strict exact match rate table using the emr metric.",
    )
    parser.add_argument(
        "--baseline-table",
        action="store_true",
        help="Print client marginal prior and uniform random reconstruction baselines by batch size.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=3,
        help="Number of decimals to print in each table cell. Default: 3",
    )
    parser.add_argument(
        "--caption",
        default=None,
        help="Optional LaTeX caption.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional LaTeX label.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset name override for the caption.",
    )
    return parser.parse_args()


def _resolve_metric_name(metric: str) -> str:
    return METRIC_ALIASES.get(metric.strip().lower(), metric.strip())


def _load_run_metadata(experiment_dir: Path) -> pd.DataFrame:
    sweep_runs_path = experiment_dir / "sweep_runs.json"
    if not sweep_runs_path.exists():
        raise FileNotFoundError(f"Missing sweep runs file: {sweep_runs_path}")

    sweep_runs = json.loads(sweep_runs_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for row in sweep_runs:
        overrides = row.get("overrides", {})
        dataset_overrides = overrides.get("dataset", {})
        model_overrides = overrides.get("model", {})
        rows.append(
            {
                "run_id": int(row["run_id"]),
                "protocol": str(row.get("protocol", "")).strip(),
                "batch_size": int(dataset_overrides["batch_size"]),
                "model_name": str(model_overrides.get("preset") or model_overrides.get("arch") or "unknown"),
            }
        )
    return pd.DataFrame(rows)


def _load_checkpoint_summary_stats(experiment_dir: Path, metadata: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in metadata.to_dict(orient="records"):
        run_id = int(record["run_id"])
        protocol = str(record["protocol"])
        stats_path = experiment_dir / protocol / f"run_{run_id:04d}" / "aggregated" / "rounds_summary_stats.csv"
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing rounds summary stats file: {stats_path}")

        stats = pd.read_csv(stats_path)
        if stats.empty:
            raise ValueError(f"Rounds summary stats file is empty: {stats_path}")

        mean_col = f"{metric_name}_mean"
        std_col = f"{metric_name}_std"
        if mean_col not in stats.columns or std_col not in stats.columns:
            raise ValueError(
                f"Metric '{metric_name}' not found in {stats_path}. "
                f"Expected columns '{mean_col}' and '{std_col}'."
            )

        stats = stats.sort_values("round").reset_index(drop=True)
        initialized_row = stats.iloc[0]
        trained_row = stats.iloc[-1]

        for stage_name, stats_row in (("Initialized", initialized_row), ("Trained", trained_row)):
            rows.append(
                {
                    "run_id": run_id,
                    "stage": stage_name,
                    "metric_mean": float(stats_row[mean_col]),
                    "metric_std": float(stats_row[std_col]),
                }
            )
    return pd.DataFrame(rows)


def _load_baseline_rows(experiment_dir: Path, metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in metadata.to_dict(orient="records"):
        run_id = int(record["run_id"])
        protocol = str(record["protocol"])
        batch_size = int(record["batch_size"])
        run_dir = experiment_dir / protocol / f"run_{run_id:04d}"
        for summary_path in sorted(run_dir.glob("seed_*/artifacts/rounds_summary.csv")):
            summary = pd.read_csv(summary_path)
            expected_cols = {"prior_tableak_acc", "random_tableak_acc"}
            missing_cols = expected_cols.difference(summary.columns)
            if missing_cols:
                raise ValueError(
                    f"Missing columns {sorted(missing_cols)} in {summary_path}."
                )
            for summary_row in summary.to_dict(orient="records"):
                rows.append(
                    {
                        "batch_size": batch_size,
                        "prior_tableak_acc": float(summary_row["prior_tableak_acc"]),
                        "random_tableak_acc": float(summary_row["random_tableak_acc"]),
                    }
                )
    if not rows:
        raise FileNotFoundError(
            f"No per-seed rounds_summary.csv files found under {experiment_dir}."
        )
    return pd.DataFrame(rows)


def _dataset_name_from_dir(experiment_dir: Path) -> str:
    name = experiment_dir.name
    for candidate in ("adult", "california", "pandemic", "mimic", "california_housing"):
        if candidate in name:
            return DATASET_CLEAN_NAMES[candidate]
    return name


def _ordered_models(model_names: list[str]) -> list[str]:
    preferred = ["fttransformer", "resnet", "small"]
    ordered = [name for name in preferred if name in model_names]
    ordered.extend(sorted(name for name in model_names if name not in preferred))
    return ordered


def _clean_model_name(model_name: str) -> str:
    return MODEL_CLEAN_NAMES.get(model_name, model_name)


def _label_slug(value: str) -> str:
    if value in DATASET_LABEL_SLUGS:
        return DATASET_LABEL_SLUGS[value]
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _default_caption(metric_name: str, dataset_name: str) -> str:
    if metric_name == "emr":
        return (
            f"Strict exact match rate at the initialized and trained attack checkpoints for {dataset_name}. "
            "EMR is the fraction of attacked rows reconstructed perfectly after assignment. "
            "Each cell reports mean $\\pm$ standard deviation across 3 seeds."
        )
    if metric_name == "tableak_acc":
        return (
            f"Reconstruction accuracy at the initialized and trained attack checkpoints for {dataset_name}. "
            "Initialized denotes the attack before any global model aggregation, and Trained denotes the final attacked checkpoint. "
            "Each cell reports mean $\\pm$ standard deviation across 3 seeds."
        )
    metric_label = metric_name.replace("_", " ")
    return (
        f"{metric_label} at the initialized and trained attack checkpoints for {dataset_name}. "
        "Each cell reports mean $\\pm$ standard deviation across 3 seeds."
    )


def build_table_frame(experiment_dir: Path, metric_name: str) -> pd.DataFrame:
    metadata = _load_run_metadata(experiment_dir)
    stats = _load_checkpoint_summary_stats(experiment_dir=experiment_dir, metadata=metadata, metric_name=metric_name)

    merged = stats.merge(metadata, on="run_id", how="inner")
    pivot = merged.pivot(index="batch_size", columns=["stage", "model_name"], values=["metric_mean", "metric_std"])

    ordered_batches = sorted(int(index) for index in pivot.index.tolist())
    ordered_models = _ordered_models(merged["model_name"].astype(str).unique().tolist())
    stage_order = ["Initialized", "Trained"]
    ordered_cols = pd.MultiIndex.from_product([["metric_mean", "metric_std"], stage_order, ordered_models])
    pivot = pivot.reindex(index=ordered_batches, columns=ordered_cols)
    return pivot


def build_baseline_table_frame(experiment_dir: Path) -> pd.DataFrame:
    metadata = _load_run_metadata(experiment_dir)
    rows = _load_baseline_rows(experiment_dir=experiment_dir, metadata=metadata)
    grouped = (
        rows.groupby("batch_size", as_index=True)
        .agg(
            prior_mean=("prior_tableak_acc", "mean"),
            prior_std=("prior_tableak_acc", "std"),
            random_mean=("random_tableak_acc", "mean"),
            random_std=("random_tableak_acc", "std"),
        )
        .sort_index()
    )
    return grouped


def format_latex_table(
    frame: pd.DataFrame,
    decimals: int,
    caption: str | None,
    label: str | None,
) -> str:
    if not isinstance(frame.columns, pd.MultiIndex):
        raise ValueError("Expected a MultiIndex column frame with metric_mean/metric_std by stage and model.")

    batch_sizes = [int(index) for index in frame.index.tolist()]
    stage_names = [str(stage) for stage in frame.columns.get_level_values(1).unique().tolist()]
    model_names = [str(model) for model in frame.columns.get_level_values(2).unique().tolist()]
    col_spec = "l" + "c" * (len(stage_names) * len(model_names))
    lines: list[str] = []

    if caption or label:
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{{label}}}")

    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")
    header_top = ""
    header_top += f"\\multicolumn{{1}}{{l}}{{}}"
    for stage_name in stage_names:
        header_top += f" & \\multicolumn{{{len(model_names)}}}{{c}}{{{stage_name}}}"
    header_top += " \\\\"
    lines.append(header_top)
    header = "Batch size"
    for _stage_name in stage_names:
        for model_name in model_names:
            header += f" & {_clean_model_name(model_name)}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\hline")

    for batch_size in batch_sizes:
        values: list[str] = []
        row = frame.loc[batch_size]
        for stage_name in stage_names:
            for model_name in model_names:
                mean_value = row[("metric_mean", stage_name, model_name)]
                std_value = row[("metric_std", stage_name, model_name)]
                if pd.isna(mean_value):
                    values.append("--")
                else:
                    values.append(f"{float(mean_value):.{decimals}f} $\\pm$ {float(std_value):.{decimals}f}")
        lines.append(f"{batch_size} & " + " & ".join(values) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")

    if caption or label:
        lines.append("\\end{table}")

    return "\n".join(lines)


def _default_baseline_caption(dataset_name: str) -> str:
    return (
        f"Client marginal prior and uniform random reconstruction baselines for {dataset_name} "
        "in the FedSGD batch size experiment. Values are computed on the same model runs and attacked "
        "checkpoints as the main reconstruction experiment and reported as mean $\\pm$ standard deviation."
    )


def _default_combined_baseline_caption() -> str:
    return (
        "Client marginal prior and uniform random reconstruction baselines for the benchmark "
        "FedSGD batch size experiments. Values are computed on the same model runs and attacked "
        "checkpoints as the main reconstruction experiments and reported as mean $\\pm$ standard deviation. "
        "Prior denotes client marginal prior reconstruction accuracy, and Random denotes uniform random reconstruction accuracy."
    )


def format_baseline_latex_table(
    frame: pd.DataFrame,
    decimals: int,
    caption: str | None,
    label: str | None,
) -> str:
    lines: list[str] = []
    if caption or label:
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{{label}}}")

    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\hline")
    lines.append(
        "Batch size & Client marginal prior recon. acc. & Uniform random recon. acc. \\\\"
    )
    lines.append("\\hline")

    for batch_size, row in frame.iterrows():
        prior_value = f"{float(row['prior_mean']):.{decimals}f} $\\pm$ {float(row['prior_std']):.{decimals}f}"
        random_value = f"{float(row['random_mean']):.{decimals}f} $\\pm$ {float(row['random_std']):.{decimals}f}"
        lines.append(f"{int(batch_size)} & {prior_value} & {random_value} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")

    if caption or label:
        lines.append("\\end{table}")

    return "\n".join(lines)


def format_combined_baseline_latex_table(
    frames_by_dataset: list[tuple[str, pd.DataFrame]],
    decimals: int,
    caption: str | None,
    label: str | None,
) -> str:
    all_batch_sizes = sorted(
        {
            int(batch_size)
            for _dataset_name, frame in frames_by_dataset
            for batch_size in frame.index.tolist()
        }
    )

    lines: list[str] = []
    if caption or label:
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\small")
        if caption:
            lines.append(f"\\caption{{{caption}}}")
        if label:
            lines.append(f"\\label{{{label}}}")

    lines.append("\\begin{tabular}{l" + "cc" * len(frames_by_dataset) + "}")
    lines.append("\\hline")
    header_top = "\\multicolumn{1}{l}{}"
    for dataset_name, _frame in frames_by_dataset:
        header_top += f" & \\multicolumn{{2}}{{c}}{{{dataset_name}}}"
    header_top += " \\\\"
    lines.append(header_top)

    header = "Batch size"
    for _dataset_name, _frame in frames_by_dataset:
        header += " & Prior & Random"
    header += " \\\\"
    lines.append(header)
    lines.append("\\hline")

    for batch_size in all_batch_sizes:
        values: list[str] = []
        for _dataset_name, frame in frames_by_dataset:
            if batch_size not in frame.index:
                values.extend(["--", "--"])
                continue
            row = frame.loc[batch_size]
            values.append(
                f"{float(row['prior_mean']):.{decimals}f} $\\pm$ {float(row['prior_std']):.{decimals}f}"
            )
            values.append(
                f"{float(row['random_mean']):.{decimals}f} $\\pm$ {float(row['random_std']):.{decimals}f}"
            )
        lines.append(f"{batch_size} & " + " & ".join(values) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")

    if caption or label:
        lines.append("\\end{table}")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    experiment_dirs = [experiment_dir.resolve() for experiment_dir in args.experiment_dirs]
    if len(experiment_dirs) > 1 and not args.baseline_table:
        raise ValueError("Multiple experiment directories are only supported with --baseline-table.")
    experiment_dir = experiment_dirs[0]
    metric_name = "emr" if args.emr_table else _resolve_metric_name(args.metric)
    dataset_name = args.dataset_name or _dataset_name_from_dir(experiment_dir)

    if args.baseline_table:
        if len(experiment_dirs) > 1:
            if args.dataset_name:
                raise ValueError("--dataset-name is only supported with a single experiment directory.")
            frames_by_dataset = [
                (_dataset_name_from_dir(path), build_baseline_table_frame(experiment_dir=path))
                for path in experiment_dirs
            ]
            caption = args.caption
            if caption is None:
                caption = _default_combined_baseline_caption()
            label = args.label
            if label is None:
                label = "tab:batch-size-benchmark-baseline-reference"
            print(
                format_combined_baseline_latex_table(
                    frames_by_dataset=frames_by_dataset,
                    decimals=args.decimals,
                    caption=caption,
                    label=label,
                )
            )
            return

        frame = build_baseline_table_frame(experiment_dir=experiment_dir)
        caption = args.caption
        if caption is None:
            caption = _default_baseline_caption(dataset_name=dataset_name)
        label = args.label
        if label is None:
            label = f"tab:batch-size-{_label_slug(dataset_name)}-baseline-reference"
        print(
            format_baseline_latex_table(
                frame=frame,
                decimals=args.decimals,
                caption=caption,
                label=label,
            )
        )
        return

    frame = build_table_frame(experiment_dir=experiment_dir, metric_name=metric_name)
    caption = args.caption
    if caption is None:
        caption = _default_caption(metric_name=metric_name, dataset_name=dataset_name)
    label = args.label
    if label is None:
        metric_slug = "strict-emr" if metric_name == "emr" else metric_name.replace("_", "-")
        label = f"tab:batch-size-{_label_slug(dataset_name)}-{metric_slug}"

    print(
        format_latex_table(
            frame=frame,
            decimals=args.decimals,
            caption=caption,
            label=label,
        )
    )


if __name__ == "__main__":
    main()
