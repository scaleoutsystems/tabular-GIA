from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

MODEL_CLEAN_NAMES = {
    "fttransformer": "FTTransformer",
    "resnet": "ResNet",
    "small": "Small MLP",
}


METRIC_ALIASES = {
    "reconstruction": "tableak_acc",
    "recon": "tableak_acc",
    "tableak": "tableak_acc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a LaTeX batch-size table from an aggregated batch-size experiment directory."
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Path to an experiment directory containing sweep_results.csv and sweep_runs.json.",
    )
    parser.add_argument(
        "--metric",
        default="tableak_acc",
        help="Metric column from sweep_results.csv to print. Default: tableak_acc",
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


def _dataset_name_from_dir(experiment_dir: Path) -> str:
    name = experiment_dir.name
    for candidate in ("adult", "california", "pandemic", "mimic", "california_housing"):
        if candidate in name:
            return candidate
    return name


def _ordered_models(model_names: list[str]) -> list[str]:
    preferred = ["fttransformer", "resnet", "small"]
    ordered = [name for name in preferred if name in model_names]
    ordered.extend(sorted(name for name in model_names if name not in preferred))
    return ordered


def _clean_model_name(model_name: str) -> str:
    return MODEL_CLEAN_NAMES.get(model_name, model_name)


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


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    metric_name = _resolve_metric_name(args.metric)

    frame = build_table_frame(experiment_dir=experiment_dir, metric_name=metric_name)
    dataset_name = args.dataset_name or _dataset_name_from_dir(experiment_dir)
    caption = args.caption
    if caption is None:
        caption = (
            f"Reconstruction accuracy at the initialized and trained attack checkpoints for {dataset_name}. "
            f"Initialized denotes the attack before any global model aggregation, and Trained denotes the final attacked checkpoint. "
            f"Each cell reports mean $\\pm$ standard deviation across 3 seeds."
        )
    label = args.label
    if label is None:
        label = f"tab:batch-size-{dataset_name}-{metric_name.replace('_', '-')}"

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
