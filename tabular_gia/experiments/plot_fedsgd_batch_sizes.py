from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_name_mappings import (
    DATASET_CLEAN_NAMES,
    MODEL_CLEAN_NAMES,
    PROTOCOL_CLEAN_NAMES,
    X_AXIS_CLEAN_NAMES,
    clean_name,
)

def _resolve_experiment_dir(results_root: Path, experiment_dir: str) -> Path:
    path = Path(experiment_dir)
    if not path.is_absolute():
        path = results_root / path
    if not path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {path}")
    return path


def _load_run_tableak_curve(run_dir: Path) -> pd.DataFrame | None:
    run_config_path = run_dir / "run_config.json"
    if not run_config_path.exists():
        return None
    with open(run_config_path, "r", encoding="utf-8") as f:
        run_cfg = json.load(f)
    batch_size = int(run_cfg["dataset"]["batch_size"])
    dataset_name = Path(str(run_cfg["dataset"]["dataset_path"])).stem
    model_cfg = run_cfg.get("model", {})
    model_name = str(model_cfg.get("preset") or model_cfg.get("arch") or "unknown")

    stats_csv = run_dir / "aggregated" / "rounds_summary_stats.csv"
    summary_csv = run_dir / "aggregated" / "rounds_summary.csv"

    if stats_csv.exists():
        df = pd.read_csv(stats_csv)
        required = {"exp_min_mean", "tableak_acc_mean"}
        if not required.issubset(df.columns):
            return None
        out = pd.DataFrame(
            {
                "exp_min": pd.to_numeric(df["exp_min_mean"], errors="coerce"),
                "tableak_acc": pd.to_numeric(df["tableak_acc_mean"], errors="coerce"),
                "tableak_ci95": pd.to_numeric(df.get("tableak_acc_ci95"), errors="coerce"),
            }
        )
    elif summary_csv.exists():
        df = pd.read_csv(summary_csv)
        required = {"exp_min", "tableak_acc"}
        if not required.issubset(df.columns):
            return None
        out = pd.DataFrame(
            {
                "exp_min": pd.to_numeric(df["exp_min"], errors="coerce"),
                "tableak_acc": pd.to_numeric(df["tableak_acc"], errors="coerce"),
                "tableak_ci95": np.nan,
            }
        )
    else:
        return None

    out = out.dropna(subset=["exp_min", "tableak_acc"]).sort_values("exp_min").reset_index(drop=True)
    out["dataset_name"] = dataset_name
    out["batch_size"] = batch_size
    out["model_name"] = model_name
    out["run_id"] = run_dir.name
    return out


def _collect_curves(experiment_dir: Path, protocol_subdir: str) -> pd.DataFrame:
    protocol_dir = experiment_dir / protocol_subdir
    if not protocol_dir.exists():
        raise FileNotFoundError(f"Protocol directory not found: {protocol_dir}")

    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for run_dir in sorted(protocol_dir.glob("run_*")):
        frame = _load_run_tableak_curve(run_dir)
        if frame is None or frame.empty:
            skipped.append(run_dir.name)
            continue
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No usable run curves found under {protocol_dir}")

    if skipped:
        print(f"Skipped {len(skipped)} run(s) with missing/invalid aggregated rounds data: {', '.join(skipped)}")

    combined = pd.concat(frames, axis=0, ignore_index=True)
    return combined


def _aggregate_for_plot(curves: pd.DataFrame) -> pd.DataFrame:
    def _ci95_from_runs(values: pd.Series) -> float:
        n = int(values.notna().sum())
        if n <= 1:
            return float("nan")
        std = float(values.std(ddof=1))
        return float(1.96 * (std / math.sqrt(n)))

    grouped = (
        curves.groupby(["dataset_name", "model_name", "batch_size", "exp_min"], as_index=False)
        .agg(
            tableak_acc_mean=("tableak_acc", "mean"),
            tableak_acc_std=("tableak_acc", "std"),
            n_runs=("tableak_acc", "count"),
            tableak_acc_ci95_runs=("tableak_acc", _ci95_from_runs),
            tableak_acc_ci95_mean=("tableak_ci95", "mean"),
        )
        .sort_values(["batch_size", "exp_min"])
        .reset_index(drop=True)
    )

    # Prefer empirical CI across repeated runs; fallback to mean of per-run CI if available.
    grouped["tableak_acc_ci95"] = grouped["tableak_acc_ci95_runs"].where(
        grouped["tableak_acc_ci95_runs"].notna(),
        grouped["tableak_acc_ci95_mean"],
    )
    grouped["tableak_acc_ci95"] = grouped["tableak_acc_ci95"].fillna(0.0)
    return grouped


def _set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def _plot_tableak_multiline(agg: pd.DataFrame, title: str) -> plt.Figure:
    _set_paper_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    batch_sizes = sorted(agg["batch_size"].astype(int).unique().tolist())
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.1, 0.95, len(batch_sizes)))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    for idx, batch_size in enumerate(batch_sizes):
        subset = agg[agg["batch_size"] == batch_size].sort_values("exp_min")
        x = subset["exp_min"].to_numpy(dtype=float)
        y = subset["tableak_acc_mean"].to_numpy(dtype=float)
        ci = subset["tableak_acc_ci95"].to_numpy(dtype=float)
        n_runs = int(subset["n_runs"].max())

        color = colors[idx]
        marker = markers[idx % len(markers)]
        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            markersize=4.8,
            linewidth=2.0,
            label=f"Batch {batch_size} (n={n_runs})",
        )
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.16, linewidth=0)

    ax.set_xlabel(clean_name("exp_min", X_AXIS_CLEAN_NAMES))
    ax.set_ylabel("TabLeak Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="major")
    ax.set_title(title)
    legend = ax.legend(title="Client Batch Size", ncol=2, frameon=True)
    legend.get_frame().set_alpha(0.92)
    return fig


def _sanitize_for_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return safe or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot FedSGD batch-size experiment: TabLeak accuracy vs minimum exposure."
    )
    parser.add_argument(
        "experiment_dir",
        help='Experiment directory name or path, e.g. "experiment_fedsgdbatchsizes_20260324_230729_514418".',
    )
    parser.add_argument("--results-root", default="results/experiments")
    parser.add_argument("--protocol-subdir", default="fedsgd")
    parser.add_argument(
        "--out",
        default="",
        help="Optional output PNG path. If omitted, writes PNG/PDF under <experiment_dir>/plots/.",
    )
    args = parser.parse_args()

    results_root = Path(args.results_root)
    experiment_dir = _resolve_experiment_dir(results_root, args.experiment_dir)
    curves = _collect_curves(experiment_dir, args.protocol_subdir)
    agg = _aggregate_for_plot(curves)

    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = sorted(agg["dataset_name"].astype(str).unique().tolist())
    for dataset_name in dataset_names:
        dataset_agg = agg[agg["dataset_name"] == dataset_name].copy()
        model_names = sorted(dataset_agg["model_name"].astype(str).unique().tolist())
        for model_name in model_names:
            model_agg = dataset_agg[dataset_agg["model_name"] == model_name].copy()
            model_suffix = _sanitize_for_filename(model_name)
            dataset_suffix = _sanitize_for_filename(dataset_name)
            protocol_clean_name = clean_name(args.protocol_subdir, PROTOCOL_CLEAN_NAMES)
            model_clean_name = clean_name(model_name, MODEL_CLEAN_NAMES)
            dataset_clean_name = clean_name(dataset_name, DATASET_CLEAN_NAMES)
            title = f"{protocol_clean_name}: TabLeak Accuracy Across Batch Sizes ({model_clean_name}, {dataset_clean_name})"
            fig = _plot_tableak_multiline(model_agg, title)

            if args.out and len(dataset_names) == 1 and len(model_names) == 1:
                out_png = Path(args.out)
                out_png.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_png = plots_dir / (
                    f"fedsgd_batch_sizes_tableak_vs_min_exposure_{model_suffix}__dataset_{dataset_suffix}.png"
                )
            data_out = plots_dir / (
                f"fedsgd_batch_sizes_tableak_vs_min_exposure_data_{model_suffix}__dataset_{dataset_suffix}.csv"
            )

            fig.savefig(out_png, dpi=300)
            plt.close(fig)
            model_agg.to_csv(data_out, index=False)
            print(out_png)
            print(data_out)


if __name__ == "__main__":
    main()
