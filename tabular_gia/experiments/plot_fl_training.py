from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_name_mappings import (
    DATASET_CLEAN_NAMES,
    DEFAULT_Y_BY_TASK,
    METRIC_CLEAN_NAMES,
    MODEL_CLEAN_NAMES,
    PROTOCOL_CLEAN_NAMES,
    TASK_CLEAN_NAMES,
    X_AXIS_CLEAN_NAMES,
    clean_name,
    infer_task_objective,
    metric_for_final_test,
)


def _resolve_experiment_dir(results_root: Path, experiment_dir: str) -> Path:
    path = Path(experiment_dir)
    if not path.is_absolute():
        path = results_root / path
    if not path.exists():
        raise FileNotFoundError(f"Experiment directory not found: {path}")
    return path


def _get_nested(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    parts = path.split(".")
    cur: Any = cfg
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _extract_group_value(cfg: dict[str, Any]) -> str:
    value = _get_nested(cfg, "dataset.batch_size")
    return str(value)


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
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def _collect_fl_rows(
    experiment_dir: Path,
    protocol_filter: str,
    phase: str,
    x_col: str,
    y_col: str,
) -> pd.DataFrame:
    p = experiment_dir / protocol_filter
    if not p.exists():
        raise FileNotFoundError(f"Protocol directory not found: {p}")
    protocol_dirs: list[Path] = [p]

    rows: list[pd.DataFrame] = []
    skipped: list[str] = []
    for protocol_dir in protocol_dirs:
        protocol = protocol_dir.name
        for run_dir in sorted(protocol_dir.glob("run_*")):
            cfg_path = run_dir / "run_config.json"
            fl_csv = run_dir / "aggregated" / "fl.csv"
            if not cfg_path.exists() or not fl_csv.exists():
                skipped.append(run_dir.as_posix())
                continue
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            df = pd.read_csv(fl_csv)
            if phase != "all":
                df = df[df["phase"] == phase]
            if x_col not in df.columns or (y_col != "auto" and y_col not in df.columns):
                skipped.append(run_dir.as_posix())
                continue

            group_value = _extract_group_value(cfg)
            model_name = str(_get_nested(cfg, "model.preset", None) or _get_nested(cfg, "model.arch", "unknown"))
            dataset_path = str(_get_nested(cfg, "dataset.dataset_path", "unknown"))
            dataset_name = Path(dataset_path).stem
            try:
                task_objective = infer_task_objective(dataset_name=dataset_name, dataset_path=dataset_path)
            except ValueError:
                skipped.append(run_dir.as_posix())
                continue
            selected_y_col = y_col if y_col != "auto" else DEFAULT_Y_BY_TASK.get(task_objective, "val_loss")

            phase_metric_pairs: list[tuple[str, str]]
            if phase == "all":
                phase_metric_pairs = [
                    ("checkpoint", selected_y_col),
                    ("final_test", metric_for_final_test(selected_y_col)),
                ]
            elif phase == "final_test":
                phase_metric_pairs = [("final_test", metric_for_final_test(selected_y_col))]
            else:
                phase_metric_pairs = [(phase, selected_y_col)]

            run_had_rows = False
            for selected_phase, selected_metric in phase_metric_pairs:
                if selected_metric not in df.columns:
                    continue
                phase_df = df[df["phase"] == selected_phase]
                part = pd.DataFrame(
                    {
                        "protocol": protocol,
                        "run_id": run_dir.name,
                        "phase": selected_phase,
                        "dataset_name": dataset_name,
                        "task_objective": task_objective,
                        "model_name": model_name,
                        "y_metric": selected_metric,
                        "group": group_value,
                        "x": pd.to_numeric(phase_df[x_col], errors="coerce"),
                        "y": pd.to_numeric(phase_df[selected_metric], errors="coerce"),
                    }
                ).dropna(subset=["x", "y"])
                if part.empty:
                    continue
                run_had_rows = True
                rows.append(part)

            if not run_had_rows:
                skipped.append(run_dir.as_posix())
                continue

    if not rows:
        raise FileNotFoundError(
            f"No usable FL rows found for x='{x_col}', y='{y_col}', phase='{phase}' in {experiment_dir}"
        )
    if skipped:
        print(f"Skipped {len(skipped)} run(s) due to missing files/columns.")
    return pd.concat(rows, axis=0, ignore_index=True)


def _aggregate(rows: pd.DataFrame) -> pd.DataFrame:
    def _ci95(values: pd.Series) -> float:
        n = int(values.notna().sum())
        if n <= 1:
            return float("nan")
        std = float(values.std(ddof=1))
        return float(1.96 * (std / math.sqrt(n)))

    out = (
        rows.groupby(["dataset_name", "task_objective", "model_name", "y_metric", "group", "x"], as_index=False)
        .agg(
            phase=("phase", "first"),
            y_mean=("y", "mean"),
            y_std=("y", "std"),
            n_runs=("y", "count"),
            y_ci95=("y", _ci95),
        )
        .sort_values(["group", "x"])
        .reset_index(drop=True)
    )
    out["y_ci95"] = out["y_ci95"].fillna(0.0)
    return out


def _plot(agg: pd.DataFrame, x_label: str, y_label: str, title: str) -> plt.Figure:
    _set_paper_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))

    groups = sorted(agg["group"].astype(str).unique().tolist())
    cmap = plt.get_cmap("tab10")
    colors = cmap(np.linspace(0.0, 1.0, max(1, len(groups))))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    for idx, group in enumerate(groups):
        sub = agg[agg["group"] == group].sort_values("x")
        x = sub["x"].to_numpy(dtype=float)
        y = sub["y_mean"].to_numpy(dtype=float)
        ci = sub["y_ci95"].to_numpy(dtype=float)
        n = int(sub["n_runs"].max())

        ax.plot(
            x,
            y,
            color=colors[idx],
            marker=markers[idx % len(markers)],
            linewidth=2.0,
            markersize=4.8,
            label=f"{group} (n={n})",
        )
        ax.fill_between(x, y - ci, y + ci, color=colors[idx], alpha=0.16, linewidth=0)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, which="major")
    ax.set_title(title)
    legend = ax.legend(title="Client Batch Size", ncol=2, frameon=True)
    legend.get_frame().set_alpha(0.92)
    return fig


def _slug(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return safe or "plot"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Protocol-agnostic FL training plotter (FedSGD/FedAvg)."
    )
    parser.add_argument("experiment_dir", help='Experiment directory name/path, e.g. "experiment_...".')
    parser.add_argument("--results-root", default="results/experiments")
    parser.add_argument("--protocol", choices=["fedsgd", "fedavg"], default="fedsgd")
    parser.add_argument("--phase", choices=["all", "checkpoint", "final_test"], default="all")
    parser.add_argument("--x", default="exp_min", help='X column from fl.csv, e.g. "exp_min" or "round".')
    parser.add_argument(
        "--y",
        default="auto",
        help='Y column from fl.csv. Use "auto" for task-aware mapping: binary->val_roc_auc, multiclass->val_f1_macro, regression->val_r2.',
    )
    parser.add_argument("--title", default="", help="Optional custom plot title.")
    parser.add_argument("--out", default="", help="Optional output PNG path.")
    args = parser.parse_args()

    experiment_dir = _resolve_experiment_dir(Path(args.results_root), args.experiment_dir)
    rows = _collect_fl_rows(
        experiment_dir=experiment_dir,
        protocol_filter=args.protocol,
        phase=args.phase,
        x_col=args.x,
        y_col=args.y,
    )
    agg = _aggregate(rows)

    plots_dir = experiment_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    dataset_names = sorted(agg["dataset_name"].astype(str).unique().tolist())
    for dataset_name in dataset_names:
        dataset_agg = agg[agg["dataset_name"] == dataset_name].copy()
        model_names = sorted(dataset_agg["model_name"].astype(str).unique().tolist())
        for model_name in model_names:
            model_agg = dataset_agg[dataset_agg["model_name"] == model_name].copy()
            y_metrics = sorted(model_agg["y_metric"].astype(str).unique().tolist())
            for y_metric in y_metrics:
                metric_agg = model_agg[model_agg["y_metric"] == y_metric].copy()
                phase_names = sorted(metric_agg["phase"].astype(str).unique().tolist())
                for phase_name in phase_names:
                    phase_metric_agg = metric_agg[metric_agg["phase"] == phase_name].copy()
                    task_objective = str(phase_metric_agg["task_objective"].iloc[0])
                    protocol_clean_name = clean_name(str(args.protocol), PROTOCOL_CLEAN_NAMES)
                    metric_clean_name = clean_name(y_metric, METRIC_CLEAN_NAMES)
                    model_clean_name = clean_name(model_name, MODEL_CLEAN_NAMES)
                    dataset_clean_name = clean_name(dataset_name, DATASET_CLEAN_NAMES)
                    task_clean_name = clean_name(task_objective, TASK_CLEAN_NAMES)
                    title = (
                        args.title
                        if args.title
                        else f"{protocol_clean_name}: {metric_clean_name} during FL training ({model_clean_name}, {dataset_clean_name}, {task_clean_name})"
                    )
                    x_clean_name = clean_name(args.x, X_AXIS_CLEAN_NAMES)
                    fig = _plot(phase_metric_agg, x_label=x_clean_name, y_label=metric_clean_name, title=title)

                    model_suffix = _slug(model_name)
                    dataset_suffix = _slug(dataset_name)
                    if (
                        args.out
                        and len(dataset_names) == 1
                        and len(model_names) == 1
                        and len(y_metrics) == 1
                        and len(phase_names) == 1
                    ):
                        out_png = Path(args.out)
                        out_png.parent.mkdir(parents=True, exist_ok=True)
                    else:
                        out_png = plots_dir / (
                            f"fl_training_{_slug(y_metric)}_vs_{_slug(args.x)}"
                            f"__phase_{_slug(phase_name)}__protocol_{_slug(args.protocol)}"
                            f"__model_{model_suffix}__dataset_{dataset_suffix}.png"
                        )
                    out_csv = out_png.with_suffix(".csv")
                    fig.savefig(out_png, dpi=300)
                    plt.close(fig)
                    phase_metric_agg.to_csv(out_csv, index=False)
                    print(out_png)
                    print(out_csv)


if __name__ == "__main__":
    main()
