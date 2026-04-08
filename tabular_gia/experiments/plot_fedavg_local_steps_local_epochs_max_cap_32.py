import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_fedsgd_batch_sizes as batch_plots
from plot_helper import (
    DATASET_CLEAN_NAMES,
    FL_METRIC_CLEAN_NAMES,
    MODEL_CLEAN_NAMES,
    PROTOCOL_CLEAN_NAMES,
    TASK_CLEAN_NAMES,
    clean_name,
    fl_metric_limits,
    infer_task_objective,
    set_plot_paper_style,
)

DEFAULT_EXPERIMENT_DIR = "experiment_fedavglocalstepslocalepochsmaxcap32_20260328_211218_446393_adult"
EFFECTIVE_STEPS_LABEL = "Effective Local Steps per Round"


def _get_nested(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _format_combo_label(steps_per_local_epoch: int, local_epochs: int) -> str:
    epoch_label = "epoch" if local_epochs == 1 else "epochs"
    step_label = "step" if steps_per_local_epoch == 1 else "steps"
    return f"{steps_per_local_epoch} {step_label}/epoch, {local_epochs} {epoch_label}"


def _load_run_metadata(run_dir: Path) -> dict[str, Any] | None:
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    dataset_path = str(_get_nested(cfg, "dataset.dataset_path", "unknown"))
    dataset_name = Path(dataset_path).stem
    task_objective = infer_task_objective(dataset_name=dataset_name, dataset_path=dataset_path)
    batch_size = int(_get_nested(cfg, "dataset.batch_size"))
    local_epochs = int(_get_nested(cfg, "fl.local_epochs"))
    max_client_dataset_examples = int(_get_nested(cfg, "fl.max_client_dataset_examples"))
    steps_per_local_epoch = int(math.ceil(max_client_dataset_examples / batch_size))
    effective_local_steps = int(local_epochs * steps_per_local_epoch)
    model_name = str(_get_nested(cfg, "model.preset") or _get_nested(cfg, "model.arch") or "unknown")
    return {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "task_objective": task_objective,
        "utility_metric": batch_plots.DEFAULT_Y_BY_TASK.get(task_objective, "val_loss"),
        "run_id": run_dir.name,
        "model_name": model_name,
        "batch_size": batch_size,
        "local_epochs": local_epochs,
        "max_client_dataset_examples": max_client_dataset_examples,
        "steps_per_local_epoch": steps_per_local_epoch,
        "effective_local_steps": effective_local_steps,
        "combo_label": _format_combo_label(steps_per_local_epoch, local_epochs),
    }


def _base_frame(metadata: dict[str, Any], size: int) -> pd.DataFrame:
    return pd.DataFrame({key: [value] * size for key, value in metadata.items()})


def _align_attack_and_utility_with_ci(
    attack_curve: pd.DataFrame,
    utility_curve: pd.DataFrame,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    targets = attack_curve["exp_min"].to_numpy(dtype=float)
    utility_interp = batch_plots._interpolate_series(utility_curve, "exp_min", "utility_value", targets)
    utility_ci_interp = batch_plots._interpolate_series(utility_curve.fillna({"utility_ci95": 0.0}), "exp_min", "utility_ci95", targets)
    out = attack_curve[["exp_min", "tableak_acc", "tableak_acc_ci95"]].copy()
    out["utility_value"] = utility_interp
    out["utility_ci95"] = utility_ci_interp
    out = out.dropna(subset=["utility_value"]).reset_index(drop=True)
    if out.empty:
        return out
    return pd.concat([_base_frame(metadata, len(out)), out], axis=1)


def _collect_plot_tables(experiment_dir: Path, protocol_subdir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol_dir = experiment_dir / protocol_subdir
    if not protocol_dir.exists():
        raise FileNotFoundError(f"Protocol directory not found: {protocol_dir}")

    attack_frames: list[pd.DataFrame] = []
    endpoint_rows: list[pd.DataFrame] = []
    skipped: list[str] = []

    for run_dir in sorted(protocol_dir.glob("run_*")):
        metadata = _load_run_metadata(run_dir)
        if metadata is None:
            skipped.append(f"{run_dir.name}:missing_run_config")
            continue

        attack_curve = batch_plots._load_attack_curve(run_dir, metadata)
        utility_curve = batch_plots._load_utility_curve(run_dir, metadata)
        if attack_curve is None or utility_curve is None:
            skipped.append(f"{run_dir.name}:missing_attack_or_utility")
            continue
        attack_frames.append(attack_curve)

        aligned = _align_attack_and_utility_with_ci(attack_curve, utility_curve, metadata)
        if aligned.empty:
            skipped.append(f"{run_dir.name}:empty_aligned_endpoint")
            continue
        endpoint_rows.append(aligned.sort_values("exp_min").tail(1))

    if skipped:
        print(f"Skipped {len(skipped)} run(s): {', '.join(skipped)}")
    if not attack_frames or not endpoint_rows:
        raise FileNotFoundError(f"No usable fedavg curves found under {protocol_dir}")
    return (
        pd.concat(attack_frames, axis=0, ignore_index=True),
        pd.concat(endpoint_rows, axis=0, ignore_index=True),
    )


def _ordered_combos(rows: pd.DataFrame) -> list[tuple[int, int, int]]:
    combo_frame = rows[["effective_local_steps", "batch_size", "local_epochs"]].drop_duplicates().copy()
    combo_frame = combo_frame.sort_values(["effective_local_steps", "local_epochs", "batch_size"]).reset_index(drop=True)
    return [
        (int(row.effective_local_steps), int(row.batch_size), int(row.local_epochs))
        for row in combo_frame.itertuples(index=False)
    ]


def _combo_styles(rows: pd.DataFrame) -> dict[tuple[int, int, int], tuple[Any, str, str]]:
    combos = _ordered_combos(rows)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.95, len(combos)))
    batch_sizes = sorted(rows["batch_size"].astype(int).unique().tolist())
    markers = {batch_size: batch_plots.MARKERS[idx % len(batch_plots.MARKERS)] for idx, batch_size in enumerate(batch_sizes)}
    labels = {}
    for combo in combos:
        steps, batch_size, local_epochs = combo
        steps_per_local_epoch = max(1, steps // max(1, local_epochs))
        labels[combo] = _format_combo_label(steps_per_local_epoch, local_epochs)
    return {combo: (colors[idx], markers[combo[1]], labels[combo]) for idx, combo in enumerate(combos)}


def _plot_final_metric_vs_effective_steps(
    endpoint_rows: pd.DataFrame,
    *,
    value_col: str,
    ci_col: str,
    ylabel: str,
    ylim: tuple[float, float] | None,
    title: str,
) -> plt.Figure:
    set_plot_paper_style()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    styles = _combo_styles(endpoint_rows)

    for combo in _ordered_combos(endpoint_rows):
        steps, batch_size, local_epochs = combo
        sub = endpoint_rows[
            (endpoint_rows["effective_local_steps"] == steps)
            & (endpoint_rows["batch_size"] == batch_size)
            & (endpoint_rows["local_epochs"] == local_epochs)
        ].copy()
        if sub.empty:
            continue
        row = sub.iloc[0]
        color, marker, label = styles[combo]
        y = np.array([float(row[value_col])], dtype=float)
        ci = np.array([float(row[ci_col]) if pd.notna(row[ci_col]) else 0.0], dtype=float)
        ax.errorbar(
            [steps],
            y,
            yerr=batch_plots._errorbar_yerr(y, ci, bounded=ylim == (0.0, batch_plots.BOUNDED_METRIC_YMAX)),
            fmt=marker,
            color=color,
            markersize=7.0,
            capsize=3.0,
            label=label,
        )

    ax.set_xticks(sorted(endpoint_rows["effective_local_steps"].astype(int).unique().tolist()))
    batch_plots._apply_axis_finish(
        ax,
        title=title,
        xlabel=EFFECTIVE_STEPS_LABEL,
        ylabel=ylabel,
        ylim=ylim,
    )
    legend = ax.legend(title="Configuration", loc="best", frameon=True)
    legend.get_frame().set_alpha(0.92)
    fig.tight_layout()
    return fig


def _plot_attack_vs_exposure(attack_rows: pd.DataFrame, *, title: str) -> plt.Figure:
    set_plot_paper_style()
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    styles = _combo_styles(attack_rows)

    for combo in _ordered_combos(attack_rows):
        steps, batch_size, local_epochs = combo
        sub = attack_rows[
            (attack_rows["effective_local_steps"] == steps)
            & (attack_rows["batch_size"] == batch_size)
            & (attack_rows["local_epochs"] == local_epochs)
        ].sort_values("exp_min")
        if sub.empty:
            continue
        color, marker, label = styles[combo]
        batch_plots._plot_line_with_band(
            ax,
            x=sub["exp_min"].to_numpy(dtype=float),
            y=sub["tableak_acc"].to_numpy(dtype=float),
            ci=sub["tableak_acc_ci95"].fillna(0.0).to_numpy(dtype=float),
            color=color,
            marker=marker,
            linewidth=1.9,
            markersize=4.6,
            alpha=0.12,
            label=label,
            bounded=True,
        )

    batch_plots._apply_axis_finish(
        ax,
        title=title,
        xlabel=clean_name("exp_min", batch_plots.X_AXIS_CLEAN_NAMES),
        ylabel=batch_plots.ATTACK_METRIC_LABEL,
        ylim=(0.0, batch_plots.BOUNDED_METRIC_YMAX),
    )
    legend = ax.legend(title="Configuration", loc="best", frameon=True)
    legend.get_frame().set_alpha(0.92)
    fig.tight_layout()
    return fig


def _plot_cross_model_attack_vs_exposure(
    dataset_attack_rows: pd.DataFrame,
    *,
    protocol_clean_name: str,
    dataset_clean_name: str,
) -> plt.Figure:
    set_plot_paper_style()
    model_names = batch_plots._ordered_model_names(dataset_attack_rows["model_name"].astype(str).unique().tolist())
    fig, axes = plt.subplots(1, len(model_names), figsize=(4.5 * len(model_names), 5.2), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes)
    styles = _combo_styles(dataset_attack_rows)

    for ax, model_name in zip(axes_array, model_names, strict=False):
        model_rows = dataset_attack_rows[dataset_attack_rows["model_name"] == model_name].copy()
        for combo in _ordered_combos(dataset_attack_rows):
            steps, batch_size, local_epochs = combo
            sub = model_rows[
                (model_rows["effective_local_steps"] == steps)
                & (model_rows["batch_size"] == batch_size)
                & (model_rows["local_epochs"] == local_epochs)
            ].sort_values("exp_min")
            if sub.empty:
                continue
            color, marker, label = styles[combo]
            batch_plots._plot_line_with_band(
                ax,
                x=sub["exp_min"].to_numpy(dtype=float),
                y=sub["tableak_acc"].to_numpy(dtype=float),
                ci=sub["tableak_acc_ci95"].fillna(0.0).to_numpy(dtype=float),
                color=color,
                marker=marker,
                linewidth=1.8,
                markersize=4.2,
                alpha=0.12,
                label=label,
                bounded=True,
            )
        batch_plots._apply_axis_finish(
            ax,
            title=clean_name(model_name, MODEL_CLEAN_NAMES),
            xlabel=clean_name("exp_min", batch_plots.X_AXIS_CLEAN_NAMES),
            ylabel=batch_plots.ATTACK_METRIC_LABEL if ax is axes_array[0] else "",
            ylim=(0.0, batch_plots.BOUNDED_METRIC_YMAX),
        )

    handles, labels = axes_array[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        title="Configuration",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=True,
    )
    legend.get_frame().set_alpha(0.92)
    fig.suptitle(
        f"{protocol_clean_name}: {batch_plots.ATTACK_METRIC_LABEL} Across Models ({dataset_clean_name})",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.80))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot FedAvg local-steps/local-epochs max-cap=32 results using effective local steps per round."
    )
    parser.add_argument(
        "experiment_dir",
        nargs="?",
        default=DEFAULT_EXPERIMENT_DIR,
        help=f'Experiment directory name or path, e.g. "{DEFAULT_EXPERIMENT_DIR}".',
    )
    parser.add_argument("--results-root", default="")
    parser.add_argument("--protocol-subdir", default="fedavg")
    args = parser.parse_args()

    experiment_dir = batch_plots._resolve_experiment_dir(args.results_root, args.experiment_dir)
    attack_rows, endpoint_rows = _collect_plot_tables(experiment_dir=experiment_dir, protocol_subdir=args.protocol_subdir)

    plots_dir = experiment_dir / "plots"
    image_singles_dir = plots_dir / "image" / "singles"
    image_panels_dir = plots_dir / "image" / "panels"
    pdf_singles_dir = plots_dir / "pdf" / "singles"
    pdf_panels_dir = plots_dir / "pdf" / "panels"
    data_dir = plots_dir / "data"
    image_singles_dir.mkdir(parents=True, exist_ok=True)
    image_panels_dir.mkdir(parents=True, exist_ok=True)
    pdf_singles_dir.mkdir(parents=True, exist_ok=True)
    pdf_panels_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in sorted(endpoint_rows["dataset_name"].astype(str).unique().tolist()):
        dataset_endpoint_rows = endpoint_rows[endpoint_rows["dataset_name"] == dataset_name].copy()
        dataset_attack_rows = attack_rows[attack_rows["dataset_name"] == dataset_name].copy()
        if dataset_endpoint_rows.empty or dataset_attack_rows.empty:
            continue

        dataset_clean_name = clean_name(dataset_name, DATASET_CLEAN_NAMES)
        task_clean_name = clean_name(str(dataset_endpoint_rows["task_objective"].iloc[0]), TASK_CLEAN_NAMES)
        protocol_clean_name = clean_name(args.protocol_subdir, PROTOCOL_CLEAN_NAMES)
        dataset_suffix = batch_plots._sanitize_for_filename(dataset_name)

        cross_model_name = f"fedavg_cross_model_attack_vs_exposure__dataset_{dataset_suffix}"
        cross_model_fig = _plot_cross_model_attack_vs_exposure(
            dataset_attack_rows,
            protocol_clean_name=protocol_clean_name,
            dataset_clean_name=dataset_clean_name,
        )
        batch_plots._save_figure(cross_model_fig, image_panels_dir / cross_model_name)
        cross_model_src_pdf = image_panels_dir / f"{cross_model_name}.pdf"
        cross_model_dst_pdf = pdf_panels_dir / f"{cross_model_name}.pdf"
        if cross_model_src_pdf.exists():
            cross_model_src_pdf.replace(cross_model_dst_pdf)
        dataset_attack_rows.to_csv(data_dir / f"{cross_model_name}_data.csv", index=False)
        print(image_panels_dir / f"{cross_model_name}.png")
        print(pdf_panels_dir / f"{cross_model_name}.pdf")

        for model_name in batch_plots._ordered_model_names(dataset_endpoint_rows["model_name"].astype(str).unique().tolist()):
            model_endpoint_rows = dataset_endpoint_rows[dataset_endpoint_rows["model_name"] == model_name].copy()
            model_attack_rows = dataset_attack_rows[dataset_attack_rows["model_name"] == model_name].copy()
            if model_endpoint_rows.empty or model_attack_rows.empty:
                continue

            model_clean_name = clean_name(model_name, MODEL_CLEAN_NAMES)
            utility_metric = str(model_endpoint_rows["utility_metric"].iloc[0])
            utility_label = clean_name(utility_metric, FL_METRIC_CLEAN_NAMES)
            utility_ylim = fl_metric_limits(utility_metric, model_endpoint_rows["utility_value"])

            attack_steps_name = f"fedavg_effective_steps_attack__model_{batch_plots._sanitize_for_filename(model_name)}__dataset_{dataset_suffix}"
            attack_steps_fig = _plot_final_metric_vs_effective_steps(
                model_endpoint_rows,
                value_col="tableak_acc",
                ci_col="tableak_acc_ci95",
                ylabel=batch_plots.ATTACK_METRIC_LABEL,
                ylim=(0.0, batch_plots.BOUNDED_METRIC_YMAX),
                title=f"{protocol_clean_name}: Final {batch_plots.ATTACK_METRIC_LABEL} vs Effective Local Steps ({model_clean_name}, {dataset_clean_name})",
            )
            batch_plots._save_figure(attack_steps_fig, image_singles_dir / attack_steps_name)
            attack_steps_src_pdf = image_singles_dir / f"{attack_steps_name}.pdf"
            attack_steps_dst_pdf = pdf_singles_dir / f"{attack_steps_name}.pdf"
            if attack_steps_src_pdf.exists():
                attack_steps_src_pdf.replace(attack_steps_dst_pdf)

            utility_steps_name = f"fedavg_effective_steps_utility__model_{batch_plots._sanitize_for_filename(model_name)}__dataset_{dataset_suffix}"
            utility_steps_fig = _plot_final_metric_vs_effective_steps(
                model_endpoint_rows,
                value_col="utility_value",
                ci_col="utility_ci95",
                ylabel=utility_label,
                ylim=utility_ylim,
                title=f"{protocol_clean_name}: Final {utility_label} vs Effective Local Steps ({model_clean_name}, {dataset_clean_name})",
            )
            batch_plots._save_figure(utility_steps_fig, image_singles_dir / utility_steps_name)
            utility_steps_src_pdf = image_singles_dir / f"{utility_steps_name}.pdf"
            utility_steps_dst_pdf = pdf_singles_dir / f"{utility_steps_name}.pdf"
            if utility_steps_src_pdf.exists():
                utility_steps_src_pdf.replace(utility_steps_dst_pdf)

            exposure_name = f"fedavg_attack_vs_exposure__model_{batch_plots._sanitize_for_filename(model_name)}__dataset_{dataset_suffix}"
            exposure_fig = _plot_attack_vs_exposure(
                model_attack_rows,
                title=f"{protocol_clean_name}: {batch_plots.ATTACK_METRIC_LABEL} vs Minimum Exposure ({model_clean_name}, {dataset_clean_name})",
            )
            batch_plots._save_figure(exposure_fig, image_singles_dir / exposure_name)
            exposure_src_pdf = image_singles_dir / f"{exposure_name}.pdf"
            exposure_dst_pdf = pdf_singles_dir / f"{exposure_name}.pdf"
            if exposure_src_pdf.exists():
                exposure_src_pdf.replace(exposure_dst_pdf)

            model_endpoint_rows.to_csv(data_dir / f"{attack_steps_name}_data.csv", index=False)
            model_endpoint_rows.to_csv(data_dir / f"{utility_steps_name}_data.csv", index=False)
            model_attack_rows.to_csv(data_dir / f"{exposure_name}_data.csv", index=False)

            print(image_singles_dir / f"{attack_steps_name}.png")
            print(pdf_singles_dir / f"{attack_steps_name}.pdf")
            print(image_singles_dir / f"{utility_steps_name}.png")
            print(pdf_singles_dir / f"{utility_steps_name}.pdf")
            print(image_singles_dir / f"{exposure_name}.png")
            print(pdf_singles_dir / f"{exposure_name}.pdf")


if __name__ == "__main__":
    main()
