import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_fedsgd_batch_sizes as batch_plots
from plot_fl_training import plot_fl_metric_vs_exposure
from plot_helper import (
    DATASET_CLEAN_NAMES,
    FL_METRIC_CLEAN_NAMES,
    MODEL_CLEAN_NAMES,
    PROTOCOL_CLEAN_NAMES,
    TASK_CLEAN_NAMES,
    clean_name,
)


ATTACK_DROP_LABEL = "Reconstruction Accuracy Drop from Initialization"


def _attack_drop_from_initial(plot_rows: pd.DataFrame) -> pd.DataFrame:
    if plot_rows.empty:
        return plot_rows.copy()

    group_cols = ["dataset_name", "model_name", "batch_size", "run_id"]
    out = plot_rows.sort_values(group_cols + ["exp_min"]).copy()
    initial = out.groupby(group_cols)["tableak_acc_mean"].transform("first")
    out["attack_drop_from_init_mean"] = initial - out["tableak_acc_mean"]
    return out


def _attack_drop_limits(values: pd.Series) -> tuple[float, float]:
    lo = float(values.min())
    hi = float(values.max())
    pad = max(0.015, 0.08 * max(1e-6, hi - lo))
    lower = min(-0.01, lo - pad)
    upper = hi + pad
    return (lower, upper)


def _plot_attack_drop_from_initial(plot_rows: pd.DataFrame, title: str, ylim: tuple[float, float]) -> plt.Figure:
    batch_plots.set_plot_paper_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    styles = batch_plots._batch_styles(plot_rows["batch_size"].astype(int).unique().tolist())
    for batch_size in sorted(styles):
        color, marker = styles[batch_size]
        sub = plot_rows[plot_rows["batch_size"] == batch_size].sort_values("exp_min")
        ax.plot(
            sub["exp_min"].to_numpy(dtype=float),
            sub["attack_drop_from_init_mean"].to_numpy(dtype=float),
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=4.8,
            label=str(batch_size),
        )
    batch_plots._apply_axis_finish(
        ax,
        title=title,
        xlabel=clean_name("exp_min", batch_plots.X_AXIS_CLEAN_NAMES),
        ylabel=ATTACK_DROP_LABEL,
        ylim=ylim,
    )
    legend = ax.legend(title="Client Batch Size", ncol=2, frameon=True)
    legend.get_frame().set_alpha(0.92)
    ax.axhline(0.0, color="0.55", linewidth=1.0, linestyle="--", alpha=0.8)
    return fig


def _plot_cross_model_attack_drop_from_initial(
    dataset_drop: pd.DataFrame,
    *,
    protocol_clean_name: str,
    dataset_clean_name: str,
    task_clean_name: str,
    ylim: tuple[float, float],
) -> plt.Figure:
    batch_plots.set_plot_paper_style()
    model_names = batch_plots._ordered_model_names(dataset_drop["model_name"].astype(str).unique().tolist())
    fig, axes = plt.subplots(1, len(model_names), figsize=(4.5 * len(model_names), 4.6), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes)
    styles = batch_plots._batch_styles(dataset_drop["batch_size"].astype(int).unique().tolist())

    for ax, model_name in zip(axes_array, model_names, strict=False):
        model_rows = dataset_drop[dataset_drop["model_name"] == model_name].copy()
        for batch_size in sorted(styles):
            color, marker = styles[batch_size]
            sub = model_rows[model_rows["batch_size"] == batch_size].sort_values("exp_min")
            ax.plot(
                sub["exp_min"].to_numpy(dtype=float),
                sub["attack_drop_from_init_mean"].to_numpy(dtype=float),
                color=color,
                marker=marker,
                linewidth=1.8,
                markersize=4.2,
                label=str(batch_size),
            )
        ax.axhline(0.0, color="0.55", linewidth=1.0, linestyle="--", alpha=0.8)
        batch_plots._apply_axis_finish(
            ax,
            title=clean_name(model_name, MODEL_CLEAN_NAMES),
            xlabel=clean_name("exp_min", batch_plots.X_AXIS_CLEAN_NAMES),
            ylabel=ATTACK_DROP_LABEL if ax is axes_array[0] else "",
            ylim=ylim,
        )

    legend = axes_array[0].legend(title="Client Batch Size", ncol=2, frameon=True, loc="upper left")
    legend.get_frame().set_alpha(0.92)
    fig.suptitle(
        f"{protocol_clean_name}: Leakage Decay from Initialization Across Models ({dataset_clean_name}, {task_clean_name})",
        y=1.02,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the full figure package for a FedSGD fixed-batch exposure-scheduled experiment."
    )
    parser.add_argument(
        "experiment_dir",
        nargs="?",
        default="experiment_fedsgdfixedbatch_20260331_113807_130703_adult",
        help='Experiment directory name or path, e.g. "experiment_fedsgdfixedbatch_20260331_113807_130703_adult".',
    )
    parser.add_argument("--results-root", default="")
    parser.add_argument("--protocol-subdir", default="fedsgd")
    args = parser.parse_args()

    experiment_dir = batch_plots._resolve_experiment_dir(args.results_root, args.experiment_dir)
    attack_rows, utility_rows, _, _ = batch_plots._collect_run_tables(
        experiment_dir=experiment_dir,
        protocol_subdir=args.protocol_subdir,
    )

    attack_plot = batch_plots._rename_plot_columns(
        attack_rows,
        {
            "tableak_acc": "tableak_acc_mean",
            "num_acc": "num_acc_mean",
            "cat_acc": "cat_acc_mean",
            "prior_tableak_acc": "prior_tableak_acc_mean",
            "prior_num_acc": "prior_num_acc_mean",
            "prior_cat_acc": "prior_cat_acc_mean",
            "random_tableak_acc": "random_tableak_acc_mean",
            "random_num_acc": "random_num_acc_mean",
            "random_cat_acc": "random_cat_acc_mean",
        },
    )
    attack_drop_plot = _attack_drop_from_initial(attack_plot)
    utility_plot = batch_plots._rename_plot_columns(utility_rows, {"utility_value": "utility_mean"})

    plots_dir = experiment_dir / "plots"
    image_dir = plots_dir / "image"
    pdf_dir = plots_dir / "pdf"
    data_dir = plots_dir / "data"
    image_singles_dir = image_dir / "singles"
    image_panels_dir = image_dir / "panels"
    pdf_singles_dir = pdf_dir / "singles"
    pdf_panels_dir = pdf_dir / "panels"
    image_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    image_singles_dir.mkdir(parents=True, exist_ok=True)
    image_panels_dir.mkdir(parents=True, exist_ok=True)
    pdf_singles_dir.mkdir(parents=True, exist_ok=True)
    pdf_panels_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = sorted(attack_plot["dataset_name"].astype(str).unique().tolist())
    for dataset_name in dataset_names:
        dataset_attack = attack_plot[attack_plot["dataset_name"] == dataset_name].copy()
        dataset_attack_drop = attack_drop_plot[attack_drop_plot["dataset_name"] == dataset_name].copy()
        dataset_utility = utility_plot[utility_plot["dataset_name"] == dataset_name].copy()
        if dataset_attack.empty:
            continue

        dataset_suffix = batch_plots._sanitize_for_filename(dataset_name)
        protocol_clean_name = f"{clean_name(args.protocol_subdir, PROTOCOL_CLEAN_NAMES)} Fixed-Batch"
        dataset_clean_name = clean_name(dataset_name, DATASET_CLEAN_NAMES)
        task_clean_name = clean_name(str(dataset_attack["task_objective"].iloc[0]), TASK_CLEAN_NAMES)

        cross_model_attack_base = plots_dir / (
            f"fedsgd_fixed_batch_sizes_cross_model_attack_vs_exposure__dataset_{dataset_suffix}"
        )
        cross_model_attack_drop_base = plots_dir / (
            f"fedsgd_fixed_batch_sizes_cross_model_attack_drop_from_init__dataset_{dataset_suffix}"
        )
        model_batch_attack_metrics_base = plots_dir / (
            f"fedsgd_fixed_batch_sizes_model_batch_attack_metrics__dataset_{dataset_suffix}"
        )
        attack_drop_ylim = _attack_drop_limits(dataset_attack_drop["attack_drop_from_init_mean"])

        batch_plots._save_figure(
            batch_plots._plot_cross_model_attack_vs_exposure(
                dataset_attack,
                protocol_clean_name=protocol_clean_name,
                dataset_clean_name=dataset_clean_name,
                task_clean_name=task_clean_name,
            ),
            image_panels_dir / cross_model_attack_base.name,
        )
        batch_plots._save_figure(
            _plot_cross_model_attack_drop_from_initial(
                dataset_attack_drop,
                protocol_clean_name=protocol_clean_name,
                dataset_clean_name=dataset_clean_name,
                task_clean_name=task_clean_name,
                ylim=attack_drop_ylim,
            ),
            image_panels_dir / cross_model_attack_drop_base.name,
        )
        batch_plots._save_figure(
            batch_plots._plot_model_batch_attack_metrics(
                dataset_attack,
                protocol_clean_name=protocol_clean_name,
                dataset_clean_name=dataset_clean_name,
                task_clean_name=task_clean_name,
            ),
            image_panels_dir / model_batch_attack_metrics_base.name,
        )

        for base_name in [
            cross_model_attack_base.name,
            cross_model_attack_drop_base.name,
            model_batch_attack_metrics_base.name,
        ]:
            src_png = image_panels_dir / f"{base_name}.png"
            src_pdf = image_panels_dir / f"{base_name}.pdf"
            dst_pdf = pdf_panels_dir / f"{base_name}.pdf"
            if src_pdf.exists():
                src_pdf.replace(dst_pdf)
            print(src_png)
            print(dst_pdf)

        dataset_attack.to_csv(data_dir / f"{cross_model_attack_base.name}_data.csv", index=False)
        dataset_attack_drop.to_csv(data_dir / f"{cross_model_attack_drop_base.name}_data.csv", index=False)
        dataset_attack.to_csv(data_dir / f"{model_batch_attack_metrics_base.name}_data.csv", index=False)

        model_names = sorted(dataset_attack["model_name"].astype(str).unique().tolist())
        for model_name in model_names:
            attack_slice = dataset_attack[dataset_attack["model_name"] == model_name].copy()
            attack_drop_slice = dataset_attack_drop[dataset_attack_drop["model_name"] == model_name].copy()
            utility_slice = dataset_utility[dataset_utility["model_name"] == model_name].copy()
            if utility_slice.empty:
                print(f"Skipping incomplete plot bundle for dataset={dataset_name} model={model_name}")
                continue

            model_suffix = batch_plots._sanitize_for_filename(model_name)
            model_clean_name = clean_name(model_name, MODEL_CLEAN_NAMES)
            utility_metric = str(utility_slice["utility_metric"].iloc[0])
            utility_label = clean_name(utility_metric, FL_METRIC_CLEAN_NAMES)

            attack_title = (
                f"{protocol_clean_name}: {batch_plots.ATTACK_METRIC_LABEL} Across Batch Sizes "
                f"({model_clean_name}, {dataset_clean_name}, {task_clean_name})"
            )
            attack_drop_title = (
                f"{protocol_clean_name}: Leakage Decay from Initialization "
                f"({model_clean_name}, {dataset_clean_name}, {task_clean_name})"
            )
            utility_title = (
                f"{protocol_clean_name}: {utility_label} Across Batch Sizes "
                f"({model_clean_name}, {dataset_clean_name}, {task_clean_name})"
            )

            attack_base = plots_dir / (
                f"fedsgd_fixed_batch_sizes_attack_vs_exposure__model_{model_suffix}__dataset_{dataset_suffix}"
            )
            attack_drop_base = plots_dir / (
                f"fedsgd_fixed_batch_sizes_attack_drop_from_init__model_{model_suffix}__dataset_{dataset_suffix}"
            )
            utility_base = plots_dir / (
                f"fedsgd_fixed_batch_sizes_utility_vs_exposure__model_{model_suffix}__dataset_{dataset_suffix}"
            )

            batch_plots._save_figure(
                batch_plots._plot_attack_vs_exposure(attack_slice, attack_title),
                image_singles_dir / attack_base.name,
            )
            batch_plots._save_figure(
                _plot_attack_drop_from_initial(attack_drop_slice, attack_drop_title, attack_drop_ylim),
                image_singles_dir / attack_drop_base.name,
            )
            batch_plots._save_figure(
                plot_fl_metric_vs_exposure(utility_slice, utility_title),
                image_singles_dir / utility_base.name,
            )

            output_base_names = [
                attack_base.name,
                attack_drop_base.name,
                utility_base.name,
            ]

            for base_name in output_base_names:
                src_root = image_singles_dir
                dst_root = pdf_singles_dir
                src_png = src_root / f"{base_name}.png"
                src_pdf = src_root / f"{base_name}.pdf"
                dst_pdf = dst_root / f"{base_name}.pdf"
                if src_pdf.exists():
                    src_pdf.replace(dst_pdf)
                print(src_png)
                print(dst_pdf)

            attack_slice.to_csv(data_dir / f"{attack_base.name}_data.csv", index=False)
            attack_drop_slice.to_csv(data_dir / f"{attack_drop_base.name}_data.csv", index=False)
            utility_slice.to_csv(data_dir / f"{utility_base.name}_data.csv", index=False)


if __name__ == "__main__":
    main()
