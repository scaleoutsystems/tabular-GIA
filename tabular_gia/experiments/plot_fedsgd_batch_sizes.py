import argparse
from pathlib import Path
from typing import Any

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_fl_training import (
    plot_fl_metric_vs_exposure,
    plot_fl_metric_vs_exposure_on_axes,
)
from plot_helper import (
    ATTACK_METRIC_LABEL,
    BASELINE_LINESTYLE,
    BASELINE_PRIOR_COLOR,
    BASELINE_RANDOM_COLOR,
    DATASET_CLEAN_NAMES,
    FL_METRIC_CLEAN_NAMES,
    MODEL_CLEAN_NAMES,
    PLOT_MARKER_EDGECOLOR,
    PLOT_MARKER_EDGEWIDTH,
    PLOT_MARKERS,
    PLOT_LINEWIDTH,
    PLOT_MARKER_ZORDER,
    PLOT_MARKERSIZE,
    PLOT_SCATTER_SIZE,
    PROTOCOL_CLEAN_NAMES,
    TASK_CLEAN_NAMES,
    X_AXIS_CLEAN_NAMES,
    align_attack_and_utility as _align_attack_and_utility,
    annotate_checkpoint_progress as _annotate_checkpoint_progress,
    apply_axis_finish as _apply_axis_finish,
    collect_run_tables as _collect_run_tables_base,
    errorbar_yerr as _errorbar_yerr,
    clean_name,
    fl_metric_limits,
    load_basic_run_metadata,
    metric_for_final_test,
    ordered_model_names as _ordered_model_names,
    plot_line_with_band as _plot_line_with_band,
    prepare_plot_dirs,
    rename_plot_columns as _rename_plot_columns,
    resolve_experiment_dir as _resolve_experiment_dir,
    sample_utility_at_attack_checkpoints as _sample_utility_at_attack_checkpoints,
    sanitize_for_filename as _sanitize_for_filename,
    save_figure as _save_figure,
    batch_styles as shared_batch_styles,
    sample_checkpoint_colors,
    sample_group_colors,
    set_plot_paper_style,
    style_legend,
)
BOUNDED_METRIC_YMAX = 1.0

def _load_run_metadata(run_dir: Path) -> dict[str, object] | None:
    return load_basic_run_metadata(run_dir)

def _collect_run_tables(
    experiment_dir: Path,
    protocol_subdir: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _collect_run_tables_base(
        experiment_dir=experiment_dir,
        protocol_subdir=protocol_subdir,
        metadata_loader=_load_run_metadata,
    )


def _batch_styles(batch_sizes: list[int]) -> dict[int, tuple[Any, str]]:
    return shared_batch_styles(batch_sizes)


def _plot_metric_vs_exposure(
    ax: plt.Axes,
    plot_rows: pd.DataFrame,
    *,
    value_col: str,
    ci_col: str,
    ylabel: str,
    ylim: tuple[float, float] | None,
) -> None:
    styles = _batch_styles(plot_rows["batch_size"].astype(int).unique().tolist())
    for batch_size in sorted(styles):
        color, marker = styles[batch_size]
        sub = plot_rows[plot_rows["batch_size"] == batch_size].sort_values("exp_min")
        _plot_line_with_band(
            ax,
            x=sub["exp_min"].to_numpy(dtype=float),
            y=sub[value_col].to_numpy(dtype=float),
            ci=sub[ci_col].to_numpy(dtype=float),
            color=color,
            marker=marker,
            linewidth=PLOT_LINEWIDTH,
            markersize=PLOT_MARKERSIZE,
            alpha=0.16,
            label=str(batch_size),
            bounded=ylim == (0.0, BOUNDED_METRIC_YMAX),
        )
    _apply_axis_finish(
        ax,
        title="",
        xlabel=clean_name("exp_min", X_AXIS_CLEAN_NAMES),
        ylabel=ylabel,
        ylim=ylim,
    )


def _plot_attack_vs_exposure(plot_rows: pd.DataFrame, title: str) -> plt.Figure:
    set_plot_paper_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    _plot_metric_vs_exposure(
        ax,
        plot_rows,
        value_col="tableak_acc_mean",
        ci_col="tableak_acc_ci95",
        ylabel=ATTACK_METRIC_LABEL,
        ylim=(0.0, BOUNDED_METRIC_YMAX),
    )
    ax.set_title(title)
    style_legend(ax.legend(title="Client Batch Size", ncol=2, frameon=True))
    return fig


def _plot_final_checkpoint_tradeoff(plot_rows: pd.DataFrame, title: str) -> plt.Figure:
    set_plot_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    styles = _batch_styles(plot_rows["batch_size"].astype(int).unique().tolist())
    utility_metric = str(plot_rows["utility_metric"].iloc[0])

    final_rows = (
        plot_rows.sort_values(["batch_size", "exp_min"])
        .groupby("batch_size", as_index=False)
        .tail(1)
        .sort_values("batch_size")
        .reset_index(drop=True)
    )

    for batch_size in sorted(styles):
        color, marker = styles[batch_size]
        row = final_rows[final_rows["batch_size"] == batch_size]
        if row.empty:
            continue
        x = float(row["utility_mean"].iloc[0])
        y = float(row["tableak_acc_mean"].iloc[0])
        ax.scatter(
            x,
            y,
            color=color,
            marker=marker,
            s=PLOT_SCATTER_SIZE,
            edgecolors=PLOT_MARKER_EDGECOLOR,
            linewidths=PLOT_MARKER_EDGEWIDTH,
            zorder=PLOT_MARKER_ZORDER,
            clip_on=False,
            label=str(batch_size),
        )

    _apply_axis_finish(
        ax,
        title=title,
        xlabel=clean_name(utility_metric, FL_METRIC_CLEAN_NAMES),
        ylabel=ATTACK_METRIC_LABEL,
        ylim=(0.0, BOUNDED_METRIC_YMAX),
    )
    xlim = fl_metric_limits(utility_metric, plot_rows["utility_mean"])
    if xlim is not None:
        ax.set_xlim(*xlim)
    style_legend(ax.legend(title="Client Batch Size", ncol=2, frameon=True))
    ax.text(
        0.02,
        0.02,
        "One point per batch size at the synchronized 100% checkpoint.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )
    return fig


def _plot_cross_model_final_checkpoint_tradeoff(
    plot_rows: pd.DataFrame,
    *,
    protocol_clean_name: str,
    dataset_clean_name: str,
    task_clean_name: str,
) -> plt.Figure:
    set_plot_paper_style()
    fig, ax = plt.subplots(figsize=(7.4, 5.3))

    utility_metric = str(plot_rows["utility_metric"].iloc[0])
    final_rows = (
        plot_rows.sort_values(["model_name", "batch_size", "exp_min"])
        .groupby(["model_name", "batch_size"], as_index=False)
        .tail(1)
        .sort_values(["batch_size", "model_name"])
        .reset_index(drop=True)
    )
    batch_styles = _batch_styles(final_rows["batch_size"].astype(int).unique().tolist())
    model_names = _ordered_model_names(final_rows["model_name"].astype(str).unique().tolist())
    model_markers = {
        model_name: PLOT_MARKERS[idx % len(PLOT_MARKERS)]
        for idx, model_name in enumerate(model_names)
    }

    for row in final_rows.itertuples(index=False):
        batch_size = int(row.batch_size)
        model_name = str(row.model_name)
        color, _ = batch_styles[batch_size]
        ax.scatter(
            float(row.utility_mean),
            float(row.tableak_acc_mean),
            color=color,
            marker=model_markers[model_name],
            s=PLOT_SCATTER_SIZE * 1.35,
            edgecolors=PLOT_MARKER_EDGECOLOR,
            linewidths=PLOT_MARKER_EDGEWIDTH,
            zorder=PLOT_MARKER_ZORDER,
            clip_on=False,
        )

    _apply_axis_finish(
        ax,
        title=(
            f"{protocol_clean_name}: Final Privacy-Utility Trade-off "
            f"({dataset_clean_name}, {task_clean_name})"
        ),
        xlabel=clean_name(utility_metric, FL_METRIC_CLEAN_NAMES),
        ylabel=ATTACK_METRIC_LABEL,
        ylim=(0.0, BOUNDED_METRIC_YMAX),
    )
    xlim = fl_metric_limits(utility_metric, final_rows["utility_mean"])
    if xlim is not None:
        ax.set_xlim(*xlim)

    batch_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=batch_styles[batch_size][0],
            markeredgecolor=PLOT_MARKER_EDGECOLOR,
            markeredgewidth=PLOT_MARKER_EDGEWIDTH,
            markersize=PLOT_MARKERSIZE + 1.0,
            label=str(batch_size),
        )
        for batch_size in sorted(batch_styles)
    ]
    model_handles = [
        Line2D(
            [0],
            [0],
            marker=model_markers[model_name],
            linestyle="",
            markerfacecolor="0.25",
            markeredgecolor=PLOT_MARKER_EDGECOLOR,
            markeredgewidth=PLOT_MARKER_EDGEWIDTH,
            markersize=PLOT_MARKERSIZE + 1.0,
            label=clean_name(model_name, MODEL_CLEAN_NAMES),
        )
        for model_name in model_names
    ]
    ax.text(
        0.02,
        0.02,
        "One point per model and batch size at the synchronized 100% checkpoint.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.8"},
    )
    batch_legend = ax.legend(
        handles=batch_handles,
        title="Client Batch Size",
        ncol=3,
        frameon=True,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.13),
        borderaxespad=0.0,
    )
    style_legend(batch_legend)
    ax.add_artist(batch_legend)
    style_legend(
        ax.legend(
            handles=model_handles,
            title="Model Architecture",
            frameon=True,
            loc="lower left",
            bbox_to_anchor=(0.02, 0.42),
            borderaxespad=0.0,
        )
    )
    return fig


def _plot_checkpoint_summary(
    plot_rows: pd.DataFrame,
    value_col: str,
    ci_col: str,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None,
) -> plt.Figure:
    set_plot_paper_style()
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    _plot_checkpoint_summary_on_axes(
        ax=ax,
        plot_rows=plot_rows,
        value_col=value_col,
        ci_col=ci_col,
        ylabel=ylabel,
        title=title,
        ylim=ylim,
    )
    return fig


def _plot_checkpoint_summary_on_axes(
    *,
    ax: plt.Axes,
    plot_rows: pd.DataFrame,
    value_col: str,
    ci_col: str,
    ylabel: str,
    title: str,
    ylim: tuple[float, float] | None,
) -> None:
    rows = plot_rows.dropna(subset=[value_col]).copy()
    checkpoints = (
        rows[["checkpoint_idx", "checkpoint_progress_pct"]]
        .drop_duplicates()
        .sort_values("checkpoint_idx")
    )
    colors = sample_checkpoint_colors(len(checkpoints))

    x_labels = sorted(rows["batch_size"].astype(int).unique().tolist())
    x_lookup = {batch_size: idx for idx, batch_size in enumerate(x_labels)}

    for idx, checkpoint in enumerate(checkpoints.itertuples(index=False)):
        sub = rows[rows["checkpoint_idx"] == int(checkpoint.checkpoint_idx)].sort_values("batch_size")
        x = np.array([x_lookup[int(batch_size)] for batch_size in sub["batch_size"].tolist()], dtype=float)
        y = sub[value_col].to_numpy(dtype=float)
        ci = sub[ci_col].to_numpy(dtype=float)
        bounded = ylim is not None and ylim[0] == 0.0 and ylim[1] == BOUNDED_METRIC_YMAX
        yerr = _errorbar_yerr(y, ci, bounded=bounded)
        progress = float(checkpoint.checkpoint_progress_pct)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt=f"{PLOT_MARKERS[idx % len(PLOT_MARKERS)]}--",
            color=colors[idx],
            linewidth=PLOT_LINEWIDTH,
            markersize=PLOT_MARKERSIZE,
            markeredgecolor=PLOT_MARKER_EDGECOLOR,
            markeredgewidth=PLOT_MARKER_EDGEWIDTH,
            capsize=3.0,
            label=f"Checkpoint {int(round(progress))}%",
            zorder=PLOT_MARKER_ZORDER,
            clip_on=False,
        )

    ax.set_xticks(np.arange(len(x_labels), dtype=float))
    ax.set_xticklabels([str(batch_size) for batch_size in x_labels], rotation=0)
    _apply_axis_finish(
        ax,
        title=title,
        xlabel="Client Batch Size",
        ylabel=ylabel,
        ylim=ylim,
    )
    style_legend(ax.legend(title="Attack Checkpoint (% FL Training)", ncol=2, frameon=True))


def _plot_panel(
    attack_plot: pd.DataFrame,
    utility_plot: pd.DataFrame,
    tradeoff_plot: pd.DataFrame,
    attack_checkpoint_plot: pd.DataFrame,
    panel_title: str,
) -> plt.Figure:
    set_plot_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0))
    styles = _batch_styles(attack_plot["batch_size"].astype(int).unique().tolist())
    utility_metric = str(utility_plot["utility_metric"].iloc[0])
    utility_ylim = fl_metric_limits(utility_metric, utility_plot["utility_mean"])
    final_rows = (
        tradeoff_plot.sort_values(["batch_size", "exp_min"])
        .groupby("batch_size", as_index=False)
        .tail(1)
        .sort_values("batch_size")
        .reset_index(drop=True)
    )

    for batch_size in sorted(styles):
        color, marker = styles[batch_size]

        attack_sub = attack_plot[attack_plot["batch_size"] == batch_size].sort_values("exp_min")
        _plot_line_with_band(
            axes[0, 0],
            x=attack_sub["exp_min"].to_numpy(dtype=float),
            y=attack_sub["tableak_acc_mean"].to_numpy(dtype=float),
            ci=attack_sub["tableak_acc_ci95"].to_numpy(dtype=float),
            color=color,
            marker=marker,
            linewidth=PLOT_LINEWIDTH,
            markersize=PLOT_MARKERSIZE,
            alpha=0.14,
            label=str(batch_size),
            bounded=True,
        )

        utility_sub = utility_plot[utility_plot["batch_size"] == batch_size].sort_values("exp_min")
        plot_fl_metric_vs_exposure_on_axes(
            axes[0, 1],
            utility_sub,
            title="",
            value_col="utility_mean",
            ci_col="utility_ci95",
            linewidth=PLOT_LINEWIDTH,
            markersize=PLOT_MARKERSIZE,
            alpha=0.14,
            styles={batch_size: (color, marker)},
            show_legend_labels=False,
            ylim=utility_ylim,
        )

        row = final_rows[final_rows["batch_size"] == batch_size]
        if not row.empty:
            x = float(row["utility_mean"].iloc[0])
            y = float(row["tableak_acc_mean"].iloc[0])
            axes[1, 0].scatter(
                x,
                y,
                color=color,
                marker=marker,
                s=PLOT_SCATTER_SIZE,
                edgecolors=PLOT_MARKER_EDGECOLOR,
                linewidths=PLOT_MARKER_EDGEWIDTH,
                zorder=PLOT_MARKER_ZORDER,
                clip_on=False,
            )

    _plot_checkpoint_summary_on_axes(
        ax=axes[1, 1],
        plot_rows=attack_checkpoint_plot,
        value_col="tableak_acc_mean",
        ci_col="tableak_acc_ci95",
        ylabel=ATTACK_METRIC_LABEL,
        title="D. Attack vs Client Batch Size at Attack Checkpoints",
        ylim=(0.0, BOUNDED_METRIC_YMAX),
    )

    _apply_axis_finish(
        axes[0, 0],
        title="A. Attack vs Exposure",
        xlabel=clean_name("exp_min", X_AXIS_CLEAN_NAMES),
        ylabel=ATTACK_METRIC_LABEL,
        ylim=(0.0, BOUNDED_METRIC_YMAX),
    )
    _apply_axis_finish(
        axes[0, 1],
        title="B. Utility vs Exposure",
        xlabel=clean_name("exp_min", X_AXIS_CLEAN_NAMES),
        ylabel=clean_name(utility_metric, FL_METRIC_CLEAN_NAMES),
        ylim=utility_ylim,
    )
    _apply_axis_finish(
        axes[1, 0],
        title="C. Final Privacy-Utility Trade-off",
        xlabel=clean_name(utility_metric, FL_METRIC_CLEAN_NAMES),
        ylabel=ATTACK_METRIC_LABEL,
        ylim=(0.0, BOUNDED_METRIC_YMAX),
    )
    xlim = fl_metric_limits(utility_metric, tradeoff_plot["utility_mean"])
    if xlim is not None:
        axes[1, 0].set_xlim(*xlim)
    style_legend(axes[0, 0].legend(title="Client Batch Size", ncol=3, frameon=True, loc="lower right"))
    fig.suptitle(panel_title, y=1.01)
    fig.tight_layout()
    return fig


def _plot_cross_model_attack_vs_exposure(
    dataset_attack: pd.DataFrame,
    *,
    protocol_clean_name: str,
    dataset_clean_name: str,
    task_clean_name: str,
) -> plt.Figure:
    set_plot_paper_style()
    model_names = _ordered_model_names(dataset_attack["model_name"].astype(str).unique().tolist())
    fig, axes = plt.subplots(1, len(model_names), figsize=(4.5 * len(model_names), 4.6), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes)
    styles = _batch_styles(dataset_attack["batch_size"].astype(int).unique().tolist())

    for ax, model_name in zip(axes_array, model_names, strict=False):
        model_rows = dataset_attack[dataset_attack["model_name"] == model_name].copy()
        for batch_size in sorted(styles):
            color, marker = styles[batch_size]
            sub = model_rows[model_rows["batch_size"] == batch_size].sort_values("exp_min")
            _plot_line_with_band(
                ax,
                x=sub["exp_min"].to_numpy(dtype=float),
                y=sub["tableak_acc_mean"].to_numpy(dtype=float),
                ci=sub["tableak_acc_ci95"].to_numpy(dtype=float),
                color=color,
                marker=marker,
                linewidth=PLOT_LINEWIDTH,
                markersize=PLOT_MARKERSIZE,
                alpha=0.14,
                label=str(batch_size),
                bounded=True,
            )
        _apply_axis_finish(
            ax,
            title=clean_name(model_name, MODEL_CLEAN_NAMES),
            xlabel=clean_name("exp_min", X_AXIS_CLEAN_NAMES),
            ylabel=ATTACK_METRIC_LABEL if ax is axes_array[0] else "",
            ylim=(0.0, BOUNDED_METRIC_YMAX),
        )

    style_legend(axes_array[0].legend(title="Client Batch Size", ncol=2, frameon=True, loc="lower right"))
    fig.suptitle(
        f"{protocol_clean_name}: {ATTACK_METRIC_LABEL} Across Models ({dataset_clean_name}, {task_clean_name})",
        y=1.01,
    )
    fig.tight_layout()
    return fig


def _plot_model_batch_attack_metrics(
    dataset_attack: pd.DataFrame,
    *,
    protocol_clean_name: str,
    dataset_clean_name: str,
    task_clean_name: str,
) -> plt.Figure:
    set_plot_paper_style()
    model_names = _ordered_model_names(dataset_attack["model_name"].astype(str).unique().tolist())
    batch_sizes = sorted(dataset_attack["batch_size"].astype(int).unique().tolist())
    metric_colors = sample_group_colors(3)
    metric_specs = [
        ("tableak_acc_mean", "tableak_acc_ci95", ATTACK_METRIC_LABEL, metric_colors[0], "o"),
        ("num_acc_mean", "num_acc_ci95", "Numerical Accuracy", metric_colors[1], "s"),
        ("cat_acc_mean", "cat_acc_ci95", "Categorical Accuracy", metric_colors[2], "^"),
    ]
    fig, axes = plt.subplots(
        len(model_names),
        len(batch_sizes),
        figsize=(2.35 * len(batch_sizes), 2.7 * len(model_names)),
        sharex=True,
        sharey=True,
    )
    axes_grid = np.atleast_2d(axes)
    legend_handles: list[Any] = []
    legend_labels: list[str] = []

    for row_idx, model_name in enumerate(model_names):
        model_rows = dataset_attack[dataset_attack["model_name"] == model_name].copy()
        for col_idx, batch_size in enumerate(batch_sizes):
            ax = axes_grid[row_idx, col_idx]
            cell_rows = model_rows[model_rows["batch_size"] == batch_size].sort_values("exp_min").copy()
            if cell_rows.empty:
                ax.set_axis_off()
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=10)
                continue

            for value_col, ci_col, metric_label, color, marker in metric_specs:
                sub = cell_rows.dropna(subset=[value_col]).copy()
                if sub.empty:
                    continue
                _plot_line_with_band(
                    ax,
                    x=sub["exp_min"].to_numpy(dtype=float),
                    y=sub[value_col].to_numpy(dtype=float),
                    ci=sub[ci_col].fillna(0.0).to_numpy(dtype=float),
                    color=color,
                    marker=marker,
                    linewidth=PLOT_LINEWIDTH,
                    markersize=PLOT_MARKERSIZE,
                    alpha=0.12,
                    label=metric_label if row_idx == 0 and col_idx == 0 else None,
                    bounded=True,
                )

            prior_sub = cell_rows.dropna(subset=["prior_tableak_acc_mean"]).sort_values("exp_min")
            if not prior_sub.empty:
                ax.plot(
                    prior_sub["exp_min"].to_numpy(dtype=float),
                    prior_sub["prior_tableak_acc_mean"].to_numpy(dtype=float),
                    color=BASELINE_PRIOR_COLOR,
                    marker="P",
                    linestyle=BASELINE_LINESTYLE,
                    linewidth=PLOT_LINEWIDTH,
                    markersize=PLOT_MARKERSIZE,
                    markeredgecolor=PLOT_MARKER_EDGECOLOR,
                    markeredgewidth=PLOT_MARKER_EDGEWIDTH,
                    alpha=0.95,
                    label="Client Prior Baseline" if row_idx == 0 and col_idx == 0 else None,
                    zorder=PLOT_MARKER_ZORDER,
                    clip_on=False,
                )
            random_sub = cell_rows.dropna(subset=["random_tableak_acc_mean"]).sort_values("exp_min")
            if not random_sub.empty:
                ax.plot(
                    random_sub["exp_min"].to_numpy(dtype=float),
                    random_sub["random_tableak_acc_mean"].to_numpy(dtype=float),
                    color=BASELINE_RANDOM_COLOR,
                    marker="X",
                    linestyle=BASELINE_LINESTYLE,
                    linewidth=PLOT_LINEWIDTH,
                    markersize=PLOT_MARKERSIZE,
                    markeredgecolor=PLOT_MARKER_EDGECOLOR,
                    markeredgewidth=PLOT_MARKER_EDGEWIDTH,
                    alpha=0.95,
                    label="Uniform Random Baseline" if row_idx == 0 and col_idx == 0 else None,
                    zorder=PLOT_MARKER_ZORDER,
                    clip_on=False,
                )
            if row_idx == 0 and col_idx == 0:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

            title = f"Client Batch Size {batch_size}" if row_idx == 0 else ""
            y_label = clean_name(model_name, MODEL_CLEAN_NAMES) if col_idx == 0 else ""
            _apply_axis_finish(
                ax,
                title=title,
                xlabel=clean_name("exp_min", X_AXIS_CLEAN_NAMES) if row_idx == len(model_names) - 1 else "",
                ylabel=y_label,
                ylim=(0.0, BOUNDED_METRIC_YMAX),
            )

    if legend_handles:
        display_order = [0, 3, 1, 4, 2]
        ordered_handles = [legend_handles[idx] for idx in display_order if idx < len(legend_handles)]
        ordered_labels = [legend_labels[idx] for idx in display_order if idx < len(legend_labels)]
        style_legend(
            fig.legend(
                ordered_handles,
                ordered_labels,
                loc="upper center",
                ncol=3,
                frameon=True,
                bbox_to_anchor=(0.5, 0.997),
            )
        )
    fig.suptitle(
        f"{protocol_clean_name}: Attack Metrics Across Models and Client Batch Sizes ({dataset_clean_name}, {task_clean_name})",
        y=1.01,
    )
    fig.tight_layout()
    return fig

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the full figure package for a FedSGD batch-size experiment."
    )
    parser.add_argument(
        "experiment_dir",
        help='Experiment directory name or path, e.g. "experiment_fedsgdbatchsizes_20260324_230729_514418_pandemic".',
    )
    parser.add_argument("--results-root", default="")
    parser.add_argument("--protocol-subdir", default="fedsgd")
    args = parser.parse_args()

    experiment_dir = _resolve_experiment_dir(args.results_root, args.experiment_dir)
    attack_rows, utility_rows, pareto_rows, final_test_rows = _collect_run_tables(
        experiment_dir=experiment_dir,
        protocol_subdir=args.protocol_subdir,
    )

    attack_plot = _rename_plot_columns(
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
    utility_plot = _rename_plot_columns(
        utility_rows,
        {"utility_value": "utility_mean"},
    )
    pareto_plot = _rename_plot_columns(
        pareto_rows,
        {"tableak_acc": "tableak_acc_mean", "utility_value": "utility_mean"},
    )
    final_test_plot = (
        _rename_plot_columns(final_test_rows, {"utility_value": "utility_mean"})
        if not final_test_rows.empty
        else pd.DataFrame()
    )
    attack_checkpoint_plot = _rename_plot_columns(
        _annotate_checkpoint_progress(attack_rows),
        {"exp_min": "exp_min_mean", "tableak_acc": "tableak_acc_mean"},
    )
    utility_checkpoint_plot = _rename_plot_columns(
        _annotate_checkpoint_progress(_sample_utility_at_attack_checkpoints(attack_rows, utility_rows)),
        {"exp_min": "exp_min_mean", "utility_value": "utility_mean"},
    )

    plots_dir = experiment_dir / "plots"
    plot_dirs = prepare_plot_dirs(plots_dir)
    data_dir = plot_dirs["data"]
    image_singles_dir = plot_dirs["image_singles"]
    image_panels_dir = plot_dirs["image_panels"]
    pdf_singles_dir = plot_dirs["pdf_singles"]
    pdf_panels_dir = plot_dirs["pdf_panels"]

    dataset_names = sorted(attack_plot["dataset_name"].astype(str).unique().tolist())
    for dataset_name in dataset_names:
        dataset_attack = attack_plot[attack_plot["dataset_name"] == dataset_name].copy()
        dataset_utility = utility_plot[utility_plot["dataset_name"] == dataset_name].copy()
        dataset_pareto = pareto_plot[pareto_plot["dataset_name"] == dataset_name].copy()
        dataset_final_test = final_test_plot[final_test_plot["dataset_name"] == dataset_name].copy() if not final_test_plot.empty else pd.DataFrame()
        dataset_attack_checkpoint = attack_checkpoint_plot[attack_checkpoint_plot["dataset_name"] == dataset_name].copy()
        dataset_utility_checkpoint = utility_checkpoint_plot[utility_checkpoint_plot["dataset_name"] == dataset_name].copy()
        if dataset_attack.empty or dataset_pareto.empty:
            continue

        dataset_suffix = _sanitize_for_filename(dataset_name)
        protocol_clean_name = clean_name(args.protocol_subdir, PROTOCOL_CLEAN_NAMES)
        dataset_clean_name = clean_name(dataset_name, DATASET_CLEAN_NAMES)
        task_clean_name = clean_name(str(dataset_attack["task_objective"].iloc[0]), TASK_CLEAN_NAMES)

        cross_model_attack_base = f"fedsgd_batch_sizes_cross_model_attack_vs_exposure__dataset_{dataset_suffix}"
        cross_model_tradeoff_base = f"fedsgd_batch_sizes_cross_model_final_checkpoint_tradeoff__dataset_{dataset_suffix}"
        model_batch_attack_metrics_base = f"fedsgd_batch_sizes_model_batch_attack_metrics__dataset_{dataset_suffix}"

        cross_png, cross_pdf = _save_figure(
            _plot_cross_model_attack_vs_exposure(
                dataset_attack,
                protocol_clean_name=protocol_clean_name,
                dataset_clean_name=dataset_clean_name,
                task_clean_name=task_clean_name,
            ),
            image_panels_dir / cross_model_attack_base,
            pdf_base_path=pdf_panels_dir / cross_model_attack_base,
        )
        tradeoff_png, tradeoff_pdf = _save_figure(
            _plot_cross_model_final_checkpoint_tradeoff(
                dataset_pareto,
                protocol_clean_name=protocol_clean_name,
                dataset_clean_name=dataset_clean_name,
                task_clean_name=task_clean_name,
            ),
            image_singles_dir / cross_model_tradeoff_base,
            pdf_base_path=pdf_singles_dir / cross_model_tradeoff_base,
        )
        metrics_png, metrics_pdf = _save_figure(
            _plot_model_batch_attack_metrics(
                dataset_attack,
                protocol_clean_name=protocol_clean_name,
                dataset_clean_name=dataset_clean_name,
                task_clean_name=task_clean_name,
            ),
            image_panels_dir / model_batch_attack_metrics_base,
            pdf_base_path=pdf_panels_dir / model_batch_attack_metrics_base,
        )
        print(cross_png)
        print(cross_pdf)
        print(tradeoff_png)
        print(tradeoff_pdf)
        print(metrics_png)
        print(metrics_pdf)

        dataset_attack.to_csv(data_dir / f"{cross_model_attack_base}_data.csv", index=False)
        cross_model_tradeoff_rows = (
            dataset_pareto.sort_values(["model_name", "batch_size", "exp_min"])
            .groupby(["model_name", "batch_size"], as_index=False)
            .tail(1)
            .sort_values(["batch_size", "model_name"])
            .reset_index(drop=True)
        )
        cross_model_tradeoff_rows.to_csv(data_dir / f"{cross_model_tradeoff_base}_data.csv", index=False)
        dataset_attack.to_csv(data_dir / f"{model_batch_attack_metrics_base}_data.csv", index=False)

        model_names = sorted(dataset_attack["model_name"].astype(str).unique().tolist())
        for model_name in model_names:
            attack_slice = dataset_attack[dataset_attack["model_name"] == model_name].copy()
            utility_slice = dataset_utility[dataset_utility["model_name"] == model_name].copy()
            pareto_slice = dataset_pareto[dataset_pareto["model_name"] == model_name].copy()
            final_test_slice = dataset_final_test[dataset_final_test["model_name"] == model_name].copy() if not dataset_final_test.empty else pd.DataFrame()
            attack_checkpoint_slice = dataset_attack_checkpoint[
                dataset_attack_checkpoint["model_name"] == model_name
            ].copy()
            utility_checkpoint_slice = dataset_utility_checkpoint[
                dataset_utility_checkpoint["model_name"] == model_name
            ].copy()

            if utility_slice.empty or pareto_slice.empty or attack_checkpoint_slice.empty or utility_checkpoint_slice.empty:
                print(f"Skipping incomplete plot bundle for dataset={dataset_name} model={model_name}")
                continue

            model_suffix = _sanitize_for_filename(model_name)
            model_clean_name = clean_name(model_name, MODEL_CLEAN_NAMES)
            utility_metric = str(utility_slice["utility_metric"].iloc[0])
            utility_label = clean_name(utility_metric, FL_METRIC_CLEAN_NAMES)

            attack_title = (
                f"{protocol_clean_name}: {ATTACK_METRIC_LABEL} Across Client Batch Sizes "
                f"({model_clean_name}, {dataset_clean_name}, {task_clean_name})"
            )
            utility_title = (
                f"{protocol_clean_name}: {utility_label} Across Client Batch Sizes "
                f"({model_clean_name}, {dataset_clean_name}, {task_clean_name})"
            )
            final_test_metric = (
                str(final_test_slice["utility_metric"].iloc[0])
                if not final_test_slice.empty
                else metric_for_final_test(utility_metric)
            )
            final_test_label = clean_name(final_test_metric, FL_METRIC_CLEAN_NAMES)
            final_test_title = (
                f"{protocol_clean_name}: {final_test_label} at Best Validation Checkpoint Exposure "
                f"({model_clean_name}, {dataset_clean_name}, {task_clean_name})"
            )
            pareto_title = (
                f"{protocol_clean_name}: Privacy-Utility Trade-off Across Client Batch Sizes "
                f"({model_clean_name}, {dataset_clean_name})"
            )
            final_checkpoint_tradeoff_title = (
                f"{protocol_clean_name}: Final Checkpoint Privacy-Utility Trade-off "
                f"({model_clean_name}, {dataset_clean_name})"
            )
            attack_summary_title = (
                f"{protocol_clean_name}: Attack Success by Client Batch Size at Attack Checkpoints "
                f"({model_clean_name}, {dataset_clean_name})"
            )
            utility_summary_title = (
                f"{protocol_clean_name}: {utility_label} at Attack Checkpoints "
                f"({model_clean_name}, {dataset_clean_name})"
            )
            panel_title = (
                f"{protocol_clean_name}: Batch-Size Privacy-Utility Overview "
                f"({model_clean_name}, {dataset_clean_name}, {task_clean_name})"
            )

            attack_base = f"fedsgd_batch_sizes_attack_vs_exposure__model_{model_suffix}__dataset_{dataset_suffix}"
            utility_base = f"fedsgd_batch_sizes_utility_vs_exposure__model_{model_suffix}__dataset_{dataset_suffix}"
            final_test_base = f"fedsgd_batch_sizes_final_test_utility_vs_exposure__model_{model_suffix}__dataset_{dataset_suffix}"
            final_checkpoint_tradeoff_base = (
                f"fedsgd_batch_sizes_final_checkpoint_tradeoff__model_{model_suffix}__dataset_{dataset_suffix}"
            )
            attack_summary_base = (
                f"fedsgd_batch_sizes_attack_by_batch_at_checkpoints__model_{model_suffix}__dataset_{dataset_suffix}"
            )
            utility_summary_base = (
                f"fedsgd_batch_sizes_utility_by_batch_at_checkpoints__model_{model_suffix}__dataset_{dataset_suffix}"
            )
            panel_base = f"fedsgd_batch_sizes_panel__model_{model_suffix}__dataset_{dataset_suffix}"

            output_paths: list[tuple[Path, Path]] = []

            output_paths.append(
                _save_figure(
                    _plot_attack_vs_exposure(attack_slice, attack_title),
                    image_singles_dir / attack_base,
                    pdf_base_path=pdf_singles_dir / attack_base,
                )
            )
            output_paths.append(
                _save_figure(
                    plot_fl_metric_vs_exposure(utility_slice, utility_title),
                    image_singles_dir / utility_base,
                    pdf_base_path=pdf_singles_dir / utility_base,
                )
            )
            if not final_test_slice.empty:
                output_paths.append(
                    _save_figure(
                        plot_fl_metric_vs_exposure(final_test_slice, final_test_title),
                        image_singles_dir / final_test_base,
                        pdf_base_path=pdf_singles_dir / final_test_base,
                    )
                )
            output_paths.append(
                _save_figure(
                    _plot_final_checkpoint_tradeoff(pareto_slice, final_checkpoint_tradeoff_title),
                    image_singles_dir / final_checkpoint_tradeoff_base,
                    pdf_base_path=pdf_singles_dir / final_checkpoint_tradeoff_base,
                )
            )
            output_paths.append(
                _save_figure(
                    _plot_checkpoint_summary(
                        attack_checkpoint_slice,
                        value_col="tableak_acc_mean",
                        ci_col="tableak_acc_ci95",
                        ylabel=ATTACK_METRIC_LABEL,
                        title=attack_summary_title,
                        ylim=(0.0, BOUNDED_METRIC_YMAX),
                    ),
                    image_singles_dir / attack_summary_base,
                    pdf_base_path=pdf_singles_dir / attack_summary_base,
                )
            )
            output_paths.append(
                _save_figure(
                    _plot_checkpoint_summary(
                        utility_checkpoint_slice,
                        value_col="utility_mean",
                        ci_col="utility_ci95",
                        ylabel=utility_label,
                        title=utility_summary_title,
                        ylim=fl_metric_limits(utility_metric, utility_checkpoint_slice["utility_mean"]),
                    ),
                    image_singles_dir / utility_summary_base,
                    pdf_base_path=pdf_singles_dir / utility_summary_base,
                )
            )
            output_paths.append(
                _save_figure(
                    _plot_panel(
                        attack_slice,
                        utility_slice,
                        pareto_slice,
                        attack_checkpoint_slice,
                        panel_title,
                    ),
                    image_panels_dir / panel_base,
                    pdf_base_path=pdf_panels_dir / panel_base,
                )
            )
            for png_path, pdf_path in output_paths:
                print(png_path)
                print(pdf_path)

            attack_slice.to_csv(data_dir / f"{attack_base}_data.csv", index=False)
            utility_slice.to_csv(data_dir / f"{utility_base}_data.csv", index=False)
            if not final_test_slice.empty:
                final_test_slice.to_csv(data_dir / f"{final_test_base}_data.csv", index=False)
            final_rows = (
                pareto_slice.sort_values(["batch_size", "exp_min"])
                .groupby("batch_size", as_index=False)
                .tail(1)
                .sort_values("batch_size")
                .reset_index(drop=True)
            )
            final_rows.to_csv(data_dir / f"{final_checkpoint_tradeoff_base}_data.csv", index=False)
            attack_checkpoint_slice.to_csv(data_dir / f"{attack_summary_base}_data.csv", index=False)
            utility_checkpoint_slice.to_csv(data_dir / f"{utility_summary_base}_data.csv", index=False)


if __name__ == "__main__":
    main()
