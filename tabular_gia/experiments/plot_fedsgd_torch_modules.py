import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import plot_fedsgd_batch_sizes as batch_plots
from plot_helper import (
    DATASET_CLEAN_NAMES,
    FL_METRIC_CLEAN_NAMES,
    PROTOCOL_CLEAN_NAMES,
    TASK_CLEAN_NAMES,
    X_AXIS_CLEAN_NAMES,
    clean_name,
    fl_metric_limits,
    infer_task_objective,
    set_plot_paper_style,
)

FACTOR_SPECS = [
    ("d_hidden", "Hidden Width", [32, 64, 128]),
    ("n_hidden_layers", "Hidden Layers", [1, 2, 3]),
    ("norm", "Normalization", ["batchnorm", "layernorm"]),
    ("dropout", "Dropout", [0.0, 0.1]),
    ("activation", "Activation", ["relu", "gelu"]),
]

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

def _get_nested(cfg: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _load_run_metadata(run_dir: Path) -> dict[str, Any] | None:
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    dataset_path = str(_get_nested(cfg, "dataset.dataset_path", "unknown"))
    dataset_name = Path(dataset_path).stem
    task_objective = infer_task_objective(dataset_name=dataset_name, dataset_path=dataset_path)
    return {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "task_objective": task_objective,
        "utility_metric": batch_plots.DEFAULT_Y_BY_TASK.get(task_objective, "val_loss"),
        "run_id": run_dir.name,
        "batch_size": int(_get_nested(cfg, "dataset.batch_size")),
        "arch": str(_get_nested(cfg, "model.arch", "unknown")),
        "d_hidden": int(_get_nested(cfg, "model.d_hidden")),
        "n_hidden_layers": int(_get_nested(cfg, "model.n_hidden_layers")),
        "norm": str(_get_nested(cfg, "model.norm")),
        "dropout": float(_get_nested(cfg, "model.dropout")),
        "activation": str(_get_nested(cfg, "model.activation")),
    }


def _collect_endpoint_rows(experiment_dir: Path, protocol_subdir: str) -> pd.DataFrame:
    protocol_dir = experiment_dir / protocol_subdir
    if not protocol_dir.exists():
        raise FileNotFoundError(f"Protocol directory not found: {protocol_dir}")

    endpoint_rows: list[dict[str, Any]] = []
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

        pareto_curve = batch_plots._align_attack_and_utility(attack_curve, utility_curve, metadata)
        if pareto_curve.empty:
            skipped.append(f"{run_dir.name}:empty_pareto")
            continue

        final_row = pareto_curve.sort_values("exp_min").iloc[-1]
        endpoint_rows.append(
            {
                **metadata,
                "exp_min": float(final_row["exp_min"]),
                "tableak_acc": float(final_row["tableak_acc"]),
                "utility_value": float(final_row["utility_value"]),
            }
        )

    if skipped:
        print(f"Skipped {len(skipped)} run(s): {', '.join(skipped)}")

    if not endpoint_rows:
        raise FileNotFoundError(f"No usable endpoint rows found under {protocol_dir}")
    return pd.DataFrame(endpoint_rows)


def _collect_attack_rows(experiment_dir: Path, protocol_subdir: str) -> pd.DataFrame:
    protocol_dir = experiment_dir / protocol_subdir
    if not protocol_dir.exists():
        raise FileNotFoundError(f"Protocol directory not found: {protocol_dir}")

    attack_frames: list[pd.DataFrame] = []
    skipped: list[str] = []

    for run_dir in sorted(protocol_dir.glob("run_*")):
        metadata = _load_run_metadata(run_dir)
        if metadata is None:
            skipped.append(f"{run_dir.name}:missing_run_config")
            continue

        attack_curve = batch_plots._load_attack_curve(run_dir, metadata)
        if attack_curve is None or attack_curve.empty:
            skipped.append(f"{run_dir.name}:missing_attack_curve")
            continue
        attack_frames.append(attack_curve)

    if skipped:
        print(f"Skipped {len(skipped)} attack run(s): {', '.join(skipped)}")

    if not attack_frames:
        raise FileNotFoundError(f"No usable attack rows found under {protocol_dir}")
    return pd.concat(attack_frames, axis=0, ignore_index=True)


def _collect_utility_rows(experiment_dir: Path, protocol_subdir: str) -> pd.DataFrame:
    protocol_dir = experiment_dir / protocol_subdir
    if not protocol_dir.exists():
        raise FileNotFoundError(f"Protocol directory not found: {protocol_dir}")

    utility_frames: list[pd.DataFrame] = []
    skipped: list[str] = []

    for run_dir in sorted(protocol_dir.glob("run_*")):
        metadata = _load_run_metadata(run_dir)
        if metadata is None:
            skipped.append(f"{run_dir.name}:missing_run_config")
            continue

        utility_curve = batch_plots._load_utility_curve(run_dir, metadata)
        if utility_curve is None or utility_curve.empty:
            skipped.append(f"{run_dir.name}:missing_utility_curve")
            continue
        utility_frames.append(utility_curve)

    if skipped:
        print(f"Skipped {len(skipped)} utility run(s): {', '.join(skipped)}")

    if not utility_frames:
        raise FileNotFoundError(f"No usable utility rows found under {protocol_dir}")
    return pd.concat(utility_frames, axis=0, ignore_index=True)


def _aggregate_factor_rows(endpoint_rows: pd.DataFrame) -> pd.DataFrame:
    aggregated_frames: list[pd.DataFrame] = []

    for factor_name, _, _ in FACTOR_SPECS:
        grouped = (
            endpoint_rows.groupby(["dataset_name", "task_objective", "utility_metric", "batch_size", factor_name], dropna=False)
            .agg(
                tableak_acc_mean=("tableak_acc", "mean"),
                tableak_acc_std=("tableak_acc", "std"),
                utility_mean=("utility_value", "mean"),
                utility_std=("utility_value", "std"),
                n_configs=("run_id", "count"),
            )
            .reset_index()
            .rename(columns={factor_name: "factor_level"})
        )
        grouped["factor_name"] = factor_name
        grouped["tableak_acc_std"] = grouped["tableak_acc_std"].fillna(0.0)
        grouped["utility_std"] = grouped["utility_std"].fillna(0.0)
        grouped["tableak_acc_ci95"] = 1.96 * grouped["tableak_acc_std"] / np.sqrt(grouped["n_configs"].clip(lower=1))
        grouped["utility_ci95"] = 1.96 * grouped["utility_std"] / np.sqrt(grouped["n_configs"].clip(lower=1))
        aggregated_frames.append(grouped)

    return pd.concat(aggregated_frames, axis=0, ignore_index=True)


def _factor_level_order(factor_name: str) -> list[Any]:
    for candidate_name, _, levels in FACTOR_SPECS:
        if candidate_name == factor_name:
            return levels
    raise KeyError(f"Unknown factor '{factor_name}'")


def _factor_title(factor_name: str) -> str:
    for candidate_name, title, _ in FACTOR_SPECS:
        if candidate_name == factor_name:
            return title
    raise KeyError(f"Unknown factor '{factor_name}'")


def _format_factor_level(factor_name: str, value: Any) -> str:
    if factor_name in FACTOR_LEVEL_LABELS:
        return FACTOR_LEVEL_LABELS[factor_name].get(str(value), str(value))
    if factor_name == "dropout":
        return f"{float(value):.1f}"
    return str(int(value)) if isinstance(value, (np.integer, int)) or float(value).is_integer() else str(value)


def _plot_endpoint_factor_panel(
    factor_rows: pd.DataFrame,
    *,
    protocol_clean_name: str,
    dataset_clean_name: str,
    task_clean_name: str,
) -> plt.Figure:
    set_plot_paper_style()
    fig, axes = plt.subplots(2, len(FACTOR_SPECS), figsize=(3.2 * len(FACTOR_SPECS), 7.0), sharey="row")
    axes_grid = np.atleast_2d(axes)
    utility_metric = str(factor_rows["utility_metric"].iloc[0])
    utility_label = clean_name(utility_metric, FL_METRIC_CLEAN_NAMES)
    utility_ylim = fl_metric_limits(utility_metric, factor_rows["utility_mean"])
    batch_styles = batch_plots._batch_styles(factor_rows["batch_size"].astype(int).unique().tolist())

    for col_idx, (factor_name, _, _) in enumerate(FACTOR_SPECS):
        attack_ax = axes_grid[0, col_idx]
        utility_ax = axes_grid[1, col_idx]
        factor_slice = factor_rows[factor_rows["factor_name"] == factor_name].copy()
        level_order = _factor_level_order(factor_name)
        x = np.arange(len(level_order), dtype=float)

        for batch_size in sorted(batch_styles):
            color, marker = batch_styles[batch_size]
            batch_slice = factor_slice[factor_slice["batch_size"] == batch_size].copy()
            batch_slice = batch_slice.set_index("factor_level").reindex(level_order)

            y_attack = batch_slice["tableak_acc_mean"].to_numpy(dtype=float)
            y_attack_ci = batch_slice["tableak_acc_ci95"].to_numpy(dtype=float)
            attack_ax.errorbar(
                x,
                y_attack,
                yerr=batch_plots._errorbar_yerr(y_attack, y_attack_ci, bounded=True),
                fmt=f"{marker}-",
                color=color,
                linewidth=1.9,
                markersize=5.0,
                capsize=3.0,
                label=str(batch_size) if col_idx == 0 else None,
            )

            y_utility = batch_slice["utility_mean"].to_numpy(dtype=float)
            y_utility_ci = batch_slice["utility_ci95"].to_numpy(dtype=float)
            bounded_utility = utility_ylim is not None and utility_ylim[0] == 0.0 and utility_ylim[1] == batch_plots.BOUNDED_METRIC_YMAX
            utility_ax.errorbar(
                x,
                y_utility,
                yerr=batch_plots._errorbar_yerr(y_utility, y_utility_ci, bounded=bounded_utility),
                fmt=f"{marker}-",
                color=color,
                linewidth=1.9,
                markersize=5.0,
                capsize=3.0,
            )

        tick_labels = [_format_factor_level(factor_name, level) for level in level_order]
        attack_ax.set_xticks(x)
        attack_ax.set_xticklabels(tick_labels, rotation=0)
        utility_ax.set_xticks(x)
        utility_ax.set_xticklabels(tick_labels, rotation=0)

        batch_plots._apply_axis_finish(
            attack_ax,
            title=_factor_title(factor_name),
            xlabel="",
            ylabel=batch_plots.ATTACK_METRIC_LABEL if col_idx == 0 else "",
            ylim=(0.0, batch_plots.BOUNDED_METRIC_YMAX),
        )
        batch_plots._apply_axis_finish(
            utility_ax,
            title="",
            xlabel=_factor_title(factor_name),
            ylabel=utility_label if col_idx == 0 else "",
            ylim=utility_ylim,
        )

    legend = axes_grid[0, 0].legend(title="Client Batch Size", ncol=2, frameon=True, loc="lower left")
    legend.get_frame().set_alpha(0.92)
    fig.suptitle(
        f"{protocol_clean_name}: Final Factor Sensitivity of Leakage and Utility ({dataset_clean_name}, {task_clean_name})",
        y=1.02,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))
    return fig


def _ordered_curve_combinations() -> list[tuple[str, str, float]]:
    combos: list[tuple[str, str, float]] = []
    for norm_value in _factor_level_order("norm"):
        for activation_value in _factor_level_order("activation"):
            for dropout_value in _factor_level_order("dropout"):
                combos.append((str(norm_value), str(activation_value), float(dropout_value)))
    return combos


def _curve_colors() -> dict[tuple[str, str, float], Any]:
    combos = _ordered_curve_combinations()
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.08, 0.95, len(combos)))
    return {combo: colors[idx] for idx, combo in enumerate(combos)}


def _curve_label(norm_value: str, activation_value: str, dropout_value: float) -> str:
    return (
        f"{_format_factor_level('activation', activation_value)}, "
        f"{_format_factor_level('norm', norm_value)}, "
        f"Dropout {_format_factor_level('dropout', dropout_value)}"
    )


def _tradeoff_axis_limits(endpoint_rows: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    x = pd.to_numeric(endpoint_rows["utility_value"], errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(endpoint_rows["tableak_acc"], errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0 or len(y) == 0:
        return (0.0, 1.0), (0.0, batch_plots.BOUNDED_METRIC_YMAX)

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_span = max(1e-6, x_max - x_min)
    x_pad = 0.08 * x_span
    xlim = (x_min - x_pad, x_max + x_pad)

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_span = max(0.02, y_max - y_min)
    y_pad = max(0.01, 0.08 * y_span)
    ylim = (
        max(0.0, y_min - y_pad),
        min(batch_plots.BOUNDED_METRIC_YMAX, y_max + y_pad),
    )
    return xlim, ylim


def _plot_endpoint_tradeoff_grid(
    endpoint_rows: pd.DataFrame,
    *,
    protocol_clean_name: str,
    dataset_clean_name: str,
    task_clean_name: str,
    batch_size: int | None,
) -> plt.Figure:
    set_plot_paper_style()
    width_levels = _factor_level_order("d_hidden")
    layer_levels = _factor_level_order("n_hidden_layers")
    utility_metric = str(endpoint_rows["utility_metric"].iloc[0])
    utility_label = clean_name(utility_metric, FL_METRIC_CLEAN_NAMES)
    combo_colors = _curve_colors()
    xlim, ylim = _tradeoff_axis_limits(endpoint_rows)

    fig, axes = plt.subplots(
        len(width_levels),
        len(layer_levels),
        figsize=(11.2, 8.8),
        sharex=True,
        sharey=True,
    )
    axes_grid = np.atleast_2d(axes)

    for row_idx, d_hidden in enumerate(width_levels):
        for col_idx, n_layers in enumerate(layer_levels):
            ax = axes_grid[row_idx, col_idx]
            config_rows = endpoint_rows[
                (endpoint_rows["d_hidden"] == d_hidden)
                & (endpoint_rows["n_hidden_layers"] == n_layers)
            ].copy()

            for point in config_rows.itertuples(index=False):
                combo = (str(point.norm), str(point.activation), float(point.dropout))
                color = combo_colors[combo]
                ax.scatter(
                    float(point.utility_value),
                    float(point.tableak_acc),
                    s=42,
                    color=color,
                    edgecolors="white",
                    linewidths=0.6,
                    alpha=0.95,
                    zorder=4,
                )

            if row_idx == 0:
                ax.set_title(f"Layers = {n_layers}")
            xlabel = utility_label if row_idx == len(width_levels) - 1 else ""
            ylabel = f"Width = {d_hidden}\n{batch_plots.ATTACK_METRIC_LABEL}" if col_idx == 0 else ""
            batch_plots._apply_axis_finish(
                ax,
                title=ax.get_title(),
                xlabel=xlabel,
                ylabel=ylabel,
                ylim=ylim,
            )
            ax.set_xlim(*xlim)
            if config_rows.empty:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=11)

    fig.suptitle(
        f"{protocol_clean_name}: Final Privacy-Utility Trade-off ({dataset_clean_name}, {task_clean_name}"
        + (f", Client Batch Size = {batch_size}" if batch_size is not None else "")
        + ")",
        y=1.03,
    )
    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []
    present_combos = set(
        (str(row.norm), str(row.activation), float(row.dropout))
        for row in endpoint_rows[["norm", "activation", "dropout"]].drop_duplicates().itertuples(index=False)
    )
    for combo in _ordered_curve_combinations():
        if combo not in present_combos:
            continue
        color = combo_colors[combo]
        norm_value, activation_value, dropout_value = combo
        label = _curve_label(str(norm_value), str(activation_value), float(dropout_value))
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                markersize=6.5,
            )
        )
        legend_labels.append(label)
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Activation, Normalization, Dropout",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=4,
            frameon=True,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    return fig


def _figure_metric_limits(
    plot_rows: pd.DataFrame,
    *,
    value_col: str,
    ci_col: str,
    default_ylim: tuple[float, float] | None,
) -> tuple[float, float] | None:
    values = pd.to_numeric(plot_rows[value_col], errors="coerce")
    cis = pd.to_numeric(plot_rows[ci_col], errors="coerce").fillna(0.0)
    valid = values.notna()
    if not valid.any():
        return default_ylim

    lower = (values[valid] - cis[valid]).to_numpy(dtype=float)
    upper = (values[valid] + cis[valid]).to_numpy(dtype=float)
    data_min = float(np.nanmin(lower))
    data_max = float(np.nanmax(upper))

    bounded_upper = (
        default_ylim is not None
        and default_ylim[0] == 0.0
        and default_ylim[1] == batch_plots.BOUNDED_METRIC_YMAX
    )
    if bounded_upper:
        data_min = max(0.0, data_min)
        data_max = min(1.0, data_max)
        span = max(0.02, data_max - data_min)
        pad = max(0.01, 0.08 * span)
        ymin = max(0.0, data_min - pad)
        ymax = min(batch_plots.BOUNDED_METRIC_YMAX, data_max + pad)
        return (ymin, ymax)

    span = max(1e-6, data_max - data_min)
    pad = 0.08 * span
    return (data_min - pad, data_max + pad)


def _plot_batch_split_metric_grid(
    plot_rows: pd.DataFrame,
    *,
    protocol_clean_name: str,
    dataset_clean_name: str,
    task_clean_name: str,
    batch_size: int,
    value_col: str,
    ci_col: str,
    y_label: str,
    ylim: tuple[float, float] | None,
    title_metric_name: str,
    tighten_figure_ylim: bool,
    baseline_specs: list[tuple[str, str]] | None = None,
    top_rect: float = 0.98,
) -> plt.Figure:
    set_plot_paper_style()
    width_levels = _factor_level_order("d_hidden")
    layer_levels = _factor_level_order("n_hidden_layers")
    fig, axes = plt.subplots(
        len(width_levels),
        len(layer_levels),
        figsize=(11.2, 8.8),
        sharex=True,
        sharey=True,
    )
    axes_grid = np.atleast_2d(axes)
    curve_colors = _curve_colors()
    baseline_specs = baseline_specs or []
    plot_ylim = (
        _figure_metric_limits(plot_rows, value_col=value_col, ci_col=ci_col, default_ylim=ylim)
        if tighten_figure_ylim
        else ylim
    )
    bounded = plot_ylim is not None and plot_ylim[0] >= 0.0 and plot_ylim[1] <= batch_plots.BOUNDED_METRIC_YMAX

    for row_idx, d_hidden in enumerate(width_levels):
        for col_idx, n_layers in enumerate(layer_levels):
            ax = axes_grid[row_idx, col_idx]
            config_rows = plot_rows[
                (plot_rows["d_hidden"] == d_hidden)
                & (plot_rows["n_hidden_layers"] == n_layers)
            ].copy()

            for norm_value in _factor_level_order("norm"):
                for activation_value in _factor_level_order("activation"):
                    for dropout_value in _factor_level_order("dropout"):
                        curve_rows = config_rows[
                            (config_rows["norm"] == norm_value)
                            & (config_rows["activation"] == activation_value)
                            & (config_rows["dropout"] == float(dropout_value))
                        ].sort_values("exp_min")
                        if curve_rows.empty:
                            continue
                        combo = (str(norm_value), str(activation_value), float(dropout_value))
                        color = curve_colors[combo]
                        label = _curve_label(norm_value, activation_value, float(dropout_value))
                        batch_plots._plot_line_with_band(
                            ax,
                            x=curve_rows["exp_min"].to_numpy(dtype=float),
                            y=curve_rows[value_col].to_numpy(dtype=float),
                            ci=curve_rows[ci_col].fillna(0.0).to_numpy(dtype=float),
                            color=color,
                            marker="o",
                            linewidth=1.6,
                            markersize=3.6,
                            alpha=0.08,
                            bounded=bounded,
                        )
                        for baseline_col, linestyle in baseline_specs:
                            baseline_rows = curve_rows.dropna(subset=[baseline_col]).copy()
                            if baseline_rows.empty:
                                continue
                            baseline_color = "0.25" if "prior_" in baseline_col else "0.45"
                            ax.plot(
                                baseline_rows["exp_min"].to_numpy(dtype=float),
                                baseline_rows[baseline_col].to_numpy(dtype=float),
                                color=baseline_color,
                                linestyle=linestyle,
                                linewidth=1.1,
                                alpha=0.95,
                            )

            if row_idx == 0:
                ax.set_title(f"Layers = {n_layers}")
            xlabel = clean_name("exp_min", X_AXIS_CLEAN_NAMES) if row_idx == len(width_levels) - 1 else ""
            ylabel = f"Width = {d_hidden}\n{y_label}" if col_idx == 0 else ""
            batch_plots._apply_axis_finish(
                ax,
                title=ax.get_title(),
                xlabel=xlabel,
                ylabel=ylabel,
                ylim=plot_ylim,
            )
            if config_rows.empty:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=11)

    fig.suptitle(
        f"{protocol_clean_name}: {title_metric_name} ({dataset_clean_name}, {task_clean_name}, Client Batch Size = {batch_size})",
        y=1.03,
    )
    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []
    present_combos = set(
        (str(row.norm), str(row.activation), float(row.dropout))
        for row in plot_rows[["norm", "activation", "dropout"]].drop_duplicates().itertuples(index=False)
    )
    for norm_value, activation_value, dropout_value in _ordered_curve_combinations():
        combo = (str(norm_value), str(activation_value), float(dropout_value))
        if combo not in present_combos:
            continue
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=curve_colors[combo],
                linestyle="-",
                linewidth=1.8,
                marker="o",
                markersize=4.0,
            )
        )
        legend_labels.append(_curve_label(norm_value, activation_value, float(dropout_value)))
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            title="Activation, Normalization, Dropout",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=4,
            frameon=True,
        )
    if baseline_specs:
        fig.text(
            0.5,
            0.915,
            "Dashed = Client Prior Baseline, dotted = Uniform Random Baseline",
            ha="center",
            va="center",
            fontsize=9.5,
            color="0.25",
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top_rect))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot factor-wise endpoint sensitivity for the FedSGD torch-modules experiment."
    )
    parser.add_argument(
        "experiment_dir",
        nargs="?",
        default="experiment_fedsgdtorchmodules_20260320_004650_016196_adult",
        help='Experiment directory name or path, e.g. "experiment_fedsgdtorchmodules_20260320_004650_016196_adult".',
    )
    parser.add_argument("--results-root", default="")
    parser.add_argument("--protocol-subdir", default="fedsgd")
    args = parser.parse_args()

    experiment_dir = batch_plots._resolve_experiment_dir(args.results_root, args.experiment_dir)
    endpoint_rows = _collect_endpoint_rows(experiment_dir=experiment_dir, protocol_subdir=args.protocol_subdir)
    attack_rows = _collect_attack_rows(experiment_dir=experiment_dir, protocol_subdir=args.protocol_subdir)
    factor_rows = _aggregate_factor_rows(endpoint_rows)
    utility_rows = _collect_utility_rows(experiment_dir=experiment_dir, protocol_subdir=args.protocol_subdir)

    plots_dir = experiment_dir / "plots"
    image_panels_dir = plots_dir / "image" / "panels"
    pdf_panels_dir = plots_dir / "pdf" / "panels"
    data_dir = plots_dir / "data"
    image_panels_dir.mkdir(parents=True, exist_ok=True)
    pdf_panels_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = sorted(factor_rows["dataset_name"].astype(str).unique().tolist())
    for dataset_name in dataset_names:
        dataset_factor_rows = factor_rows[factor_rows["dataset_name"] == dataset_name].copy()
        if dataset_factor_rows.empty:
            continue

        dataset_clean_name = clean_name(dataset_name, DATASET_CLEAN_NAMES)
        task_clean_name = clean_name(str(dataset_factor_rows["task_objective"].iloc[0]), TASK_CLEAN_NAMES)
        protocol_clean_name = clean_name(args.protocol_subdir, PROTOCOL_CLEAN_NAMES)
        dataset_suffix = batch_plots._sanitize_for_filename(dataset_name)
        base_name = f"fedsgd_torch_modules_endpoint_factor_sensitivity__dataset_{dataset_suffix}"

        fig = _plot_endpoint_factor_panel(
            dataset_factor_rows,
            protocol_clean_name=protocol_clean_name,
            dataset_clean_name=dataset_clean_name,
            task_clean_name=task_clean_name,
        )
        batch_plots._save_figure(fig, image_panels_dir / base_name)
        src_pdf = image_panels_dir / f"{base_name}.pdf"
        dst_pdf = pdf_panels_dir / f"{base_name}.pdf"
        if src_pdf.exists():
            src_pdf.replace(dst_pdf)
        print(image_panels_dir / f"{base_name}.png")
        print(dst_pdf)

        dataset_factor_rows.to_csv(data_dir / f"{base_name}_data.csv", index=False)
        endpoint_rows[endpoint_rows["dataset_name"] == dataset_name].to_csv(
            data_dir / f"{base_name}_endpoint_rows.csv",
            index=False,
        )

        dataset_endpoint_rows = endpoint_rows[endpoint_rows["dataset_name"] == dataset_name].copy()
        for batch_size in sorted(dataset_endpoint_rows["batch_size"].astype(int).unique().tolist()):
            batch_endpoint_rows = dataset_endpoint_rows[dataset_endpoint_rows["batch_size"] == batch_size].copy()
            tradeoff_base_name = (
                f"fedsgd_torch_modules_final_tradeoff_grid__dataset_{dataset_suffix}__batch_{batch_size}"
            )
            tradeoff_fig = _plot_endpoint_tradeoff_grid(
                batch_endpoint_rows,
                protocol_clean_name=protocol_clean_name,
                dataset_clean_name=dataset_clean_name,
                task_clean_name=task_clean_name,
                batch_size=batch_size,
            )
            batch_plots._save_figure(tradeoff_fig, image_panels_dir / tradeoff_base_name)
            tradeoff_src_pdf = image_panels_dir / f"{tradeoff_base_name}.pdf"
            tradeoff_dst_pdf = pdf_panels_dir / f"{tradeoff_base_name}.pdf"
            if tradeoff_src_pdf.exists():
                tradeoff_src_pdf.replace(tradeoff_dst_pdf)
            print(image_panels_dir / f"{tradeoff_base_name}.png")
            print(tradeoff_dst_pdf)
            batch_endpoint_rows.to_csv(
                data_dir / f"{tradeoff_base_name}_data.csv",
                index=False,
            )

        dataset_attack_rows = attack_rows[attack_rows["dataset_name"] == dataset_name].copy()
        dataset_utility_rows = utility_rows[utility_rows["dataset_name"] == dataset_name].copy()
        if dataset_attack_rows.empty or dataset_utility_rows.empty:
            continue
        utility_metric = str(dataset_utility_rows["utility_metric"].iloc[0])
        utility_label = clean_name(utility_metric, FL_METRIC_CLEAN_NAMES)
        utility_ylim = fl_metric_limits(utility_metric, dataset_utility_rows["utility_value"])
        for batch_size in sorted(dataset_attack_rows["batch_size"].astype(int).unique().tolist()):
            batch_attack_rows = dataset_attack_rows[dataset_attack_rows["batch_size"] == batch_size].copy()
            batch_utility_rows = dataset_utility_rows[dataset_utility_rows["batch_size"] == batch_size].copy()
            if batch_attack_rows.empty or batch_utility_rows.empty:
                continue

            attack_base_name = (
                f"fedsgd_torch_modules_config_attack_grid__dataset_{dataset_suffix}__batch_{batch_size}"
            )
            attack_fig = _plot_batch_split_metric_grid(
                batch_attack_rows,
                protocol_clean_name=protocol_clean_name,
                dataset_clean_name=dataset_clean_name,
                task_clean_name=task_clean_name,
                batch_size=batch_size,
                value_col="tableak_acc",
                ci_col="tableak_acc_ci95",
                y_label=batch_plots.ATTACK_METRIC_LABEL,
                ylim=(0.0, batch_plots.BOUNDED_METRIC_YMAX),
                title_metric_name=batch_plots.ATTACK_METRIC_LABEL,
                tighten_figure_ylim=False,
                baseline_specs=[
                    ("prior_tableak_acc", "--"),
                    ("random_tableak_acc", ":"),
                ],
                top_rect=0.96,
            )
            batch_plots._save_figure(attack_fig, image_panels_dir / attack_base_name)
            attack_src_pdf = image_panels_dir / f"{attack_base_name}.pdf"
            attack_dst_pdf = pdf_panels_dir / f"{attack_base_name}.pdf"
            if attack_src_pdf.exists():
                attack_src_pdf.replace(attack_dst_pdf)
            print(image_panels_dir / f"{attack_base_name}.png")
            print(attack_dst_pdf)
            batch_attack_rows.to_csv(data_dir / f"{attack_base_name}_data.csv", index=False)

            num_base_name = (
                f"fedsgd_torch_modules_config_num_grid__dataset_{dataset_suffix}__batch_{batch_size}"
            )
            num_fig = _plot_batch_split_metric_grid(
                batch_attack_rows,
                protocol_clean_name=protocol_clean_name,
                dataset_clean_name=dataset_clean_name,
                task_clean_name=task_clean_name,
                batch_size=batch_size,
                value_col="num_acc",
                ci_col="num_acc_ci95",
                y_label="Numerical Accuracy",
                ylim=(0.0, batch_plots.BOUNDED_METRIC_YMAX),
                title_metric_name="Numerical Accuracy",
                tighten_figure_ylim=False,
                baseline_specs=[
                    ("prior_num_acc", "--"),
                    ("random_num_acc", ":"),
                ],
                top_rect=0.96,
            )
            batch_plots._save_figure(num_fig, image_panels_dir / num_base_name)
            num_src_pdf = image_panels_dir / f"{num_base_name}.pdf"
            num_dst_pdf = pdf_panels_dir / f"{num_base_name}.pdf"
            if num_src_pdf.exists():
                num_src_pdf.replace(num_dst_pdf)
            print(image_panels_dir / f"{num_base_name}.png")
            print(num_dst_pdf)
            batch_attack_rows.to_csv(data_dir / f"{num_base_name}_data.csv", index=False)

            cat_base_name = (
                f"fedsgd_torch_modules_config_cat_grid__dataset_{dataset_suffix}__batch_{batch_size}"
            )
            cat_fig = _plot_batch_split_metric_grid(
                batch_attack_rows,
                protocol_clean_name=protocol_clean_name,
                dataset_clean_name=dataset_clean_name,
                task_clean_name=task_clean_name,
                batch_size=batch_size,
                value_col="cat_acc",
                ci_col="cat_acc_ci95",
                y_label="Categorical Accuracy",
                ylim=(0.0, batch_plots.BOUNDED_METRIC_YMAX),
                title_metric_name="Categorical Accuracy",
                tighten_figure_ylim=False,
                baseline_specs=[
                    ("prior_cat_acc", "--"),
                    ("random_cat_acc", ":"),
                ],
                top_rect=0.96,
            )
            batch_plots._save_figure(cat_fig, image_panels_dir / cat_base_name)
            cat_src_pdf = image_panels_dir / f"{cat_base_name}.pdf"
            cat_dst_pdf = pdf_panels_dir / f"{cat_base_name}.pdf"
            if cat_src_pdf.exists():
                cat_src_pdf.replace(cat_dst_pdf)
            print(image_panels_dir / f"{cat_base_name}.png")
            print(cat_dst_pdf)
            batch_attack_rows.to_csv(data_dir / f"{cat_base_name}_data.csv", index=False)

            utility_base_name = (
                f"fedsgd_torch_modules_config_utility_grid__dataset_{dataset_suffix}__batch_{batch_size}"
            )
            utility_fig = _plot_batch_split_metric_grid(
                batch_utility_rows,
                protocol_clean_name=protocol_clean_name,
                dataset_clean_name=dataset_clean_name,
                task_clean_name=task_clean_name,
                batch_size=batch_size,
                value_col="utility_value",
                ci_col="utility_ci95",
                y_label=utility_label,
                ylim=utility_ylim,
                title_metric_name=utility_label,
                tighten_figure_ylim=True,
            )
            batch_plots._save_figure(utility_fig, image_panels_dir / utility_base_name)
            utility_src_pdf = image_panels_dir / f"{utility_base_name}.pdf"
            utility_dst_pdf = pdf_panels_dir / f"{utility_base_name}.pdf"
            if utility_src_pdf.exists():
                utility_src_pdf.replace(utility_dst_pdf)
            print(image_panels_dir / f"{utility_base_name}.png")
            print(utility_dst_pdf)
            batch_utility_rows.to_csv(data_dir / f"{utility_base_name}_data.csv", index=False)


if __name__ == "__main__":
    main()
