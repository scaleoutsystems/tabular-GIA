from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_helper import (
    DEFAULT_BOUNDED_METRIC_YMAX,
    FL_METRIC_CLEAN_NAMES,
    X_AXIS_CLEAN_NAMES,
    clean_name,
    fl_metric_limits,
    ordered_plot_groups,
    plot_group_styles,
    set_plot_paper_style,
)


def _clip_ci_upper(y: np.ndarray, ci: np.ndarray, upper_bound: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    return y - ci, np.minimum(y + ci, upper_bound)


def _ci_bounds(y: np.ndarray, ci: np.ndarray, *, bounded: bool) -> tuple[np.ndarray, np.ndarray]:
    return _clip_ci_upper(y, ci) if bounded else (y - ci, y + ci)


def plot_fl_metric_vs_exposure_on_axes(
    ax: plt.Axes,
    plot_rows: pd.DataFrame,
    *,
    title: str = "",
    x_col: str = "exp_min",
    group_col: str = "batch_size",
    value_col: str = "utility_mean",
    ci_col: str = "utility_ci95",
    utility_metric_col: str = "utility_metric",
    xlabel: str | None = None,
    ylabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    linewidth: float = 2.0,
    markersize: float = 4.8,
    alpha: float = 0.16,
    styles: dict[int | str, tuple[Any, str]] | None = None,
    show_legend_labels: bool = True,
) -> None:
    utility_metric = str(plot_rows[utility_metric_col].iloc[0])
    resolved_ylim = ylim if ylim is not None else fl_metric_limits(utility_metric, plot_rows[value_col])
    bounded = resolved_ylim == (0.0, DEFAULT_BOUNDED_METRIC_YMAX)
    resolved_xlabel = xlabel if xlabel is not None else clean_name(x_col, X_AXIS_CLEAN_NAMES)
    resolved_ylabel = ylabel if ylabel is not None else clean_name(utility_metric, FL_METRIC_CLEAN_NAMES)
    typed_groups = plot_rows[group_col].dropna().unique().tolist()
    style_map = styles if styles is not None else plot_group_styles(typed_groups)

    for group in ordered_plot_groups(typed_groups):
        color, marker = style_map[group]
        sub = plot_rows[plot_rows[group_col] == group].sort_values(x_col)
        x = sub[x_col].to_numpy(dtype=float)
        y = sub[value_col].to_numpy(dtype=float)
        ci = sub[ci_col].fillna(0.0).to_numpy(dtype=float)
        lower, upper = _ci_bounds(y, ci, bounded=bounded)
        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            linewidth=linewidth,
            markersize=markersize,
            label=str(group) if show_legend_labels else None,
        )
        ax.fill_between(x, lower, upper, color=color, alpha=alpha, linewidth=0)

    ax.set_title(title)
    ax.set_xlabel(resolved_xlabel)
    ax.set_ylabel(resolved_ylabel)
    if resolved_ylim is not None:
        ax.set_ylim(*resolved_ylim)
    ax.grid(True, which="major")


def plot_fl_metric_vs_exposure(plot_rows: pd.DataFrame, title: str) -> plt.Figure:
    set_plot_paper_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    plot_fl_metric_vs_exposure_on_axes(ax, plot_rows, title=title)
    legend = ax.legend(title="Client Batch Size", ncol=2, frameon=True)
    legend.get_frame().set_alpha(0.92)
    return fig
