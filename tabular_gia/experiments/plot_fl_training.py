from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_helper import (
    apply_axis_finish,
    DEFAULT_LEGEND_FRAME_ALPHA,
    DEFAULT_BOUNDED_METRIC_YMAX,
    FL_METRIC_CLEAN_NAMES,
    PLOT_LINEWIDTH,
    PLOT_LINE_ZORDER,
    PLOT_BAND_ZORDER,
    PLOT_MARKERSIZE,
    X_AXIS_CLEAN_NAMES,
    plot_line_with_band,
    clean_name,
    fl_metric_limits,
    ordered_plot_groups,
    plot_group_styles,
    set_plot_paper_style,
    style_legend,
)


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
    linewidth: float = PLOT_LINEWIDTH,
    markersize: float = PLOT_MARKERSIZE,
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
        plot_line_with_band(
            ax,
            x=x,
            y=y,
            ci=ci,
            color=color,
            marker=marker,
            linewidth=linewidth,
            markersize=markersize,
            alpha=alpha,
            label=str(group) if show_legend_labels else None,
            bounded=bounded,
        )
    apply_axis_finish(
        ax,
        title=title,
        xlabel=resolved_xlabel,
        ylabel=resolved_ylabel,
        ylim=resolved_ylim,
    )


def plot_fl_metric_vs_exposure(plot_rows: pd.DataFrame, title: str) -> plt.Figure:
    set_plot_paper_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    plot_fl_metric_vs_exposure_on_axes(ax, plot_rows, title=title)
    style_legend(ax.legend(title="Client Batch Size", ncol=2, frameon=True))
    return fig
