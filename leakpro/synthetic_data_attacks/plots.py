"""Plotting functions for different risks."""
from typing import List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from leakpro.synthetic_data_attacks.inference_utils import InferenceResults
from leakpro.synthetic_data_attacks.linkability_utils import LinkabilityResults
from leakpro.synthetic_data_attacks.singling_out_utils import SinglingOutResults

# Set global plot properties
colors = ["b", "g", "orange"]
alpha = 1
bar_width = 1/3 * 0.95
conf_bar_width = 0.03
# Set confidence level for plots
conf_level = 0.95
tail_conf_level = (1-conf_level)/2

def plot_save_high_res() -> None:
    """Function to set plot (and save) figure dpi."""
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300

def get_figure_axes(*, two_axes_flag: bool = False, fig_title: str = "") -> Axes:
    """Function to get plot figure axes."""
    # Set extra
    extra = int(two_axes_flag)
    # Get fig, axs
    fig, axs = plt.subplots(1+extra, 1, figsize=(11, 4+extra*3))
    # Set fig title
    if fig_title:
        fig.suptitle(fig_title, fontsize=11)
    return axs

def set_labels_and_title(*, ax: Axes, xlabel: str, ylabel: str, title: str) -> None:
    """Function to set axes labels and title."""
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)

def set_ticks(*, ax: Axes, xlabels: List[str] = None) -> None:
    """Function to set axes ticks."""
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xticks(list(range(len(xlabels))))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)

def set_legend(*, ax: Axes) -> None:
    """Function to set axes legend."""
    legend_labels = ["Main attack", "Naive attack", "Residual risk"]
    legend_handles = [mpatches.Patch(color=c, label=ll) for c, ll in zip(colors, legend_labels)]
    ax.legend(handles=legend_handles, fontsize=10)

def iterate_values_plot_bar_charts(*,
    ax: Axes,
    res: np.array,
    set_values: set,
    values: List,
    max_value_flag: bool = False
) -> None:
    """Function to iterate through set_values and plot them in bar charts."""
    # Iterate through set of values
    for x_idx, value in enumerate(set_values):
        # Get idx of data
        idx = np.where(values==value)[0]
        # Set data
        data = res[idx, 4:7]
        if data.shape[0]>0:
            # Iterate through risks
            for r in range(3):
                data_r = data[:, r]
                median = np.median(data_r)
                up = np.quantile(data_r, 1-tail_conf_level)
                down = np.quantile(data_r, tail_conf_level)
                # Plot bar charts
                ax.bar(x_idx+r*bar_width, median, alpha=alpha, width=bar_width, color=colors[r], align="center")
                ax.bar(x_idx+r*bar_width, up-down, alpha=alpha, width=conf_bar_width, color="black", align="center", bottom=down)
    # Add extra space between the plot box and the highest value
    if max_value_flag:
        max_value = res[:, 4:7].max()
        ax.set_ylim(0, max_value * 1.05)

def plot_linkability(*, link_res: LinkabilityResults, high_res_flag: bool = True) -> None:
    """Function to plot linkability results from given res.

    Note: function is not tested and is used in examples.
    """
    # Get res and aux_cols_nr
    res = np.array(link_res.res)
    set_nr_aux_cols = np.unique(res[:,-1].astype(int))
    # High res flag
    if high_res_flag:
        plot_save_high_res()
    # Set up the figure and get axes
    ax = get_figure_axes()
    # Iterate through nr of columns and plot bar charts
    iterate_values_plot_bar_charts(ax=ax, res=res, set_values=set_nr_aux_cols, values=res[:, -1])
    # Adding labels and title
    set_labels_and_title(
        ax = ax,
        xlabel = "Nr aux cols",
        ylabel = "Risk",
        title = f"Linkability risk {conf_level} confidence, total attacks: {int(res[:,0].sum())}"
    )
    # Adding ticks
    set_ticks(ax=ax, xlabels=set_nr_aux_cols)
    # Adding legend
    set_legend(ax=ax)
    # Show plot
    plt.show()

def plot_ir_worst_case(*, inf_res: InferenceResults, high_res_flag: bool = True) -> None:
    """Function to plot inference results worst case given results.

    Note: function is not tested and is used in examples.
    """
    #Set res, secrets and set_secrets
    res = np.array(inf_res.res)
    secrets = np.array(inf_res.secrets)
    set_secrets = sorted(set(secrets))
    # High res flag
    if high_res_flag:
        plot_save_high_res()
    # Set up the figure and get axes
    ax = get_figure_axes()
    # Iterate through secrets and plot bar charts
    iterate_values_plot_bar_charts(
        ax = ax,
        res = res,
        set_values = set_secrets,
        values = secrets,
        max_value_flag = True
    )
    # Adding labels and title
    set_labels_and_title(
        ax = ax,
        xlabel = "Secret col",
        ylabel = "Risk",
        title = f"Inference risk, worst case scenario, total attacks: {int(res[:,0].sum())}"
    )
    # Adding ticks
    set_ticks(ax=ax, xlabels=set_secrets)
    # Adding legend
    set_legend(ax=ax)
    # Show plot
    plt.show()

def plot_ir_base_case(*, inf_res: InferenceResults, high_res_flag: bool = True) -> None:
    """Function to plot inference results base case given results.

    Note: function is not tested and is used in examples.
    """
    #Set res, secrets, set_secrets and set_nr_aux_cols
    res = np.array(inf_res.res)
    secrets = np.array(inf_res.secrets)
    set_secrets = sorted(set(secrets))
    set_nr_aux_cols = np.unique(res[:,-1].astype(int))
    # High res flag
    if high_res_flag:
        plot_save_high_res()
    # Set up the figure and get axes
    fig_title = f"Inference risk, base case scenario, {conf_level} confidence, total attacks: {int(res[:,0].sum())}"
    axs = get_figure_axes(two_axes_flag=True, fig_title=fig_title)
    # Set plot variables
    titles = ["Risk per column", "Risk per Nr aux cols"]
    xlabels = ["Secret col", "Nr aux cols"]
    sets_values = [set_secrets, set_nr_aux_cols]
    valueses = [secrets, res[:,-1]]
    assert len(axs) == len(titles)
    assert len(axs) == len(xlabels)
    assert len(axs) == len(sets_values)
    assert len(axs) == len(valueses)
    #Plotting
    for ax, title, xlabel, set_values, values in zip(axs, titles, xlabels, sets_values, valueses):
        set_labels_and_title(
            ax = ax,
            xlabel = xlabel,
            ylabel = "Risk",
            title = title
        )
        # Iterate through values and plot bar charts
        iterate_values_plot_bar_charts(ax=ax, res=res, set_values=set_values, values=values)
        # Adding ticks
        set_ticks(ax=ax, xlabels=set_values)
        # Adding legend
        set_legend(ax=ax)
    plt.tight_layout()
    plt.show()

def plot_singling_out(*,
            sin_out_res: SinglingOutResults,
            high_res_flag: bool = True,
            show: bool = True,
            save: bool = False,
            save_name: str = None
    ) -> None:
    """Function to plot singling out given results.

    Note: function is not tested and is used in examples.
    """
    #Set res, n_cols and set_n_cols
    res = np.array(sin_out_res.res)
    n_cols = res[:,-1].astype(int).tolist()
    set_n_cols = np.unique(n_cols)
    # High res flag
    if high_res_flag:
        plot_save_high_res()
    # Set up the figure and get axes
    ax = get_figure_axes()
    # Iterate through values and plot bar charts
    iterate_values_plot_bar_charts(
        ax = ax,
        res = res,
        set_values = set_n_cols,
        values = n_cols,
        max_value_flag = True
    )
    # Adding labels and title
    fig_title = f"Singling out risk total attacks: {int(res[:,0].sum())}"
    if res.shape[0]==1:
        fig_title += f", n_cols={int(res[0,-1])}"
    set_labels_and_title(
        ax = ax,
        xlabel = "n_cols for predicates",
        ylabel = "Risk",
        title = fig_title
    )
    # Adding ticks
    set_ticks(ax=ax, xlabels=set_n_cols)
    # Adding legend
    set_legend(ax=ax)

    if save:
        plt.savefig(fname=f"{save_name}.png", dpi=1000, bbox_inches="tight")
    # Show plot
    if show:
        plt.show()
    else:
        plt.clf()
