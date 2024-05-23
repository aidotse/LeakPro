"""Plotting functions for different risks."""
from typing import List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from leakpro.synthetic_data_attacks.inference_utils import InferenceResults
from leakpro.synthetic_data_attacks.linkability_utils import LinkabilityFullResults

# Set global plot properties
colors = ["b", "g", "orange"]
alpha = 1
bar_width = 1/3 * 0.95
bar_width2 = 0.03

def set_figure() -> None:
    """Function to set plot figure."""
    plt.figure(figsize=(11, 5))

def set_labels_and_title(*, xlabel: str, ylabel: str, title: str) -> None:
    """Function to set plot labels and title."""
    plt.xlabel(xlabel, fontsize=9)
    plt.ylabel(ylabel, fontsize=9)
    plt.title(title, fontsize=10)

def set_ticks(xticks: List[int], xlabels: List[str] = None) -> None:
    """Function to set plot ticks."""
    plt.yticks(fontsize=7)
    if xlabels is None:
        xlabels = xticks
    plt.xticks(xticks, xlabels, rotation=45, ha="right", fontsize=7)

def set_legend() -> None:
    """Function to set plot legend."""
    legend_labels = ["Main attack", "Naive attack", "Residual risk"]
    legend_handles = [mpatches.Patch(color=c, label=ll) for c, ll in zip(colors, legend_labels)]
    plt.legend(handles=legend_handles, fontsize=10)

def plot_linkability(*, full_link_res: LinkabilityFullResults) -> None:
    """Function to plot linkability results from given res.

    Note: function is not tested and is used in examples.
    """
    # Get res and aux_cols_nr
    res = np.array(full_link_res.res)
    aux_cols_nr = np.unique(res[:, -1])
    # Set confidence level
    conf_level = 0.95
    tail_conf_level = (1-conf_level)/2
    # Set up the figure and axes
    set_figure()
    # Plotting the bar charts
    for n_col in aux_cols_nr:
        idx = np.where(res[:,-1]==n_col)[0]
        res_ = res[idx, :]
        if res_.shape[0]>0:
            for i in range(3):
                data = res_[:, i+4]
                median = np.median(data)
                up = np.quantile(data, 1-tail_conf_level)
                down = np.quantile(data, tail_conf_level)
                plt.bar(n_col+i*bar_width, median, alpha=alpha, width=bar_width, color=colors[i], align="center")
                plt.bar(n_col+i*bar_width, up-down, alpha=alpha, width=bar_width2, color="black", align="center", bottom=down)
    # Adding labels and title
    set_labels_and_title(
        xlabel = "Nr. aux cols",
        ylabel = "Risk",
        title = f"Full linkability risk {conf_level} confidence, total attacks: {int(res[:,0].sum())}"
    )
    # Adding ticks
    xticks = [int(i) for i in aux_cols_nr]
    set_ticks(xticks=xticks)
    # Adding legend
    set_legend()
    # Show plot
    plt.show()

def plot_ir_each_against_rest_columns(*, inf_res: InferenceResults) -> None:
    """Function to plot inference results each column against rest of columns.

    Note: function is not tested and is used in examples.
    """
    #Get res
    res = np.array(inf_res.res)
    # Set up the figure and axes
    set_figure()
    # Plotting the bar charts
    for i in range(3):
        for j in range(res.shape[0]):
            plt.bar(j+i*bar_width, res[j, i+4], width=bar_width, color=colors[i], align="center")
    # Adding labels and title
    set_labels_and_title(
        xlabel = "Secret col",
        ylabel = "Risk",
        title = f"Inference risk, each column against rest of columns, total attacks: {int(res[:,0].sum())}"
    )
    # Adding ticks
    xticks = list(range(res.shape[0]))
    xlabels = [inf_res.secrets[i] for i in xticks]
    set_ticks(xticks=xticks, xlabels=xlabels)
    # Adding legend
    set_legend()
    # Show plot
    plt.show()
