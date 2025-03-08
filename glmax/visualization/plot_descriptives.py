#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot descriptives.

@author: E. N. Aslinger
"""

import seaborn as sns
import matplotlib.pyplot as plt


def plot_distributions(data, variables=None, kind="kde",
                       hues=None, palette="tab10",
                       fill=True, common_norm=False, figsize=None, **kwargs):
    """Plot distributions of variables, color-coded if desired."""
    if figsize is None:
        figsize = (20, 20)
    if isinstance(hues, str) or hues is None:
        hues = [hues]
    if variables is None:
        variables = list(data.columns)
    fig, axs = plt.subplots(len(variables), len(hues), figsize=figsize,
                            sharex=False, sharey=False, squeeze=False)
    for j, y in enumerate(hues):
        for i, v in enumerate(variables):
            if kind == "kde":
                sns.kdeplot(data, x=v, ax=axs[i, j], hue=y, palette=palette,
                            common_norm=common_norm, cut=0,
                            fill=fill, **kwargs)
            else:
                sns.histplot(data, x=v, ax=axs[i, j], hue=y,
                             palette=palette, **kwargs)
            axs[i, j].set_title(f"{v}" + str(f" by {y}" if y else ""),
                                fontweight="bold")
        # if j != len(vsd) - 1:
        #     axs[i, j].legend_.set_visible(False)
    fig.tight_layout()
    return fig
