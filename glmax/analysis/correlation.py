#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform correlation analysis and visualization.

@author: E. N. Aslinger
"""

from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def calculate_correlation(data, variables=None, p_stars=True, **kwargs):
    """Calculate and plot correlation matrix."""
    if variables is None:
        variables = list(data.columns)
    corr = data[variables].corr()
    p_values = pd.concat([pd.Series([
        stats.pearsonr(data.dropna(subset=[i, j])[i],
                       data.dropna(subset=[i, j])[j], **kwargs)[1]
        for i in variables], index=variables) for j in variables],
                         keys=variables).unstack().loc[
                             corr.index, corr.columns]
    sig_stars = p_values.map(
        lambda x: "***" if x < 0.001 else "**" if x < 0.01 else "*" if (
            x < 0.05) else "")
    corr_full = corr.stack().to_frame("r").join(p_values.stack().to_frame(
        "p").join(p_values.stack().to_frame("p_adjusted")))
    corr_stars = corr_full.join(sig_stars.stack().to_frame("stars")).apply(
        lambda x: f"{round(x['r'], 2)}{x['stars']}", axis=1).unstack().loc[
            corr.index, corr.columns]
    plt.figure(figsize=(12, 10))  # initiate to plot the heatmap
    fig = sns.heatmap(corr, annot=corr_stars if p_stars is True else None,
                      fmt="", cmap="coolwarm",
                      center=0, vmin=-1, vmax=1)  # heatmap (annotated)
    print(corr_full.round(3))
    return corr_full, corr_stars, fig
