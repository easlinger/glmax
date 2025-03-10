#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base class for moderation analysis and visualization.

@author: E. N. Aslinger
"""

import matplotlib.pyplot as plt
from statsmodels.graphics.factorplots import interaction_plot
from glmax import Regression
from glmax.visualization import square_grid


class Moderation(Regression):
    """A class for regression and moderation."""

    def __init__(self, **kwargs):
        """
        Initialize Moderation class object.
        """
        super().__init__(**kwargs)

    def plot_interaction(self, ix_list=None, swap_axes=False,
                         figsize=None, tight_layout=True, **kwargs):
        """Plot categorical interactions."""
        ix_list = self.model["ixs"] if ix_list is None else list([
            ix_list] if isinstance(ix_list[0], str) else ix_list)
        fig, a_x = plt.subplots(*square_grid(len(ix_list)),
                                figsize=figsize, squeeze=False)
        kwargs = {**dict(ms=10, legendloc="right"), **kwargs}
        for u, i in enumerate(ix_list):
            interaction_plot(
                x=self.data[i[1 if swap_axes is True else 0]],
                trace=self.data[i[0 if swap_axes is True else 1]],
                response=self.data[self.model["y"]],
                ax=a_x.flatten()[u], **kwargs)
            a_x.flatten()[u].set_title(" x ".join(i))
        if tight_layout is True:
            fig.tight_layout()
        return fig, a_x
