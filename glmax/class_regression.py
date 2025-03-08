#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base class for regression and moderation analysis and visualization.

@author: E. N. Aslinger
"""


import os
import statsmodels.api as sm
# import statsmodels.formula.api as smf
import seaborn as sns
import pandas as pd
import glmax

COLOR_PALETTE = "tab20"
COLOR_MAP = "coolwarm"


class Regression(object):
    """A class for regression and moderation."""

    def __init__(self, file_path=None, model=None, **kwargs):
        """
        Initialize Regression class object.

        Args:
            file_path (PathLike or DataFrame, optional): Path or object
                containing data (if present). If None, the data can be
                simulated later.
            model (str or dict, optional): Either a `statsmodels`-style
                formula string or a dictionary containing the outcome
                variable (keyed by 'y'), predictor variables
                (keyed by 'x'), and interaction terms (keyed by 'ixs').
            distributions (dict, optional): A list of names of
                distributions for non-Gaussian variables.
                Variables not specified here will be assumed Gaussian.
                If data are provided (not to be simulated later), this
                argument is only needed if the outcome variable is
                non-Gaussian and you want to (a) run a model
                with a different family/link and/or (b) compute and
                plot diagnostics related to violations of
                normal regression assumptions. You also only need to
                specify the outcome variable such a case.
                Use `statsmodels` conventions:
                https://www.statsmodels.org/stable/glm.html.
                Append "_zinfl" to the distribution name to specify
                a zero-inflated distribution
                (e.g., "poisson_zinfl" or "nbinom_zinfl").
        """
        # Let Property Setters Run
        self.data = file_path if isinstance(
            file_path, pd.DataFrame) else pd.read_csv(file_path) if (
                os.path.splitext(file_path) == ".csv") else pd.read_excel(
                    file_path)  # read data
        self.model = model

    @property
    def data(self):
        """Get data."""
        return self._data

    @data.setter
    def data(self, value) -> None:
        """Set data (placeholder property setter)."""
        self._data = value

    @property
    def model(self):
        """Get model."""
        return self._model

    @model.setter
    def model(self, value) -> None:
        """Set model."""
        if not isinstance(value, (dict, str)):
            raise TypeError("`model` must be a dictionary or string.")
        form = value if isinstance(value, str) else glmax.tl.create_formula(
            value["y"], value["x"],
            interactions=value["ixs"] if "ixs" in value else None)  # formula
        mod = glmax.tl.parse_formula(form)  # predictor/outcome lists
        self._model = {"x": mod["x"], "y": mod["y"],
                       "ixs": mod["ixs"], "formula": form}

    def describe(self, groups=None, fill=True, kind_dist="kde",
                 figsize=None, sharey=False, sharex=False, **kwargs):
        """Tabulate and plot descriptives."""
        tabs, figs = {}, {}
        if figsize is None:
            figsize = (20, 20)
        if isinstance(groups, str):
            groups = [groups]
        diag_kws = kwargs.pop("diag_kws", {} if kind_dist == "kde" else {
            "cut": 0, "fill": fill})
        grid_kws = {"diag_sharey": sharey, **kwargs.pop("grid_kws", {})}

        tabs["numeric"] = self.data.describe()  # numeric descriptives
        cat_vars = [i for i in [self.model["y"]] + self.model[
            "x"] if i not in tabs["numeric"]]  # categorical variables
        if len(cat_vars) > 0:
            tabs["categorical"] = self.data[cat_vars].value_counts()
        if groups is not None:
            tabs[f"numeric_{'.'.join(groups)}"] = self.data.groupby(
                groups).describe()  # numeric descriptives by all groups
            tabs["groups"] = self.data[groups].value_counts()
        iters = [None] if groups is None else [
            None] + groups if None not in groups else groups
        for x in iters:  # iterate grouping variables
            if x is not None:
                tabs[f"numeric_{x}"] = self.data.groupby(
                    x).describe()  # numeric descriptives by group x
                if len(cat_vars) > 0:
                    tabs[f"categorical_{x}"] = self.data.groupby(
                        x)[cat_vars].value_counts()
            vvv = [i for i in [self.model["y"]] + self.model["x"] if i != x]
            figs["pair" if x is None else f"pair_{x}"] = sns.pairplot(
                self.data[[self.model["y"]] + self.model["x"]],
                diag_kind=kind_dist, diag_kws=diag_kws,
                hue=x, grid_kws=grid_kws, height=figsize[1] / 5,
                aspect=figsize[0] / figsize[1])  # pairplot with hue
        figs["dist"] = glmax.pl.plot_distributions(
            self.data, variables=vvv, kind="kde", hues=groups,
            palette="tab10", fill=True,
            common_norm=False, figsize=figsize)  # distributions
        return figs

    def correlate(self, intercorr=True, p_stars=True, **kwargs):
        """
        Calculate correlation between each predictor and outcome
        (and among predictors if `intercorr` is True).
        """
        out = glmax.ax.calculate_correlation(
            self.data, variables=[self.model["y"]] + self.model["x"],
            p_stars=p_stars, **kwargs)
        return out

    def run(self, formula=None, family=None, link=None):
        """Run the regression model.

        Args:
            formula (str, optional): A `statsmodels`-style
                formula string. If None, the model will be run using
                the model specified in the class constructor (if any).
                If a model was specified in the constructor,
                this argument will override it.
        """
        if formula is not None:
            self.model = formula
        if self.model is None:
            raise ValueError("No model specified.")
        if family is None and link is None:
            sm.OLS()


