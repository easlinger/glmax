#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base class for regression and moderation analysis and visualization.

@author: E. N. Aslinger
"""


import os
# import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import NegativeBinomial
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
from statsmodels.discrete.count_model import ZeroInflatedPoisson
from statsmodels.genmod.families import Poisson
from statsmodels.miscmodels.ordinal_model import OrderedModel
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

    def run(self, formula=None, family=None, link=None,
            show=True, inplace=True,
            kws_model=None, kws_diagnostics=False, **kwargs):
        """Run the regression model.

        Args:
            formula (str, optional): A `statsmodels`-style
                formula string. If None, the model will be run using
                the model specified in the class constructor (if any).
                If a model was specified in the constructor,
                this argument will override it. SPECIFYING THIS WILL
                OVERRIDE `self.model`.
            family (str, optional): The name of the family to use
                for non-Gaussian models. If None, the model will be
                run as a Gaussian model. Examples: 'OrderedModel',
                'Poisson', 'NegativeBinomial', 'ZeroInflatedPoisson',
                'ZeroInflatedNegativeBinomialP', etc.
                Not case sensitive.
            link (str, optional): The name of the link function to
                use for non-Gaussian models. If None, the model will
                be run as a Gaussian model.
            show (bool, optional): If True, print the model summary.
            inplace (bool, optional): If True, store the results
                in the `self.results` attribute.
            kws_model (dict, optional): A dictionary of
                keyword arguments to pass to the `statsmodels`
                model construction function.
            kws_diagnostics (bool or dict, optional): If True,
                run diagnostics (checking model assumptions/violations,
                e.g., heteroskedasticity) using the default arguments
                in `glmax.ax.run_regression_diagnostics()`. If False,
                don't run diagnostics. If a dictionary, pass the
                dictionary as keyword arguments to
                `glmax.ax.run_regression_diagnostics()`.
            kwargs (dict, optional): Additional keyword arguments
                to pass to the `statsmodels` model fitting function.
        """
        if formula is not None:  # if new formula specified...
            self.model = formula  # ...set model dictionary attribute
        kws_model = {} if kws_model is None else {**kws_model}
        extras = {}  # will change later for models with more outputs
        if self.model is None:
            raise ValueError("No model specified.")
        if family is None and link is None:  # Gaussian
            model = smf.ols(formula=self.model["formula"],
                            data=self.data).fit(**kwargs)  # fit OLS
            summary = model.summary()
        elif isinstance(family, str) and family.lower(
                ) in glmax.constants.models:  # non-Gaussian
            f_x = glmax.constants.models[family.lower()]
            if "from_formula" in dir(f_x):
                f_x = f_x.from_formula
            model = f_x(formula=self.model["formula"],
                        data=self.data).fit(**kwargs)  # fit non-Gaussian
            summary = model.summary()
            print("\nPseudo-R-Squared: ", model.prsquared, "\n")
            if family in ["binary", "logit"]:
                extras["marginal_effects"] = model.get_margeff().summary()
        else:
            raise NotImplementedError(f"{family} not implemented.")
        if isinstance(kws_diagnostics, dict) or kws_diagnostics is True:
            kws_diagnostics = {**kws_diagnostics} if isinstance(
                kws_diagnostics, dict) else {}
            fig, dis, summ_text = glmax.ax.run_regression_diagnostics(
                model, **kws_diagnostics)  # assumption/violation diagnostics
        if show is True:
            print(summary)
            if len(extras) > 0:
                for k, v in extras.items():
                    print(f"\n\n{k}:\n\n {v}\n")
        if inplace is True:
            self.results = model
        return model, summary, extras
