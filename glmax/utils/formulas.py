#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for parsing and creating GLM formulas.

@author: E. N. Aslinger
"""

from typing import List, Union
import functools
from itertools import combinations
import warnings
import numpy as np


def create_formula(outcome: str, predictors: Union[List[str], str],
                   interactions=None):
    """Create a formula from specification of variable roles.

    Args:
        outcome (str): The outcome variable.
        predictors (str or list): A string (simple regression) or list
            (multiple regression) of predictor variables.
        interactions (optional, list or None):
            A list of interactions of the format ('x1:x2' or
            'x1*x2*x3', for instance) to include (moderation).
            The default is None.

    Returns:
        str: A formula string.
    """
    if isinstance(predictors, str):
        predictors = [predictors]
    form = f"{outcome} ~ {' + '.join(predictors)}"
    if interactions is not None:
        if not isinstance(interactions, (list, str)):
            raise TypeError("`interactions` must be a list or string.")
        if isinstance(interactions, str):
            interactions = [interactions]
        form += f" + {' + '.join(interactions)}"
    return form


def parse_formula(formula: str):
    """Parse a formula string into its components.

    Args:
        formula (str): A formula string.

    Returns:
        dict: A dictionary containing the outcome variable, predictor
    """
    if not isinstance(formula, str):
        raise TypeError("`formula` must be a string.")
    form = formula.split("~")
    if len(form) != 2:
        raise ValueError("Formula must be of the form 'y ~ x1 + x2 + ...'.")
    outcome = form[0].strip()
    predictors = form[1].strip().split("+")
    preds = [p.strip() for p in predictors if ":" not in p and "*" not in p]

    # Interaction Terms
    if "*" in formula or ":" in formula:  # if any interaction terms
        ixs = [p for p in predictors if ":" in p or "*" in p]
        ixs = [str(":" if ":" in p else "*").join([i.strip() for i in p.split(
            str(":" if ":" in p else "*"))]) for p in ixs]  # strip whitespace
        if any((":" in p and "*" in p for p in ixs)):
            raise ValueError("Cannot include both ':' and '*' in the "
                             "same interaction term.")

        # Check Interaction Term Validity & Reformulate "*" Multi-Interactions
        ix_star = [[i.strip() for i in p.split("*")] for p in ixs if "*" in p]
        ix_col = [[i.strip() for i in p.split(":")] for p in ixs if ":" in p]
        ix_star_reform = functools.reduce(lambda i, j: i + j, [
            [list(p) for p in combinations(x, 2)]
            for x in ix_star])  # pair-wise combinations of "*" ix terms
        ixs_lists = ix_col + ix_star_reform
        unique_pairs = {frozenset(p) for p in ixs_lists}
        ixs_lists = [list(p) for p in unique_pairs]

        # Check All Interaction Terms Included as Predictors
        ix_preds = np.unique(functools.reduce(lambda i, j: i + j, ixs_lists))
        missing_preds = [p for p in ix_preds if p not in preds]
        if len(missing_preds) > 0:
            warnings.warn(f"Interaction variables {', '.join(missing_preds)}"
                          " added explicitly as predictors.")
            preds += missing_preds
    else:
        ixs_lists = None

    # TODO: Add warning if lower-order interactions not included
    # (e.g., x1:x2:x3 specified, but x1:x2 not specified)

    return {"y": outcome, "x": preds, "ixs": ixs_lists}


def create_formula_sem(model):
    """
    Create an SEM formula from a dictionary of
        outcomes and predictor sets.

    Args:
        model (dict): A dictionary containing the model specification.

    Returns:
        str: A formula string.
    """
    raise NotImplementedError("SEM formula creation is not yet implemented.")
