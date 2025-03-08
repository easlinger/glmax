#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for plotting.

@author: E. N. Aslinger
"""

import math
import numpy as np


def square_grid(num):
    """Return row-column dimensions (approximately a square)."""
    if isinstance(num, (np.ndarray, list, set, tuple, pd.Series)):
        num = len(num)  # if provided actual object, calculate length
    if num == 2:
        rows, cols = 1, 2
    else:
        rows = int(np.sqrt(num))  # number of rows
        cols = rows if rows * 2 == num else math.ceil(num / rows)  # column #
    return rows, cols
