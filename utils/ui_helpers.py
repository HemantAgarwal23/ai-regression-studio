"""
UI Helpers

Small pure functions that shape data for Streamlit widgets. They live here
rather than in app.py so they can be unit tested without starting the app.
"""

import re

import numpy as np
import pandas as pd

# Column names that commonly denote a regression target. Matched against
# whole name tokens, never as substrings: a substring test lets the
# single-letter 'y' match any column containing that letter, so a column
# like "age_years" would outrank an actual "price" column.
TARGET_KEYWORDS = ('price', 'cost', 'value', 'amount', 'target', 'y',
                   'label', 'outcome', 'sales', 'revenue', 'score')

# Split on underscores, hyphens, spaces, dots and camelCase boundaries.
_TOKEN_SPLIT = re.compile(r'[^0-9a-zA-Z]+|(?<=[a-z0-9])(?=[A-Z])')


def safe_number_input_bounds(series):
    """
    Derive usable min/max/default/step for a numeric prediction input.

    A constant column has a standard deviation of zero and an all-NaN column has
    none at all; both produce a degenerate range that ``st.number_input``
    rejects outright. This widens such columns to a workable band instead of
    letting the Prediction Lab crash.

    Parameters:
    -----------
    series : pandas.Series
        The dataset column the widget is for. Non-numeric entries are ignored.

    Returns:
    --------
    tuple
        (min_value, max_value, default_value, step), all finite, with
        min_value < max_value and min_value <= default_value <= max_value.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()

    if clean.empty:
        return -1.0, 1.0, 0.0, 0.1

    col_min = float(clean.min())
    col_max = float(clean.max())
    col_mean = float(clean.mean())
    col_std = float(clean.std())

    # std is NaN for a single row and 0.0 for a constant column.
    if not np.isfinite(col_std) or col_std <= 0:
        spread = abs(col_mean) * 0.5 if col_mean else 1.0
    else:
        spread = 2 * col_std

    low = min(col_min, col_mean) - spread
    high = max(col_max, col_mean) + spread
    step = max((high - low) / 100, 1e-6)

    return low, high, col_mean, step


def tokenize_column_name(name):
    """
    Split a column name into lowercase word tokens.

    ``"sale_price"`` -> ``['sale', 'price']``, ``"SalePrice"`` -> ``['sale', 'price']``.

    Parameters:
    -----------
    name : str
        Column name.

    Returns:
    --------
    list of str
        Lowercased tokens, empty strings removed.
    """
    return [part.lower() for part in _TOKEN_SPLIT.split(str(name)) if part]


def suggest_target_column(numeric_cols):
    """
    Guess which column the user most likely wants to predict.

    Matches keywords against whole name tokens rather than as substrings, and
    prefers the keyword earliest in ``TARGET_KEYWORDS`` so a column named
    "price" beats one named "score" when both are present.

    Parameters:
    -----------
    numeric_cols : list
        Candidate numeric column names, in dataframe order.

    Returns:
    --------
    str or None
        The suggested column, or None when nothing matches.
    """
    best = None
    best_rank = len(TARGET_KEYWORDS)

    for col in numeric_cols:
        tokens = set(tokenize_column_name(col))
        for rank, keyword in enumerate(TARGET_KEYWORDS):
            if keyword in tokens and rank < best_rank:
                best, best_rank = col, rank
                break

    return best
