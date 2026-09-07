"""
UI Helpers

Small pure functions that shape data for Streamlit widgets. They live here
rather than in app.py so they can be unit tested without starting the app.
"""

import numpy as np
import pandas as pd


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
