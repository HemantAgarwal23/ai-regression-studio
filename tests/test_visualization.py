"""
Tests for the Plotly figure builders.

These check that each builder returns a well-formed figure and survives the
degenerate inputs the app can actually hand it, rather than asserting on
cosmetic styling.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.visualization import (
    create_correlation_heatmap,
    create_prediction_scatter,
    create_residual_plot,
)


def test_create_correlation_heatmap_returns_figure():
    """A numeric frame produces a heatmap with the correlation scale pinned."""
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [2, 4, 6, 8, 10],
        'c': [5, 3, 4, 1, 2],
    })

    fig = create_correlation_heatmap(df, ['a', 'b', 'c'])

    assert isinstance(fig, go.Figure)
    # px.imshow puts the pinned range on the shared coloraxis, not on the trace.
    assert fig.layout.coloraxis.cmin == -1
    assert fig.layout.coloraxis.cmax == 1


def test_create_correlation_heatmap_ignores_non_numeric_columns():
    """A text column must not blank out the whole matrix."""
    df = pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': [4, 3, 2, 1],
        'label': ['w', 'x', 'y', 'z'],
    })

    fig = create_correlation_heatmap(df, ['a', 'b', 'label'])

    assert isinstance(fig, go.Figure)
    assert not np.isnan(np.asarray(fig.data[0].z, dtype=float)).all()


def test_create_residual_plot_centres_on_zero_line():
    """Residuals are actual minus predicted, with a zero reference line."""
    y_test = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    fig = create_residual_plot(y_test, y_pred)

    assert isinstance(fig, go.Figure)
    assert np.allclose(fig.data[0].y, y_test - y_pred)
    assert any(shape.type == 'line' for shape in fig.layout.shapes)


def test_create_residual_plot_accepts_pandas_series():
    """The app passes a Series for y_test, not an ndarray."""
    fig = create_residual_plot(pd.Series([1.0, 2.0]), np.array([1.5, 1.5]))

    assert np.allclose(fig.data[0].y, [-0.5, 0.5])


def test_create_prediction_scatter_adds_reference_line():
    """The y = x line spans the combined range of both series."""
    y_test = np.array([1.0, 5.0, 9.0])
    y_pred = np.array([2.0, 4.0, 11.0])

    fig = create_prediction_scatter(y_test, y_pred)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2

    line = fig.data[1]
    assert line.name == 'Perfect Prediction'
    assert line.x == (1.0, 11.0)
    assert line.y == (1.0, 11.0)


def test_create_prediction_scatter_handles_empty_input():
    """Empty arrays yield a placeholder figure rather than a plotly error."""
    fig = create_prediction_scatter(np.array([]), np.array([]))

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0
    assert fig.layout.annotations[0].text == 'No data to plot'


def test_create_residual_plot_handles_empty_input():
    """The residual plot takes the same empty-input path."""
    fig = create_residual_plot(np.array([]), np.array([]))

    assert isinstance(fig, go.Figure)
    assert fig.layout.annotations[0].text == 'No data to plot'
