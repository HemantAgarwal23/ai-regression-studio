"""
Visualization Utilities

This module contains functions for creating interactive visualizations using Plotly.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

PLOT_TEMPLATE = "plotly_white"


def _empty_figure(title, x_title, y_title):
    """
    Build a placeholder figure carrying an explanatory annotation.

    ``px.scatter`` raises on empty x and y arrays, so callers that may be handed
    an empty split get this instead of a traceback.
    """
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template=PLOT_TEMPLATE,
        annotations=[{
            'text': 'No data to plot',
            'xref': 'paper',
            'yref': 'paper',
            'x': 0.5,
            'y': 0.5,
            'showarrow': False,
        }],
    )
    return fig


def create_correlation_heatmap(df, columns, title="Feature Correlation Matrix"):
    """
    Create a correlation heatmap for selected columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    columns : list
        List of column names to include in correlation matrix
    title : str, default="Feature Correlation Matrix"
        Title for the plot

    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # numeric_only keeps a stray text column from turning the whole matrix NaN.
    corr_data = df[columns].corr(numeric_only=True)

    fig = px.imshow(
        corr_data,
        title=title,
        template=PLOT_TEMPLATE,
        color_continuous_scale="RdBu",
        # Correlations are bounded, so pinning the scale keeps the midpoint at
        # zero instead of letting it drift with the data.
        zmin=-1,
        zmax=1,
        aspect="auto",
    )

    fig.update_layout(xaxis_title="Features", yaxis_title="Features")

    return fig


def create_residual_plot(y_test, y_pred, title="Residuals Plot"):
    """
    Create a residual plot (residuals vs predicted values).

    Parameters:
    -----------
    y_test : array-like
        Actual target values
    y_pred : array-like
        Predicted target values
    title : str, default="Residuals Plot"
        Title for the plot

    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    actual = np.asarray(y_test, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    if actual.size == 0 or predicted.size == 0:
        return _empty_figure(title, 'Predicted Values', 'Residuals')

    residuals = actual - predicted

    fig = px.scatter(
        x=predicted,
        y=residuals,
        title=title,
        labels={'x': 'Predicted Values', 'y': 'Residuals'},
        template=PLOT_TEMPLATE,
    )

    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Residual Line")

    return fig


def create_prediction_scatter(y_test, y_pred, title="Actual vs Predicted Values"):
    """
    Create a scatter plot of actual vs predicted values.

    Parameters:
    -----------
    y_test : array-like
        Actual target values
    y_pred : array-like
        Predicted target values
    title : str, default="Actual vs Predicted Values"
        Title for the plot

    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    actual = np.asarray(y_test, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    if actual.size == 0 or predicted.size == 0:
        return _empty_figure(title, 'Actual Values', 'Predicted Values')

    fig = px.scatter(
        x=actual,
        y=predicted,
        title=title,
        labels={'x': 'Actual Values', 'y': 'Predicted Values'},
        template=PLOT_TEMPLATE,
    )

    # The y = x reference line, spanning the combined range of both series.
    min_val = float(min(actual.min(), predicted.min()))
    max_val = float(max(actual.max(), predicted.max()))

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line={'dash': 'dash', 'color': 'red'},
    ))

    return fig


def create_shap_waterfall(feature_names, shap_values, base_value, feature_values=None,
                          max_features=10, title="Why this prediction?"):
    """
    Build a waterfall of SHAP contributions for a single prediction.

    Drawn natively in Plotly rather than via ``shap``'s own plotting, which
    renders through matplotlib and would clash with the interactive Plotly
    figures used everywhere else in the app.

    Parameters:
    -----------
    feature_names : list
        Feature names in model order.
    shap_values : array-like
        One row of SHAP attributions, in target units.
    base_value : float
        The model's expected output over the background set.
    feature_values : array-like, optional
        The row's actual feature values, shown in the axis labels for context.
    max_features : int, default=10
        Largest contributors to show individually; the rest are pooled.
    title : str
        Figure title.

    Returns:
    --------
    plotly.graph_objects.Figure
    """
    contributions = np.asarray(shap_values, dtype=float).ravel()

    if len(contributions) != len(feature_names):
        raise ValueError(
            f"Got {len(contributions)} attributions for {len(feature_names)} feature names."
        )

    if contributions.size == 0:
        return _empty_figure(title, 'Contribution', 'Feature')

    labels = list(feature_names)
    if feature_values is not None:
        values = np.asarray(feature_values).ravel()
        labels = [
            f"{name} = {value:g}" if isinstance(value, (int, float, np.number))
            else f"{name} = {value}"
            for name, value in zip(labels, values, strict=False)
        ]

    # Rank by magnitude: the largest movers are what the reader needs to see.
    order = np.argsort(np.abs(contributions))[::-1]
    shown = order[:max_features]
    pooled = order[max_features:]

    steps = [(labels[i], contributions[i]) for i in shown]
    if pooled.size:
        steps.append((f"{pooled.size} other features", float(contributions[pooled].sum())))

    prediction = float(base_value + contributions.sum())

    fig = go.Figure(go.Waterfall(
        orientation='h',
        measure=['relative'] * len(steps) + ['total'],
        y=[name for name, _ in steps] + ['Prediction'],
        x=[value for _, value in steps] + [prediction],
        base=base_value,
        text=[f"{value:+,.2f}" for _, value in steps] + [f"{prediction:,.2f}"],
        textposition='outside',
        connector={'line': {'color': '#a0aec0'}},
        increasing={'marker': {'color': '#e53e3e'}},
        decreasing={'marker': {'color': '#3182ce'}},
        totals={'marker': {'color': '#4c6ef5'}},
    ))

    fig.update_layout(
        title=title,
        template=PLOT_TEMPLATE,
        xaxis_title=f"Contribution (baseline = {base_value:,.2f})",
        # Largest contributor at the top reads more naturally than at the bottom.
        yaxis={'title': 'Feature', 'autorange': 'reversed'},
        showlegend=False,
        margin={'l': 10, 'r': 10},
    )

    return fig


def create_shap_importance_bar(ranked_features, title="Mean impact on predictions"):
    """
    Horizontal bar chart of mean absolute SHAP value per feature.

    This is a model-agnostic replacement for ``feature_importances_``: it works
    for linear models and SVR too, and its units are the target's own.

    Parameters:
    -----------
    ranked_features : list of tuple
        ``(feature_name, mean_abs_shap)``, as returned by
        ``utils.explainability.summarize_global_importance``.
    title : str
        Figure title.

    Returns:
    --------
    plotly.graph_objects.Figure
    """
    if not ranked_features:
        return _empty_figure(title, 'Mean |SHAP|', 'Feature')

    names = [name for name, _ in ranked_features]
    values = [float(value) for _, value in ranked_features]

    fig = px.bar(
        x=values,
        y=names,
        orientation='h',
        title=title,
        labels={'x': 'Mean |SHAP| (target units)', 'y': 'Feature'},
        template=PLOT_TEMPLATE,
    )

    # Most important at the top.
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)

    return fig
