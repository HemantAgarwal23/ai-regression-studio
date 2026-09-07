"""
Tests for the SHAP explainability layer.

``shap`` is an optional dependency, so the tests that need it skip cleanly when
it is absent. The explainer-selection logic and the plot builders are pure and
are tested unconditionally.
"""

import numpy as np
import plotly.graph_objects as go
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from utils.explainability import (
    compute_shap_values,
    select_explainer_kind,
    summarize_global_importance,
)
from utils.visualization import create_shap_importance_bar, create_shap_waterfall

shap = pytest.importorskip("shap", reason="shap is an optional dependency")


@pytest.fixture(scope="module")
def regression_data():
    """Small deterministic regression problem shared across the SHAP tests."""
    X, y = make_regression(n_samples=80, n_features=4, noise=5, random_state=0)
    return X[:60], y[:60], X[60:], y[60:]


# --- explainer selection (pure, no shap needed at call time) ----------------


@pytest.mark.parametrize("model,expected", [
    (DecisionTreeRegressor(), 'tree'),
    (RandomForestRegressor(), 'tree'),
    (GradientBoostingRegressor(), 'tree'),
    (LinearRegression(), 'linear'),
    (Ridge(), 'linear'),
    (Lasso(), 'linear'),
    (ElasticNet(), 'linear'),
    (SVR(), 'kernel'),
])
def test_select_explainer_kind(model, expected):
    """Each model family routes to the algorithm that suits it."""
    assert select_explainer_kind(model) == expected


# --- SHAP computation -------------------------------------------------------


def test_shap_values_reconstruct_the_prediction(regression_data):
    """
    SHAP's core guarantee: base value plus attributions equals the prediction.

    If this drifts, the waterfall chart is lying about how the model got there.
    """
    X_train, y_train, X_test, _ = regression_data
    model = RandomForestRegressor(n_estimators=20, random_state=0).fit(X_train, y_train)

    row = X_test[:1]
    result = compute_shap_values(model, X_train, row)

    reconstructed = result.base_value + result.values[0].sum()
    assert np.isclose(reconstructed, model.predict(row)[0], rtol=0.02)
    assert result.explainer_kind == 'tree'


def test_shap_values_shape_matches_input(regression_data):
    """One attribution per feature, per explained row."""
    X_train, y_train, X_test, _ = regression_data
    model = LinearRegression().fit(X_train, y_train)

    result = compute_shap_values(model, X_train, X_test[:5])

    assert result.values.shape == (5, X_train.shape[1])
    assert result.explainer_kind == 'linear'


def test_shap_linear_reconstructs_prediction(regression_data):
    """The additivity guarantee holds for the linear explainer too."""
    X_train, y_train, X_test, _ = regression_data
    model = Ridge().fit(X_train, y_train)

    row = X_test[:1]
    result = compute_shap_values(model, X_train, row)

    reconstructed = result.base_value + result.values[0].sum()
    assert np.isclose(reconstructed, model.predict(row)[0], rtol=0.02)


def test_shap_rejects_mismatched_feature_counts(regression_data):
    """A background/explain width mismatch is caught with a clear message."""
    X_train, y_train, X_test, _ = regression_data
    model = LinearRegression().fit(X_train, y_train)

    with pytest.raises(ValueError, match="Feature count mismatch"):
        compute_shap_values(model, X_train, X_test[:1, :2])


def test_shap_rejects_empty_input(regression_data):
    """An empty explain set raises rather than producing an empty plot."""
    X_train, y_train, _, _ = regression_data
    model = LinearRegression().fit(X_train, y_train)

    with pytest.raises(ValueError, match="non-empty"):
        compute_shap_values(model, X_train, np.empty((0, X_train.shape[1])))


def test_summarize_global_importance_ranks_by_magnitude(regression_data):
    """Global importance is sorted descending by mean absolute attribution."""
    X_train, y_train, X_test, _ = regression_data
    model = RandomForestRegressor(n_estimators=20, random_state=0).fit(X_train, y_train)

    result = compute_shap_values(model, X_train, X_test[:10])
    names = [f"f{i}" for i in range(X_train.shape[1])]
    ranked = summarize_global_importance(result, names)

    assert len(ranked) == len(names)
    values = [value for _, value in ranked]
    assert values == sorted(values, reverse=True)
    assert all(value >= 0 for value in values)


def test_summarize_global_importance_rejects_wrong_name_count(regression_data):
    """A name/attribution count mismatch is an error, not a silent zip truncation."""
    X_train, y_train, X_test, _ = regression_data
    model = LinearRegression().fit(X_train, y_train)
    result = compute_shap_values(model, X_train, X_test[:3])

    with pytest.raises(ValueError):
        summarize_global_importance(result, ["only", "two"])


# --- plot builders ----------------------------------------------------------


def test_create_shap_waterfall_totals_to_the_prediction():
    """The waterfall's total step equals base value plus all contributions."""
    shap_values = np.array([10.0, -4.0, 2.0])
    base = 100.0

    fig = create_shap_waterfall(['a', 'b', 'c'], shap_values, base)

    assert isinstance(fig, go.Figure)
    trace = fig.data[0]
    assert trace.measure[-1] == 'total'
    assert trace.x[-1] == pytest.approx(base + shap_values.sum())


def test_create_shap_waterfall_pools_features_beyond_the_cap():
    """Only the largest movers get their own row; the rest are pooled into one."""
    shap_values = np.arange(1, 16, dtype=float)
    names = [f"f{i}" for i in range(15)]

    fig = create_shap_waterfall(names, shap_values, 0.0, max_features=5)

    labels = list(fig.data[0].y)
    assert "10 other features" in labels
    # 5 individual features + the pooled row + the total
    assert len(labels) == 7


def test_create_shap_waterfall_labels_include_feature_values():
    """Feature values are shown inline so the reader sees what drove the row."""
    fig = create_shap_waterfall(['area'], np.array([5.0]), 1.0, feature_values=[1500])

    assert 'area = 1500' in list(fig.data[0].y)[0]


def test_create_shap_waterfall_rejects_length_mismatch():
    """Attribution and name counts must agree."""
    with pytest.raises(ValueError, match="attributions"):
        create_shap_waterfall(['a', 'b'], np.array([1.0]), 0.0)


def test_create_shap_importance_bar_builds_figure():
    """The global importance chart renders from a ranked list."""
    fig = create_shap_importance_bar([('area', 12.0), ('age', 3.5)])

    assert isinstance(fig, go.Figure)
    assert set(fig.data[0].y) == {'area', 'age'}


def test_create_shap_importance_bar_handles_empty_ranking():
    """An empty ranking yields the placeholder figure, not a crash."""
    fig = create_shap_importance_bar([])

    assert fig.layout.annotations[0].text == 'No data to plot'
