"""
Explainability Utilities

Wraps SHAP so the app can answer "why this prediction?" rather than only
"which features matter on average".

Two things this module exists to handle:

1. **Explainer choice.** SHAP has an exact, fast algorithm for tree ensembles
   and another for linear models. Everything else falls back to KernelExplainer,
   which approximates by sampling and is orders of magnitude slower. Picking the
   right one per model is the difference between an instant plot and a hang.

2. **Cost control.** KernelExplainer's runtime scales with both the background
   set and the number of rows explained, so both are capped here rather than
   left to whatever the user happened to upload.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.base import is_classifier
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

# Rows sampled from the training set to represent "typical" input. SHAP values
# are attributions relative to this reference, so it must be representative.
BACKGROUND_SAMPLES = 100

# KernelExplainer is approximate and slow; keep its work bounded.
KERNEL_BACKGROUND_CLUSTERS = 25
KERNEL_NSAMPLES = 200

# Cap on rows explained at once for the dataset-wide summary.
MAX_SUMMARY_ROWS = 200

TREE_MODELS = (
    DecisionTreeRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)

LINEAR_MODELS = (LinearRegression, Ridge, Lasso, ElasticNet)


@dataclass
class ShapResult:
    """
    SHAP attributions for one or more rows.

    Attributes:
    -----------
    values : numpy.ndarray
        Shape (n_rows, n_features). Each entry is that feature's contribution to
        moving the prediction away from ``base_value``, in target units.
    base_value : float
        The model's expected output over the background set. Attributions plus
        this base reconstruct the prediction.
    explainer_kind : str
        'tree', 'linear' or 'kernel' — which algorithm produced the values.
    """

    values: np.ndarray
    base_value: float
    explainer_kind: str

    def mean_abs(self):
        """Mean absolute attribution per feature, for a global importance ranking."""
        return np.abs(self.values).mean(axis=0)


def _sample_background(X, n_samples, random_state=42):
    """Take a representative subsample of the background data."""
    X = np.asarray(X, dtype=float)
    if len(X) <= n_samples:
        return X

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=n_samples, replace=False)
    return X[idx]


def select_explainer_kind(model):
    """
    Decide which SHAP algorithm suits a model.

    Parameters:
    -----------
    model : sklearn estimator
        A fitted regressor.

    Returns:
    --------
    str
        'tree', 'linear', or 'kernel'.
    """
    if isinstance(model, TREE_MODELS):
        return 'tree'
    if isinstance(model, LINEAR_MODELS):
        return 'linear'
    return 'kernel'


def compute_shap_values(model, X_background, X_explain, random_state=42):
    """
    Compute SHAP attributions for the given rows.

    Parameters:
    -----------
    model : sklearn estimator
        Fitted regressor to explain.
    X_background : array-like
        Training data, used as the reference distribution.
    X_explain : array-like
        Rows to explain. Shape (n_rows, n_features).
    random_state : int, default=42
        Seed for background sampling, so a given input explains identically twice.

    Returns:
    --------
    ShapResult

    Raises:
    -------
    ImportError
        If the ``shap`` package is not installed.
    ValueError
        If the model is a classifier, or the inputs are empty or mismatched.
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - exercised only without shap
        raise ImportError(
            "SHAP explanations require the 'shap' package. Install it with "
            "`pip install shap`."
        ) from exc

    if is_classifier(model):
        raise ValueError("compute_shap_values supports regressors only.")

    background = _sample_background(X_background, BACKGROUND_SAMPLES, random_state)
    rows = np.atleast_2d(np.asarray(X_explain, dtype=float))

    if background.size == 0 or rows.size == 0:
        raise ValueError("Both background and explain sets must be non-empty.")

    if background.shape[1] != rows.shape[1]:
        raise ValueError(
            f"Feature count mismatch: background has {background.shape[1]}, "
            f"rows to explain have {rows.shape[1]}."
        )

    kind = select_explainer_kind(model)

    if kind == 'tree':
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(rows, check_additivity=False)
    elif kind == 'linear':
        explainer = shap.LinearExplainer(model, background)
        values = explainer.shap_values(rows)
    else:
        # KernelExplainer cost grows with the background size, so summarise it
        # into a handful of weighted centroids first.
        summary = shap.kmeans(background, min(KERNEL_BACKGROUND_CLUSTERS, len(background)))
        explainer = shap.KernelExplainer(model.predict, summary)
        values = explainer.shap_values(rows, nsamples=KERNEL_NSAMPLES, silent=True)

    values = np.atleast_2d(np.asarray(values, dtype=float))

    base = explainer.expected_value
    # Some explainers hand back a one-element array rather than a scalar.
    base_value = float(np.asarray(base).ravel()[0])

    return ShapResult(values=values, base_value=base_value, explainer_kind=kind)


def summarize_global_importance(shap_result, feature_names):
    """
    Rank features by mean absolute SHAP value.

    Unlike ``feature_importances_``, this works for every model type and is
    expressed in target units, so the numbers are directly interpretable.

    Parameters:
    -----------
    shap_result : ShapResult
        Output of :func:`compute_shap_values` over multiple rows.
    feature_names : list
        Feature names, in the order the model was fitted on.

    Returns:
    --------
    list of tuple
        ``(feature_name, mean_abs_shap)`` sorted most important first.
    """
    means = shap_result.mean_abs()

    if len(means) != len(feature_names):
        raise ValueError(
            f"Got {len(means)} attributions for {len(feature_names)} feature names."
        )

    pairs = zip(feature_names, means, strict=True)
    return sorted(pairs, key=lambda pair: pair[1], reverse=True)
