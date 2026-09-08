"""
Unit tests for model training utilities
"""

import joblib
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

from utils.model_training import (
    export_model_bundle,
    get_model_config,
    resolve_cv_folds,
    train_model,
)

EXPECTED_MODELS = [
    "Linear Regression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "Support Vector Regression",
]


def split_synthetic(n_samples=100, n_features=5, seed=42):
    """Return a deterministic train/test split of synthetic regression data."""
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=10, random_state=seed)
    idx = int(0.8 * len(X))
    return X[:idx], X[idx:], y[:idx], y[idx:]


def test_get_model_config():
    """Every advertised model is present and instantiated."""
    config = get_model_config()

    for model_name in EXPECTED_MODELS:
        assert model_name in config
        assert config[model_name] is not None


def test_get_model_config_applies_random_state():
    """Seeded models pick up the requested seed."""
    config = get_model_config(random_state=7)

    assert config["Random Forest"].random_state == 7
    assert config["Gradient Boosting"].random_state == 7


def test_train_model():
    """Training returns predictions and the full metric set."""
    X_train, X_test, y_train, y_test = split_synthetic()
    model = get_model_config()["Linear Regression"]

    result = train_model(model, X_train, y_train, X_test, y_test, cv_folds=3)

    for key in ('model', 'predictions', 'r2', 'rmse', 'mae', 'mse', 'cv_mean', 'cv_std'):
        assert key in result

    assert len(result['predictions']) == len(y_test)
    assert result['r2'] > 0.5  # a linear model on linear data should do well
    assert np.isclose(result['rmse'], np.sqrt(result['mse']))
    assert result['cv_folds_used'] == 3


def test_resolve_cv_folds_clamps_to_sample_count():
    """Fold count never exceeds what the sample count supports."""
    assert resolve_cv_folds(100, 5) == 5
    assert resolve_cv_folds(6, 10) == 3
    assert resolve_cv_folds(3, 5) == 0  # too few samples to cross-validate


def test_train_model_skips_cv_on_tiny_training_set():
    """A training set too small for the requested folds must not raise."""
    X_train, X_test, y_train, y_test = split_synthetic(n_samples=8, n_features=2)

    result = train_model(
        get_model_config()["Linear Regression"],
        X_train, y_train, X_test, y_test,
        cv_folds=10,
    )

    assert result['cv_folds_used'] <= len(X_train)
    assert len(result['predictions']) == len(y_test)


def test_export_model_bundle_roundtrip():
    """A bundle reloads with the estimator and its preprocessing intact."""
    X_train, X_test, y_train, y_test = split_synthetic(n_samples=60, n_features=3)
    model = RandomForestRegressor(n_estimators=5, random_state=42).fit(X_train, y_train)

    payload = export_model_bundle(
        model=model,
        feature_cols=['a', 'b', 'c'],
        target_col='y',
        scaler=None,
        encoders={},
        model_name='Random Forest',
        metrics_summary={'r2': 0.9},
    )

    import io
    bundle = joblib.load(io.BytesIO(payload))

    assert bundle['model_name'] == 'Random Forest'
    assert bundle['feature_cols'] == ['a', 'b', 'c']
    assert bundle['target_col'] == 'y'
    assert bundle['metrics']['r2'] == 0.9
    # The reloaded estimator must still predict identically.
    assert np.allclose(bundle['model'].predict(X_test), model.predict(X_test))
