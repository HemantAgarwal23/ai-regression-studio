"""
Model Training Utilities

This module contains functions for:
- Model configuration
- Model training and evaluation
- Cross-validation
- Exporting a trained model as a self-contained bundle
"""

import io

import joblib
import numpy as np
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Cross-validation needs at least this many samples per fold to be meaningful.
MIN_SAMPLES_PER_FOLD = 2


def get_model_config(random_state=42):
    """
    Get configuration for all available regression models.

    Parameters:
    -----------
    random_state : int, default=42
        Seed applied to every model that accepts one, so runs are reproducible.

    Returns:
    --------
    dict
        Dictionary mapping model names to their sklearn model instances
    """
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=random_state),
        "Lasso": Lasso(alpha=1.0, random_state=random_state),
        "ElasticNet": ElasticNet(alpha=1.0, random_state=random_state),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state, max_depth=10),
        "Random Forest": RandomForestRegressor(random_state=random_state, n_estimators=100),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
        "Support Vector Regression": SVR(),
    }


def resolve_cv_folds(n_samples, requested_folds):
    """
    Clamp the requested fold count to what the training set can support.

    ``cross_val_score`` raises when ``cv`` exceeds the sample count, which is
    easy to hit on the small datasets people try the app with first.

    Parameters:
    -----------
    n_samples : int
        Number of training samples available.
    requested_folds : int
        Fold count the user asked for.

    Returns:
    --------
    int
        A usable fold count, or 0 when the data is too small to cross-validate.
    """
    max_folds = n_samples // MIN_SAMPLES_PER_FOLD
    if max_folds < 2:
        return 0
    return int(min(requested_folds, max_folds))


def train_model(model, X_train, y_train, X_test, y_test, cv_folds=5):
    """
    Train a model and evaluate its performance.

    Parameters:
    -----------
    model : sklearn model
        Model instance to train
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    X_test : array-like
        Test features
    y_test : array-like
        Test target
    cv_folds : int, default=5
        Requested number of folds for cross-validation. Reduced automatically
        when the training set is too small; cross-validation is skipped
        entirely below four samples.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'model': Trained model
        - 'predictions': Test predictions
        - 'mae': Mean Absolute Error
        - 'mse': Mean Squared Error
        - 'rmse': Root Mean Squared Error
        - 'r2': R2 score
        - 'cv_mean': Mean cross-validation score (nan when skipped)
        - 'cv_std': Standard deviation of cross-validation scores (nan when skipped)
        - 'cv_folds_used': Fold count actually used (0 when skipped)
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = metrics.mean_squared_error(y_test, y_pred)

    effective_folds = resolve_cv_folds(len(X_train), cv_folds)
    if effective_folds >= 2:
        cv_scores = cross_val_score(model, X_train, y_train, cv=effective_folds, scoring='r2')
        cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())
    else:
        cv_mean, cv_std = float('nan'), float('nan')

    return {
        'model': model,
        'predictions': y_pred,
        'mae': float(metrics.mean_absolute_error(y_test, y_pred)),
        'mse': float(mse),
        'rmse': float(np.sqrt(mse)),
        'r2': float(metrics.r2_score(y_test, y_pred)),
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'cv_folds_used': effective_folds,
    }


def export_model_bundle(model, feature_cols, target_col, scaler=None, encoders=None,
                        model_name=None, metrics_summary=None):
    """
    Serialize a trained model together with everything needed to reuse it.

    A bare pickled estimator is useless without the exact scaler, encoders and
    column order used at training time, so all of it travels in one bundle.

    Parameters:
    -----------
    model : sklearn model
        Trained model instance.
    feature_cols : list
        Feature names in the order the model expects them.
    target_col : str
        Name of the target the model predicts.
    scaler : sklearn transformer, optional
        Fitted scaler, or None when scaling was disabled.
    encoders : dict, optional
        Fitted LabelEncoders keyed by column name.
    model_name : str, optional
        Human-readable model name.
    metrics_summary : dict, optional
        Evaluation metrics to record alongside the model.

    Returns:
    --------
    bytes
        The serialized bundle, ready to hand to a download button or write to
        disk. Load it back with ``joblib.load``.
    """
    bundle = {
        'model': model,
        'model_name': model_name,
        'feature_cols': list(feature_cols),
        'target_col': target_col,
        'scaler': scaler,
        'encoders': encoders or {},
        'metrics': metrics_summary or {},
    }

    buffer = io.BytesIO()
    joblib.dump(bundle, buffer)
    return buffer.getvalue()
