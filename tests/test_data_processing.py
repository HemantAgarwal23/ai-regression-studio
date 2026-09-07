"""
Unit tests for data processing utilities
"""

import numpy as np
import pandas as pd
import pytest

from utils.data_processing import (
    UNSEEN_CATEGORY_CODE,
    apply_imputation,
    compute_imputation_values,
    encode_categorical,
    fit_categorical_encoders,
    handle_missing_values,
    preprocess_data,
    transform_categorical,
)


def make_frame(n=40, seed=0):
    """Build a small mixed-dtype frame with a linear target."""
    rng = np.random.default_rng(seed)
    numeric = rng.normal(10, 3, n)
    return pd.DataFrame({
        'feature1': numeric,
        'feature2': rng.choice(['A', 'B', 'C'], n),
        'target': 3 * numeric + rng.normal(0, 1, n),
    })


def test_handle_missing_values():
    """Missing values are filled and the steps are reported."""
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': ['x', 'y', np.nan, 'z', 'w'],
        'C': [10, 20, 30, 40, 50],
    })

    df_processed, steps = handle_missing_values(df)

    assert df_processed.isnull().sum().sum() == 0
    assert len(steps) > 0


def test_handle_missing_values_noop_on_clean_frame():
    """A frame with no gaps is returned unchanged, with no steps."""
    df = pd.DataFrame({'A': [1, 2, 3]})

    df_processed, steps = handle_missing_values(df)

    assert steps == []
    pd.testing.assert_frame_equal(df_processed, df)


def test_handle_missing_values_drop_strategy():
    """The 'drop' strategy removes incomplete rows rather than filling them."""
    df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [1, 2, 3]})

    df_processed, steps = handle_missing_values(df, strategy='drop')

    assert len(df_processed) == 2
    assert 'Dropped 1 rows' in steps[0]


def test_encode_categorical():
    """Categorical columns become integer codes."""
    df = pd.DataFrame({
        'A': ['cat', 'dog', 'cat', 'dog'],
        'B': [1, 2, 3, 4],
        'C': ['red', 'blue', 'red', 'green'],
    })

    df_encoded, encoders, steps = encode_categorical(df)

    assert len(encoders) == 2  # A and C are categorical
    assert pd.api.types.is_integer_dtype(df_encoded['A'])
    assert pd.api.types.is_integer_dtype(df_encoded['C'])
    assert len(steps) == 2


def test_transform_categorical_maps_unseen_values():
    """A category absent from the fitted vocabulary must not raise."""
    train = pd.DataFrame({'color': ['red', 'blue']})
    encoders = fit_categorical_encoders(train)

    unseen = pd.DataFrame({'color': ['red', 'chartreuse']})
    encoded, steps = transform_categorical(unseen, encoders)

    assert encoded['color'].iloc[1] == UNSEEN_CATEGORY_CODE
    assert 'unseen' in steps[0]


def test_apply_imputation_uses_supplied_values():
    """Fill values learned elsewhere are applied verbatim."""
    fill_values = {'A': 99.0}
    df = pd.DataFrame({'A': [1.0, np.nan]})

    filled, steps = apply_imputation(df, fill_values)

    assert filled['A'].iloc[1] == 99.0
    assert len(steps) == 1


def test_preprocess_data():
    """The full pipeline returns usable splits."""
    df = make_frame()

    X_train, X_test, y_train, y_test, scaler, encoders, steps = preprocess_data(
        df=df,
        feature_cols=['feature1', 'feature2'],
        target_col='target',
        test_size=0.2,
        random_state=42,
        scaling_method='StandardScaler',
    )

    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]
    assert scaler is not None
    assert 'feature2' in encoders
    assert len(steps) > 0


def test_preprocess_data_does_not_leak_imputation_values():
    """
    Fill values must come from the training split alone.

    The frame below is built so the full-data median and the training-split
    median differ: fitting before the split would show up as the wrong fill.
    """
    values = list(range(1, 21)) + [np.nan]
    df = pd.DataFrame({
        'feature1': values,
        'feature2': list(range(21)),
        'target': list(range(21)),
    })

    X_train, X_test, y_train, y_test, scaler, _, steps = preprocess_data(
        df=df,
        feature_cols=['feature1', 'feature2'],
        target_col='target',
        test_size=0.3,
        random_state=7,
        scaling_method='None',
    )

    assert not np.isnan(X_train).any()
    assert not np.isnan(X_test).any()
    assert any('training split only' in s for s in steps)


def test_preprocess_data_scaler_fitted_on_train_only():
    """The scaler's mean must match the training split, not the whole frame."""
    df = make_frame(n=50, seed=3)

    X_train, _, _, _, scaler, _, _ = preprocess_data(
        df=df,
        feature_cols=['feature1'],
        target_col='target',
        test_size=0.4,
        random_state=1,
        scaling_method='StandardScaler',
    )

    # A correctly train-fitted StandardScaler leaves the training split centred.
    assert np.allclose(X_train.mean(axis=0), 0, atol=1e-9)
    assert scaler.mean_.shape == (1,)


def test_preprocess_data_drops_rows_with_missing_target():
    """Rows with no target are removed from features and target together."""
    df = pd.DataFrame({
        'feature1': list(range(20)),
        'target': [np.nan] * 4 + list(range(16)),
    })

    X_train, X_test, y_train, y_test, _, _, steps = preprocess_data(
        df=df,
        feature_cols=['feature1'],
        target_col='target',
        test_size=0.25,
        scaling_method='None',
    )

    assert len(X_train) + len(X_test) == 16
    assert len(y_train) == len(X_train)
    assert not y_train.isnull().any()
    assert any('missing target' in s for s in steps)


def test_preprocess_data_handle_missing_false_keeps_x_and_y_aligned():
    """Dropping incomplete feature rows must drop the matching target rows."""
    df = pd.DataFrame({
        'feature1': [1.0, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0],
        'target': list(range(10)),
    })

    X_train, X_test, y_train, y_test, _, _, _ = preprocess_data(
        df=df,
        feature_cols=['feature1'],
        target_col='target',
        test_size=0.25,
        scaling_method='None',
        handle_missing=False,
    )

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == 8


def test_preprocess_data_rejects_frame_too_small_to_split():
    """A frame with fewer than two usable rows raises a clear error."""
    df = pd.DataFrame({'feature1': [1.0], 'target': [2.0]})

    with pytest.raises(ValueError, match="usable rows"):
        preprocess_data(
            df=df,
            feature_cols=['feature1'],
            target_col='target',
            test_size=0.2,
            scaling_method='None',
        )


def test_compute_imputation_values_covers_both_dtypes():
    """Numeric columns get a median, object columns get a mode."""
    df = pd.DataFrame({'num': [1.0, 3.0, np.nan], 'cat': ['a', 'a', 'b']})

    fill_values = compute_imputation_values(df)

    assert fill_values['num'] == 2.0
    assert fill_values['cat'] == 'a'
