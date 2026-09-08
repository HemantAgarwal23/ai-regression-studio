"""
Tests for the bundled demo dataset.

The demo is the first thing a visitor sees, so it needs to stay well-formed and
to remain learnable end to end.
"""

import numpy as np
import pandas as pd

from utils.data_processing import preprocess_data
from utils.model_training import get_model_config, train_model
from utils.sample_data import generate_sample_dataset

FEATURE_COLS = [
    'area_sqft', 'bedrooms', 'bathrooms', 'age_years', 'garage_spaces',
    'distance_to_city_km', 'neighborhood', 'condition', 'lot_noise',
]


def test_generate_sample_dataset_shape_and_columns():
    """The demo frame has the expected size and schema."""
    df = generate_sample_dataset(n_samples=200)

    assert len(df) == 200
    assert list(df.columns) == FEATURE_COLS + ['price']
    assert pd.api.types.is_numeric_dtype(df['price'])


def test_generate_sample_dataset_is_deterministic():
    """The same seed produces the same data, so the demo never shifts."""
    pd.testing.assert_frame_equal(
        generate_sample_dataset(n_samples=50),
        generate_sample_dataset(n_samples=50),
    )


def test_generate_sample_dataset_exercises_preprocessing():
    """The demo deliberately contains gaps and categorical columns."""
    df = generate_sample_dataset(n_samples=300)

    assert df['area_sqft'].isnull().any()
    assert df['distance_to_city_km'].isnull().any()
    assert df['neighborhood'].dtype == object
    assert df['condition'].dtype == object
    assert df['price'].isnull().sum() == 0


def test_demo_dataset_trains_to_a_good_score():
    """
    End-to-end guard: the demo must stay learnable.

    Thresholds sit well below the scores these models actually reach (Linear
    around 0.84, Random Forest around 0.77 on the default 500 rows), so the test
    catches a real preprocessing regression without failing on ordinary
    run-to-run variation. Linear gets the higher bar because the data really is
    generated from a linear formula.
    """
    df = generate_sample_dataset()

    X_train, X_test, y_train, y_test, _, _, _ = preprocess_data(
        df=df,
        feature_cols=FEATURE_COLS,
        target_col='price',
        test_size=0.2,
        random_state=42,
        scaling_method='StandardScaler',
    )

    config = get_model_config()

    linear = train_model(config["Linear Regression"], X_train, y_train, X_test, y_test, cv_folds=3)
    assert linear['r2'] > 0.75
    assert np.isfinite(linear['cv_mean'])

    forest = train_model(config["Random Forest"], X_train, y_train, X_test, y_test, cv_folds=3)
    assert forest['r2'] > 0.65
    assert np.isfinite(forest['cv_mean'])
