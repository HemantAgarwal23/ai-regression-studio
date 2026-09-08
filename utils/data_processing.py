"""
Data Processing Utilities

This module contains functions for preprocessing data including:
- Missing value handling
- Categorical encoding
- Feature scaling
- Train-test splitting

Leakage policy
--------------
Every transformation that learns a statistic from the data (imputation fill
values, categorical vocabularies, scaler parameters) is *fitted on the training
split only* and then applied to the test split. Fitting on the full frame before
splitting would let test-set information influence training and inflate the
reported metrics.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler

# Value assigned to categories that appear in the test split (or in a prediction
# request) but were never seen while fitting the encoder.
UNSEEN_CATEGORY_CODE = -1


def compute_imputation_values(df, strategy='median'):
    """
    Learn the fill value for every column that needs one.

    Parameters:
    -----------
    df : pandas.DataFrame
        Frame to learn from. Pass the *training* split only.
    strategy : str, default='median'
        Strategy for numerical columns: 'median' or 'mean'.

    Returns:
    --------
    dict
        Mapping of column name to the value that should fill its gaps.
    """
    fill_values = {}

    for col in df.columns:
        if df[col].dtype in ['object', 'category']:
            mode_value = df[col].mode()
            fill_values[col] = mode_value[0] if not mode_value.empty else 'Unknown'
        elif strategy == 'mean':
            fill_values[col] = df[col].mean()
        else:
            fill_values[col] = df[col].median()

    return fill_values


def apply_imputation(df, fill_values):
    """
    Apply pre-computed fill values to a frame.

    Parameters:
    -----------
    df : pandas.DataFrame
        Frame to fill.
    fill_values : dict
        Mapping produced by :func:`compute_imputation_values`.

    Returns:
    --------
    pandas.DataFrame
        Frame with missing values filled.
    list
        List of preprocessing steps performed.
    """
    df_processed = df.copy()
    steps = []

    for col in df_processed.columns:
        missing_count = int(df_processed[col].isnull().sum())
        if missing_count == 0 or col not in fill_values:
            continue

        fill_value = fill_values[col]
        if pd.isna(fill_value):
            # Column is entirely missing in the training split; nothing was
            # learned, so fall back to a neutral value rather than crashing.
            fill_value = 'Unknown' if df_processed[col].dtype in ['object', 'category'] else 0

        df_processed[col] = df_processed[col].fillna(fill_value)

        if isinstance(fill_value, (int, float, np.number)):
            steps.append(f"Filled {missing_count} missing values in '{col}' with {fill_value:.2f}")
        else:
            steps.append(f"Filled {missing_count} missing values in '{col}' with '{fill_value}'")

    return df_processed, steps


def handle_missing_values(df, strategy='median'):
    """
    Handle missing values in a single dataframe.

    Convenience wrapper that learns and applies fill values on the same frame.
    Use :func:`compute_imputation_values` + :func:`apply_imputation` when the
    train and test splits must be treated separately.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with potential missing values
    strategy : str, default='median'
        Strategy for filling missing values: 'median', 'mean', or 'drop'

    Returns:
    --------
    pandas.DataFrame
        Dataframe with missing values handled
    list
        List of preprocessing steps performed
    """
    if df.isnull().sum().sum() == 0:
        return df.copy(), []

    if strategy == 'drop':
        df_processed = df.dropna()
        dropped = len(df) - len(df_processed)
        return df_processed, [f"Dropped {dropped} rows containing missing values"]

    fill_values = compute_imputation_values(df, strategy=strategy)
    return apply_imputation(df, fill_values)


def fit_categorical_encoders(df):
    """
    Fit a LabelEncoder for each categorical column.

    Parameters:
    -----------
    df : pandas.DataFrame
        Frame to learn vocabularies from. Pass the *training* split only.

    Returns:
    --------
    dict
        Mapping of column name to a fitted LabelEncoder.
    """
    encoders = {}

    for col in df.select_dtypes(include=['object', 'category']).columns:
        encoder = LabelEncoder()
        encoder.fit(df[col].astype(str))
        encoders[col] = encoder

    return encoders


def transform_categorical(df, encoders):
    """
    Apply fitted encoders to a frame, tolerating unseen categories.

    Categories absent from the encoder's vocabulary map to
    ``UNSEEN_CATEGORY_CODE`` instead of raising, so a test split or a live
    prediction request cannot crash the pipeline.

    Parameters:
    -----------
    df : pandas.DataFrame
        Frame to encode.
    encoders : dict
        Mapping produced by :func:`fit_categorical_encoders`.

    Returns:
    --------
    pandas.DataFrame
        Frame with categorical columns encoded as integers.
    list
        List of preprocessing steps performed.
    """
    df_encoded = df.copy()
    steps = []

    for col, encoder in encoders.items():
        if col not in df_encoded.columns:
            continue

        lookup = {label: code for code, label in enumerate(encoder.classes_)}
        values = df_encoded[col].astype(str)
        df_encoded[col] = values.map(lookup).fillna(UNSEEN_CATEGORY_CODE).astype(int)

        unseen = int((df_encoded[col] == UNSEEN_CATEGORY_CODE).sum())
        step = f"Label encoded '{col}' with {len(encoder.classes_)} unique categories"
        if unseen:
            step += f" ({unseen} unseen values mapped to {UNSEEN_CATEGORY_CODE})"
        steps.append(step)

    return df_encoded, steps


def encode_categorical(df):
    """
    Encode categorical variables using LabelEncoder.

    Convenience wrapper that fits and transforms on the same frame.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with categorical columns

    Returns:
    --------
    pandas.DataFrame
        Dataframe with categorical columns encoded
    dict
        Dictionary mapping column names to LabelEncoder objects
    list
        List of preprocessing steps performed
    """
    encoders = fit_categorical_encoders(df)
    df_encoded, steps = transform_categorical(df, encoders)
    return df_encoded, encoders, steps


def preprocess_data(df, feature_cols, target_col, test_size=0.2, random_state=42,
                    scaling_method='StandardScaler', handle_missing=True):
    """
    Complete data preprocessing pipeline.

    Order of operations is chosen to avoid train/test leakage:

    1. Drop rows with a missing target (they carry no supervision signal)
    2. Train-test split
    3. Fit imputation values on train, apply to train and test
    4. Fit categorical encoders on train, apply to train and test
    5. Fit the scaler on train, apply to train and test

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    feature_cols : list
        List of feature column names
    target_col : str
        Name of target column
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
    scaling_method : str, default='StandardScaler'
        Scaling method: 'StandardScaler', 'RobustScaler', or 'None'
    handle_missing : bool, default=True
        Whether to impute missing features. When False, rows with any missing
        feature are dropped instead.

    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, scaler, encoders, preprocessing_steps)

    Raises:
    -------
    ValueError
        If no usable rows remain, or the split leaves a side empty.
    """
    preprocessing_steps = []

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Rows without a target cannot be trained on or scored against.
    target_mask = y.notnull()
    if not target_mask.all():
        preprocessing_steps.append(
            f"Removed {int((~target_mask).sum())} rows with missing target values"
        )
        X = X[target_mask]
        y = y[target_mask]

    if not handle_missing:
        # Dropping must remove the matching target rows too, or X and y desync.
        feature_mask = X.notnull().all(axis=1)
        if not feature_mask.all():
            preprocessing_steps.append(
                f"Removed {int((~feature_mask).sum())} rows with missing feature values"
            )
            X = X[feature_mask]
            y = y[feature_mask]

    if len(X) < 2:
        raise ValueError(
            f"Only {len(X)} usable rows remain after cleaning; need at least 2 to split."
        )

    # --- Split first: everything below is fitted on the training side only ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            f"test_size={test_size} leaves an empty split for {len(X)} rows. "
            "Use a larger dataset or a different test size."
        )

    preprocessing_steps.append(
        f"Split data: {len(X_train)} training samples, {len(X_test)} test samples "
        f"(test_size={test_size})"
    )

    # --- Imputation: fill values learned from the training split ---
    if handle_missing and X.isnull().any().any():
        fill_values = compute_imputation_values(X_train)
        X_train, train_steps = apply_imputation(X_train, fill_values)
        X_test, _ = apply_imputation(X_test, fill_values)
        preprocessing_steps.extend(train_steps)
        preprocessing_steps.append("Imputation values learned from the training split only")

    # --- Categorical encoding: vocabulary learned from the training split ---
    encoders = fit_categorical_encoders(X_train)
    if encoders:
        X_train, encoding_steps = transform_categorical(X_train, encoders)
        X_test, _ = transform_categorical(X_test, encoders)
        preprocessing_steps.extend(encoding_steps)
    else:
        preprocessing_steps.append("No categorical features to encode")

    # --- Scaling: parameters learned from the training split ---
    scaler = None
    if scaling_method == 'StandardScaler':
        scaler = StandardScaler()
    elif scaling_method == 'RobustScaler':
        scaler = RobustScaler()

    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        preprocessing_steps.append(f"Applied {scaling_method} for feature normalization")
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values
        preprocessing_steps.append("No feature scaling applied")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoders, preprocessing_steps
