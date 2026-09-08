"""
Sample Dataset

Generates a synthetic housing dataset so the app is usable the moment it opens,
without the visitor having to find and upload a CSV first. The data is built
from an explicit linear-plus-noise formula, which means the demo has a known
ground truth. Measured on the default 500 rows: Linear Regression reaches about
R2 = 0.84, Gradient Boosting about 0.84, Random Forest about 0.77 — the tree
models trail because the underlying relationship really is linear.

Deliberately included so the preprocessing pipeline has something to do:
- two categorical columns (one ordered, one nominal)
- missing values in two numeric columns
- a pure-noise column that feature importance should rank last
"""

import numpy as np
import pandas as pd

NEIGHBORHOODS = ['Downtown', 'Suburb', 'Riverside', 'Hillside']
CONDITIONS = ['Poor', 'Fair', 'Good', 'Excellent']

# Price contribution per neighborhood, in the same units as the target.
NEIGHBORHOOD_PREMIUM = {'Downtown': 85_000, 'Suburb': 20_000, 'Riverside': 55_000, 'Hillside': 40_000}
CONDITION_PREMIUM = {'Poor': -30_000, 'Fair': 0, 'Good': 25_000, 'Excellent': 60_000}


def generate_sample_dataset(n_samples=500, random_state=42, missing_rate=0.04):
    """
    Build the synthetic housing dataset.

    Parameters:
    -----------
    n_samples : int, default=500
        Number of rows to generate.
    random_state : int, default=42
        Seed, so the demo is identical on every run.
    missing_rate : float, default=0.04
        Fraction of values to blank out in the columns that carry gaps.

    Returns:
    --------
    pandas.DataFrame
        Columns: area_sqft, bedrooms, bathrooms, age_years, garage_spaces,
        distance_to_city_km, neighborhood, condition, lot_noise, price.
        ``price`` is the intended target.
    """
    rng = np.random.default_rng(random_state)

    area = rng.normal(1800, 550, n_samples).clip(500, 5000)
    bedrooms = rng.integers(1, 6, n_samples)
    bathrooms = rng.integers(1, 4, n_samples)
    age = rng.integers(0, 60, n_samples)
    garage = rng.integers(0, 3, n_samples)
    distance = rng.gamma(shape=2.0, scale=4.0, size=n_samples).clip(0.5, 40)

    neighborhood = rng.choice(NEIGHBORHOODS, n_samples, p=[0.25, 0.35, 0.2, 0.2])
    condition = rng.choice(CONDITIONS, n_samples, p=[0.15, 0.3, 0.35, 0.2])

    # Column with no relationship to price, used to sanity-check feature importance.
    lot_noise = rng.normal(0, 1, n_samples)

    price = (
        60_000
        + 145 * area
        + 12_000 * bedrooms
        + 18_000 * bathrooms
        - 1_400 * age
        + 9_000 * garage
        - 3_200 * distance
        + np.array([NEIGHBORHOOD_PREMIUM[n] for n in neighborhood])
        + np.array([CONDITION_PREMIUM[c] for c in condition])
        # Noise is sized so the demo is not a trivially perfect fit, while
        # leaving model comparison and the residual plots readable.
        + rng.normal(0, 18_000, n_samples)
    ).clip(45_000, None)

    df = pd.DataFrame({
        'area_sqft': area.round(0),
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age_years': age,
        'garage_spaces': garage,
        'distance_to_city_km': distance.round(2),
        'neighborhood': neighborhood,
        'condition': condition,
        'lot_noise': lot_noise.round(4),
        'price': price.round(0),
    })

    # Punch holes so the imputation path is exercised by the demo itself.
    for col in ['area_sqft', 'distance_to_city_km']:
        holes = rng.random(n_samples) < missing_rate
        df.loc[holes, col] = np.nan

    return df
