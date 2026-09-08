# Code Explanation - AI Regression Studio

This document provides a comprehensive explanation of the AI Regression Studio project, its architecture, code structure, and machine learning models.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Architecture Overview](#architecture-overview)
3. [Machine Learning Models](#machine-learning-models)
4. [Code Flow](#code-flow)
5. [Data Flow](#data-flow)
6. [Key Components](#key-components)
7. [Interview Talking Points](#interview-talking-points)

---

## Project Structure

```
ai-regression-studio/
├── app.py                      # Streamlit UI: tabs, widgets, wiring
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Test and lint tooling
├── pyproject.toml              # pytest and ruff configuration
├── Dockerfile                  # Two-stage build, non-root runtime
├── .dockerignore               # Docker ignore file
├── README.md                   # Project documentation
├── CODE_EXPLANATION.md         # This file
├── .streamlit/
│   └── config.toml             # Upload cap, XSRF, theme
├── .github/workflows/
│   └── ci.yml                  # Tests, lint, Docker health check
├── assets/
│   └── styles.css              # Application theme
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── data_processing.py      # Leakage-free preprocessing pipeline
│   ├── explainability.py       # SHAP explainer selection and cost control
│   ├── model_training.py       # Model registry, training, export bundle
│   ├── sample_data.py          # Synthetic demo dataset
│   ├── ui_helpers.py           # Pure helpers for widget bounds
│   └── visualization.py        # Plotly visualization functions
└── tests/                      # Unit and end-to-end tests
    ├── __init__.py
    ├── test_app_smoke.py       # Drives app.py via Streamlit AppTest
    ├── test_data_processing.py # Includes explicit leakage guards
    ├── test_models.py
    ├── test_sample_data.py
    └── test_visualization.py
```

---

## Architecture Overview

The project follows a **modular architecture** with clear separation of concerns:

### 1. **Main Application (`app.py`)**
   - Streamlit UI and user interaction
   - Tab-based navigation (Data Upload, Explorer, Training, Results, Predictions)
   - Session state management
   - Integration of utility modules

### 2. **Data Processing Module (`utils/data_processing.py`)**
   - Missing value handling
   - Categorical encoding (LabelEncoder), tolerant of unseen categories
   - Train-test splitting
   - Feature scaling (StandardScaler, RobustScaler)
   - **Every fitted statistic is learned from the training split only** - see
     the leakage section below, which is the part most worth explaining

### 3. **Model Training Module (`utils/model_training.py`)**
   - Model configuration and initialization
   - Model training and evaluation
   - Cross-validation
   - Metric calculation (R², RMSE, MAE)

### 4. **Visualization Module (`utils/visualization.py`)**
   - Correlation heatmaps
   - Residual plots
   - Prediction scatter plots
   - All using Plotly for interactivity

---

## Machine Learning Models

The application supports **8 regression models**, each with different characteristics:

### Linear Models

#### 1. **Linear Regression**
- **Algorithm**: Ordinary Least Squares (OLS)
- **Use Case**: Baseline model, interpretable relationships
- **Pros**: Fast, interpretable, no hyperparameters
- **Cons**: Assumes linear relationships, sensitive to outliers
- **When to Use**: Simple datasets with linear patterns

#### 2. **Ridge Regression**
- **Algorithm**: L2 regularization added to OLS
- **Use Case**: Multicollinearity, overfitting prevention
- **Pros**: Handles multicollinearity, reduces overfitting
- **Cons**: Doesn't perform feature selection
- **Hyperparameter**: `alpha` (regularization strength)

#### 3. **Lasso Regression**
- **Algorithm**: L1 regularization added to OLS
- **Use Case**: Feature selection, sparse models
- **Pros**: Automatic feature selection, handles high-dimensional data
- **Cons**: May eliminate important features
- **Hyperparameter**: `alpha` (regularization strength)

#### 4. **ElasticNet**
- **Algorithm**: Combination of L1 and L2 regularization
- **Use Case**: Best of both Ridge and Lasso
- **Pros**: Handles multicollinearity and performs feature selection
- **Cons**: More complex, two hyperparameters
- **Hyperparameters**: `alpha`, `l1_ratio`

### Tree-Based Models

#### 5. **Decision Tree Regressor**
- **Algorithm**: Recursive binary splitting
- **Use Case**: Non-linear relationships, interpretable rules
- **Pros**: Interpretable, handles non-linear patterns, no scaling needed
- **Cons**: Prone to overfitting, unstable
- **Hyperparameters**: `max_depth`, `min_samples_split`

#### 6. **Random Forest Regressor**
- **Algorithm**: Ensemble of decision trees with bagging
- **Use Case**: Robust default choice, feature importance
- **Pros**: Reduces overfitting, handles non-linear patterns, feature importance
- **Cons**: Less interpretable, slower than single tree
- **Hyperparameters**: `n_estimators`, `max_depth`, `min_samples_split`

#### 7. **Gradient Boosting Regressor**
- **Algorithm**: Sequential ensemble, each tree corrects previous errors
- **Use Case**: High accuracy, complex patterns
- **Pros**: Often best performance, handles complex relationships
- **Cons**: Slower training, more hyperparameters, can overfit
- **Hyperparameters**: `n_estimators`, `learning_rate`, `max_depth`

### Advanced Models

#### 8. **Support Vector Regression (SVR)**
- **Algorithm**: Support Vector Machines adapted for regression
- **Use Case**: Non-linear relationships with kernel trick
- **Pros**: Handles non-linear patterns, robust to outliers
- **Cons**: Slow on large datasets, requires feature scaling
- **Hyperparameters**: `C`, `epsilon`, `kernel`

---

## Code Flow

### 1. **Data Upload Tab**
```
User uploads file → Load CSV/Excel → Store in session_state → Display metrics
```

### 2. **Data Explorer Tab**
```
Load data → Select target variable → Select features → Visualize relationships
```

### 3. **Model Training Tab**
```
Preprocess data (utils/data_processing.py) → 
Train selected models (utils/model_training.py) → 
Store results in session_state
```

### 4. **Results Dashboard Tab**
```
Load training results → Compare models → 
Visualize best model performance (utils/visualization.py) → 
Display feature importance
```

### 5. **Prediction Lab Tab**
```
Select model → Input feature values → 
Encode categorical features → Scale features → 
Make prediction → Display results with confidence interval
```

---

## Data Flow

### Preprocessing Pipeline

1. **Missing Value Handling**
   - Categorical: Fill with mode
   - Numerical: Fill with median
   - Documented in `preprocessing_steps`

2. **Categorical Encoding**
   - LabelEncoder for each categorical column
   - Encoders stored for prediction phase

3. **Train-Test Split**
   - Default: 80% train, 20% test
   - Configurable via sidebar

4. **Feature Scaling**
   - StandardScaler: Mean=0, Std=1
   - RobustScaler: Median-based, robust to outliers
   - Optional: No scaling

### Model Training Flow

1. **Model Initialization**
   - Get model config from `get_model_config()`
   - Filter to selected models

2. **Training Loop**
   - For each selected model:
     - Fit on training data
     - Predict on test data
     - Calculate metrics (R², RMSE, MAE)
     - Perform cross-validation

3. **Results Storage**
   - Store in `session_state.training_results`
   - Include model, predictions, and all metrics

### Prediction Flow

1. **Input Preparation**
   - User inputs feature values
   - Encode categorical features using stored encoders
   - Reorder to match training feature order

2. **Scaling**
   - Apply same scaler used during training

3. **Prediction**
   - Use trained model to predict
   - Calculate confidence interval using RMSE

4. **Display**
   - Show predicted value
   - Show confidence interval
   - Show feature importance (if available)

---

## Key Components

### Session State Variables

- `data_loaded`: Boolean flag for data upload
- `df`: Loaded dataframe
- `target_col`: Selected target variable
- `feature_cols`: Selected feature columns
- `training_results`: Dictionary of model results
- `encoders`: Dictionary of LabelEncoders for categorical features
- `scaler`: Scaler used for feature normalization
- `models_trained`: Boolean flag for training completion

### Utility Functions

#### `preprocess_data()`
- Complete preprocessing pipeline
- Returns: X_train, X_test, y_train, y_test, scaler, encoders, steps

#### `get_model_config()`
- Returns dictionary of all available models
- Each model initialized with default hyperparameters

#### `train_model()`
- Trains a single model
- Returns dictionary with model, predictions, and metrics

#### Visualization Functions
- `create_correlation_heatmap()`: Feature correlation matrix
- `create_residual_plot()`: Residuals vs predictions
- `create_prediction_scatter()`: Actual vs predicted values

---

## Train/Test Leakage: The Design Decision Worth Explaining

The obvious way to build this pipeline is also wrong:

```python
# WRONG - the shape most tutorials use
X = handle_missing_values(X)        # median computed over ALL rows
X, encoders = encode_categorical(X) # vocabulary built from ALL rows
X_train, X_test, ... = train_test_split(X, y)
scaler.fit(X_train)                 # only the scaler gets this right
```

The median used to fill a training row is computed partly from test rows, and
the encoder's vocabulary includes categories that only appear in the test split.
Information flows backwards from the held-out data into training, so the
reported R2 is measuring something better than what the model would actually
achieve on genuinely unseen data.

The pipeline in `preprocess_data` splits first:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)

fill_values = compute_imputation_values(X_train)   # learned on train
X_train, _ = apply_imputation(X_train, fill_values)
X_test, _  = apply_imputation(X_test, fill_values) # applied to test

encoders = fit_categorical_encoders(X_train)       # vocabulary from train
X_train, _ = transform_categorical(X_train, encoders)
X_test, _  = transform_categorical(X_test, encoders)

scaler.fit(X_train)                                # parameters from train
```

Splitting first creates a second problem worth naming: the test split can now
contain a category the encoder has never seen, and `LabelEncoder.transform`
raises on those. `transform_categorical` maps them to `UNSEEN_CATEGORY_CODE`
(-1) instead, which is also what protects a live prediction request from
crashing when a user types in a novel value.

Ordering matters within the split too. Rows with a missing **target** are
dropped *before* the split, because they carry no supervision signal on either
side. Rows with missing **features** are imputed *after* it.

Two tests pin this down so it cannot silently regress:
`test_preprocess_data_does_not_leak_imputation_values` and
`test_preprocess_data_scaler_fitted_on_train_only`.

**Interview framing:** the honest number is the useful one. A pipeline that
reports 0.95 by leaking and 0.84 without it has not gotten worse - it has
started telling the truth, and 0.84 is the number that will hold up in
production.

---

## Explainability: Choosing the Right SHAP Algorithm

SHAP answers "why *this* prediction" rather than "which features matter on
average". The design decision worth explaining is that there is no single SHAP
algorithm - there are several, with very different costs:

| Model family | Algorithm | Cost |
|---|---|---|
| Tree ensembles | `TreeExplainer` | Exact, polynomial time |
| Regularized linear | `LinearExplainer` | Exact, closed form |
| Everything else (SVR) | `KernelExplainer` | Approximate, sampled, orders of magnitude slower |

`select_explainer_kind` routes each model to the right one. Calling
`KernelExplainer` on a Random Forest would still produce correct-looking numbers,
just far more slowly and with sampling error - which is exactly the kind of thing
that looks fine in a demo and falls over on real data.

Cost is bounded in two more places. The background set is subsampled to 100 rows,
and for the kernel path it is further compressed into 25 weighted centroids via
`shap.kmeans`, because KernelExplainer's runtime scales with the background size.

**The property that makes the waterfall trustworthy** is additivity: the base
value plus every feature's attribution must reconstruct the model's actual
prediction. `test_shap_values_reconstruct_the_prediction` asserts exactly that
for both the tree and linear paths. If it ever drifts, the chart is misattributing
and the test fails rather than quietly lying.

The plots are drawn natively in Plotly rather than through SHAP's own plotting,
which renders via matplotlib. That keeps one charting library in the project
instead of two, and keeps the figures interactive like everything else.

**Interview framing:** global feature importance tells you the model leans on
square footage. A per-prediction waterfall tells a customer *why their* house was
valued at 442k - the baseline was 402k, square footage added 59k, age took off
16k. The second is what makes a model deployable in a setting where decisions
have to be justified.

**A useful sanity check:** the demo dataset deliberately contains `lot_noise`, a
column with no relationship to price. It should rank near the bottom of the SHAP
importances. If it ever climbs, something in the pipeline is leaking or
overfitting.

---

## Interview Talking Points

### Technical Highlights

1. **Modular Architecture**
   - Separation of concerns (data processing, model training, visualization)
   - Reusable utility functions
   - Easy to extend with new models or features

2. **Best Practices**
   - Dependencies constrained with lower and upper bounds, so a major release
     upstream cannot silently break a fresh install
   - Error handling scoped to the exceptions actually expected, rather than
     bare `except:` clauses that swallow real bugs
   - Session state reset whenever the dataset changes, so a stale encoder or
     trained model can never be applied to columns it was not fitted on
   - CI on four Python versions, plus a lint pass and a Docker health check
   - 97% test coverage on the `utils` package, including an end-to-end test
     that drives the real Streamlit app

3. **Model Selection Strategy**
   - Start with simple models (Linear Regression)
   - Progress to ensemble methods (Random Forest, Gradient Boosting)
   - Compare using multiple metrics (R², RMSE, MAE, CV score)

4. **Feature Engineering**
   - Automatic missing value handling
   - Categorical encoding
   - Feature scaling options
   - Feature importance analysis

5. **Evaluation Metrics**
   - **R² Score**: Proportion of variance explained (higher is better)
   - **RMSE**: Root mean squared error (lower is better)
   - **MAE**: Mean absolute error (lower is better)
   - **Cross-Validation**: Robust model evaluation

### Deployment Considerations

- Two-stage Docker build: `build-essential` lives only in the builder stage, so
  compilers never ship in the runtime image
- The container runs as an unprivileged user (uid 10001), not root
- Health check endpoint for orchestrator monitoring
- Requirements copied before application code, so the dependency layer stays
  cached across code changes
- Upload size capped in the app and again in `.streamlit/config.toml`
- XSRF protection on, usage stats off, tracebacks hidden from end users

### Future Enhancements

1. **Hyperparameter Tuning**: GridSearchCV or RandomizedSearchCV per model
2. **One-hot encoding**: label encoding imposes a false ordinal relationship on
   nominal columns, which hurts the linear models
3. **Calibrated intervals**: quantile regression or conformal prediction, in
   place of the current flat +/-1.96 x RMSE band
4. **Feature Engineering**: polynomial features, interaction terms
5. **Advanced Metrics**: adjusted R2, AIC, BIC
6. **LIME**: a second attribution method to cross-check SHAP
7. **Background training**: training currently blocks the UI thread

Already implemented: model persistence, via the `.joblib` bundle export that
carries the estimator together with its scaler, encoders and column order; and
SHAP explainability, covered in its own section below.

---

## Model Selection Guide

**When to use each model:**

- **Linear Regression**: Baseline, interpretable, fast
- **Ridge/Lasso**: Multicollinearity, feature selection
- **Decision Tree**: Interpretable rules, non-linear patterns
- **Random Forest**: Robust default, feature importance
- **Gradient Boosting**: Best accuracy, complex patterns
- **SVR**: Non-linear with kernel, small datasets

**Model Comparison Strategy:**
1. Start with Linear Regression as baseline
2. Try Ridge/Lasso if overfitting
3. Use Random Forest for robust performance
4. Use Gradient Boosting for best accuracy
5. Compare all using cross-validation

---

## Conclusion

This project demonstrates:
- **Full ML Pipeline**: From data loading to predictions
- **Multiple Models**: 8 different regression algorithms
- **Best Practices**: Modular code, error handling, documentation
- **Production Ready**: Docker, health checks, deployment ready
- **User-Friendly**: Interactive UI, visualizations, clear feedback

Perfect for demonstrating ML engineering skills in interviews!

