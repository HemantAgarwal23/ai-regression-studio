# AI Regression Studio

[![CI](https://github.com/HemantAgarwal23/ai-regression-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/HemantAgarwal23/ai-regression-studio/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A Streamlit application that takes a tabular dataset from upload to a trained,
> comparable, exportable regression model — without writing any code.

Upload a CSV or Excel file (or load the bundled demo dataset), pick a target
column, and the app preprocesses the data, trains up to eight scikit-learn
regressors, ranks them on held-out data, and lets you make live predictions with
the winner. Trained models export as a self-contained bundle you can load
elsewhere.

---

## Why the preprocessing is worth a look

The easy way to build this app leaks test data into training: impute missing
values and fit the label encoders and scaler on the whole dataframe, *then*
split. The reported R² comes out flattering and wrong.

This app splits first. Imputation fill values, categorical vocabularies and
scaler parameters are all fitted on the training split only and then applied to
the test split — see [`utils/data_processing.py`](utils/data_processing.py) and
the leakage tests in
[`tests/test_data_processing.py`](tests/test_data_processing.py). Categories the
encoder never saw during training map to a sentinel code rather than raising, so
neither a test row nor a live prediction can crash the pipeline.

## Features

**Data**
- CSV and Excel upload, with a 50 MB cap enforced in the app and at the server
- A bundled synthetic housing dataset, so the app is usable with nothing to hand
- Data quality summary: missingness, dtype counts, duplicate rows
- Target suggestion from column naming, and correlation-ranked feature preselection

**Modelling**
- Eight regressors: Linear, Ridge, Lasso, ElasticNet, Decision Tree,
  Random Forest, Gradient Boosting, SVR
- Leakage-free preprocessing pipeline with every step reported in the UI
- K-fold cross-validation, with the fold count clamped to what the data supports
- Per-model failures are isolated: one model erroring does not lose the rest

**Results**
- Leaderboard ranked on numeric R² (not on formatted strings)
- R² and RMSE comparison charts, actual-vs-predicted and residual plots
- Feature importance for tree-based models
- Exports: leaderboard CSV, test predictions CSV, and a `.joblib` model bundle
  carrying the estimator, scaler, encoders and column order together

**Predictions**
- Per-feature inputs bounded by the dataset's own range
- An uncertainty band derived from test RMSE, labelled as the approximation it is
- **SHAP waterfall per prediction**: which features moved *this* prediction, and
  by how much, in the target's own units

**Explainability**
- Per-prediction attribution via a Plotly waterfall, drawn natively rather than
  through SHAP's matplotlib plots
- Global importance by mean absolute SHAP value — model-agnostic, so it works
  for linear models and SVR, not only tree ensembles
- Explainer chosen per model: exact TreeSHAP for ensembles, the linear explainer
  for regularized linear models, sampled KernelExplainer as the fallback

## Quick start

```bash
git clone https://github.com/HemantAgarwal23/ai-regression-studio.git
cd ai-regression-studio

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`, click **Load demo dataset** on the first tab, and
work left to right through the tabs.

### Docker

```bash
docker build -t ai-regression-studio .
docker run -p 8501:8501 ai-regression-studio
```

The image is a two-stage build — compilers stay in the builder stage — and the
container runs as an unprivileged user.

## Project layout

```
ai-regression-studio/
├── app.py                      # Streamlit UI: tabs, widgets, wiring
├── assets/
│   └── styles.css              # Application theme
├── utils/
│   ├── data_processing.py      # Leakage-free preprocessing pipeline
│   ├── explainability.py       # SHAP explainer selection and cost control
│   ├── model_training.py       # Model registry, training, export bundle
│   ├── sample_data.py          # Synthetic demo dataset
│   ├── ui_helpers.py           # Pure helpers for widget bounds
│   └── visualization.py        # Plotly figure builders
├── tests/
│   ├── test_app_smoke.py       # Drives app.py via Streamlit AppTest
│   ├── test_data_processing.py # Includes explicit leakage guards
│   ├── test_explainability.py  # SHAP additivity and explainer routing
│   ├── test_models.py
│   ├── test_sample_data.py     # End-to-end guard on the demo
│   └── test_visualization.py
├── .streamlit/config.toml      # Upload cap, XSRF, theme
├── .github/workflows/ci.yml    # Tests on 3.10-3.13, lint, Docker health check
├── Dockerfile
├── pyproject.toml              # pytest and ruff configuration
├── requirements.txt
└── requirements-dev.txt
```

## Usage

| Tab | What it does |
|-----|--------------|
| **Data Upload** | Load a file or the demo dataset; review data quality |
| **Data Explorer** | Preview rows, choose target and features, view distributions and correlations |
| **Model Training** | Inspect the preprocessing steps, then train the models selected in the sidebar |
| **Results Dashboard** | Leaderboard, comparison charts, residuals, feature importance, exports |
| **Prediction Lab** | Enter feature values and predict with any trained model |

Sidebar controls: test size (10–50%), cross-validation folds (3/5/10), scaling
method (StandardScaler / RobustScaler / None), and which models to train. The
random seed is fixed at 42 so a given dataset always produces the same split.

## Models

| Model | Type | Use case |
|-------|------|----------|
| Linear Regression | Linear | Simple, interpretable baseline |
| Ridge | Regularized linear | Multicollinear features |
| Lasso | Regularized linear | Sparse solutions / feature selection |
| ElasticNet | Regularized linear | Combined L1/L2 penalty |
| Decision Tree | Tree | Non-linear relationships, `max_depth=10` |
| Random Forest | Ensemble | Robust default, feature importance |
| Gradient Boosting | Ensemble | Usually the strongest on tabular data |
| Support Vector Regression | Kernel | Non-linear patterns, small datasets |

Models run at scikit-learn defaults apart from the seeds and depth noted above.
Per-model hyperparameter tuning is not exposed in the UI.

## Reusing an exported model

```python
import joblib
import pandas as pd

bundle = joblib.load("random_forest_bundle.joblib")

row = pd.DataFrame([{...}])[bundle["feature_cols"]]

for col, encoder in bundle["encoders"].items():
    lookup = {label: code for code, label in enumerate(encoder.classes_)}
    row[col] = lookup.get(str(row[col].iloc[0]), -1)

X = row.values
if bundle["scaler"] is not None:
    X = bundle["scaler"].transform(X)

print(bundle["model"].predict(X)[0])
```

## Development

```bash
pip install -r requirements-dev.txt

pytest                          # run the suite
pytest --cov=utils              # with coverage
ruff check .                    # lint

streamlit run app.py --server.runOnSave true
```

CI runs the tests on Python 3.10 through 3.13, lints with ruff, and builds the
Docker image and waits for its health check to pass.

## Known limitations

- The prediction band is `±1.96 × test RMSE`. It assumes homoscedastic, normally
  distributed errors and is the same width for every prediction — indicative,
  not calibrated.
- Categorical features are label-encoded, which imposes an ordinal relationship
  that does not exist for nominal columns. Tree models cope; linear models are
  affected.
- Single light theme. There is no theme toggle.
- `shap` is optional. Without it the app runs fully and the explainability
  panels say so; it is kept optional because it pulls in numba and llvmlite,
  which are a large share of the memory budget on small free-tier hosts.
- SHAP for SVR falls back to KernelExplainer, which approximates by sampling and
  is markedly slower than the exact tree and linear paths.
- Training is synchronous and in-process; very large datasets will block the UI.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

Built on Streamlit, scikit-learn and Plotly.

---

Made by [Hemant Agarwal](https://github.com/HemantAgarwal23)
