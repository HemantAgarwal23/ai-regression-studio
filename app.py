"""
AI Regression Studio - Main Application

A machine learning platform for regression analysis with:
- Leakage-free automated preprocessing
- Eight regression models trained and compared side by side
- Interactive Plotly visualizations
- Live predictions with an uncertainty band
- Model and prediction export

Application logic lives in this file; reusable pieces live in the ``utils``
package, and the theme lives in ``assets/styles.css``.
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils.data_processing import UNSEEN_CATEGORY_CODE, preprocess_data
from utils.explainability import (
    MAX_SUMMARY_ROWS,
    compute_shap_values,
    select_explainer_kind,
    summarize_global_importance,
)
from utils.model_training import export_model_bundle, get_model_config, train_model
from utils.sample_data import generate_sample_dataset
from utils.ui_helpers import safe_number_input_bounds, suggest_target_column
from utils.visualization import (
    create_correlation_heatmap,
    create_prediction_scatter,
    create_residual_plot,
    create_shap_importance_bar,
    create_shap_waterfall,
)

# --- Constants -------------------------------------------------------------

APP_DIR = Path(__file__).parent
CSS_PATH = APP_DIR / "assets" / "styles.css"

# Reject oversized uploads before pandas tries to parse them into memory.
MAX_UPLOAD_MB = 50

# Scatter plots are capped so a large dataset does not stall the browser.
MAX_SCATTER_FEATURES = 4
MAX_SCATTER_POINTS = 2000

# Reproducibility: fixed so repeated runs on the same data give the same split.
RANDOM_STATE = 42

# Page configuration
st.set_page_config(
    page_title="AI Regression Studio",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)


# --- Helpers ---------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_css(path_str, mtime):
    """
    Read the stylesheet from disk.

    ``mtime`` is not used in the body; it is part of the cache key so editing
    the CSS file invalidates the cache on the next rerun.
    """
    return Path(path_str).read_text(encoding="utf-8")


def apply_theme():
    """Inject the application stylesheet, falling back silently if it is absent."""
    if not CSS_PATH.exists():
        return
    css = load_css(str(CSS_PATH), CSS_PATH.stat().st_mtime)
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def read_uploaded_file(file_bytes, filename):
    """
    Parse an uploaded CSV or Excel file into a dataframe.

    Cached on the file's bytes so switching tabs or moving a slider does not
    re-parse the same upload on every rerun.

    Raises:
    -------
    ValueError
        If the file cannot be parsed or contains no rows.
    """
    buffer = io.BytesIO(file_bytes)

    if filename.lower().endswith(".csv"):
        df = pd.read_csv(buffer)
    else:
        df = pd.read_excel(buffer)

    if df.empty:
        raise ValueError("The file parsed successfully but contains no rows.")

    return df


@st.cache_data(show_spinner=False)
def load_demo_dataset():
    """Return the bundled synthetic housing dataset."""
    return generate_sample_dataset()


@st.cache_data(show_spinner=False)
def to_csv_bytes(df):
    """Encode a dataframe as CSV bytes for a download button."""
    return df.to_csv(index=False).encode("utf-8")


def set_active_dataset(df, source):
    """
    Install a dataset as the one the rest of the app works on.

    Clears every downstream artifact, because a target column, encoders or a
    trained model carried over from the previous dataset would silently no
    longer match the columns now in play.
    """
    st.session_state.df = df
    st.session_state.data_source = source
    st.session_state.data_loaded = True

    for key in ("target_col", "feature_cols", "training_results", "X_train",
                "X_test", "y_test", "encoders", "scaler"):
        st.session_state.pop(key, None)
    st.session_state.models_trained = False


def render_data_quality(df):
    """Render the four headline data-quality metrics for a dataframe."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        cells = df.shape[0] * df.shape[1]
        missing_pct = (df.isnull().sum().sum() / cells * 100) if cells else 0.0
        st.metric("Missing Data", f"{missing_pct:.1f}%")

    with col2:
        st.metric("Numeric Features", len(df.select_dtypes(include=[np.number]).columns))

    with col3:
        st.metric("Categorical Features", len(df.select_dtypes(include=["object", "category"]).columns))

    with col4:
        st.metric("Duplicates", int(df.duplicated().sum()))



def shap_is_available():
    """
    Report whether the optional ``shap`` package can be imported.

    SHAP is not a hard requirement: the app runs fully without it and the
    explainability panels say so rather than erroring.
    """
    try:
        import shap  # noqa: F401
    except ImportError:
        return False
    return True


@st.cache_data(show_spinner=False)
def cached_shap_values(_model, X_background, X_explain, cache_key):
    """
    Compute SHAP values, memoised on ``cache_key``.

    The model is passed with a leading underscore so Streamlit does not try to
    hash the estimator itself; ``cache_key`` carries the identity that actually
    determines the result (model name and the rows being explained).
    """
    return compute_shap_values(_model, X_background, X_explain)


def render_shap_error(exc):
    """Present a SHAP failure as a message rather than a traceback."""
    st.warning(f"Could not compute SHAP values: {exc}")


apply_theme()

# Header
st.markdown('<h1 class="main-header">AI Regression Studio</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <h3>Machine Learning Platform for Regression Analysis</h3>
    <p>Upload your data, select features, train multiple models, and make predictions with confidence intervals.</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Sidebar configuration
with st.sidebar:
    st.markdown("### Control Panel")
    
    # Simplified advanced settings (no expander, random_state hardcoded)
    st.markdown("#### Advanced Settings")
    test_size = st.slider("Test Size (%)", 10, 50, 20, 5, help="Percentage of data to use for testing")
    cv_folds = st.selectbox("Cross-Validation Folds", [3, 5, 10], index=1, help="Number of folds for cross-validation")
    scaling_method = st.selectbox("Scaling Method", ["StandardScaler", "RobustScaler", "None"], help="Method for feature scaling")
    
    # Fixed seed so a given dataset always produces the same split and models.
    random_state = RANDOM_STATE

    st.markdown("---")
    
    # Model selection
    st.markdown("### Model Selection")
    model_categories = {
        "Linear Models": ["Linear Regression", "Ridge", "Lasso", "ElasticNet"],
        "Tree Models": ["Decision Tree", "Random Forest", "Gradient Boosting"],
        "Advanced": ["Support Vector Regression"]
    }
    
    selected_models = []
    for category, models in model_categories.items():
        st.markdown(f"**{category}**")
        for model in models:
            if st.checkbox(model, key=f"model_{model}", value=model in ["Linear Regression", "Random Forest"]):
                selected_models.append(model)

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Upload", 
    "Data Explorer", 
    "Model Training", 
    "Results Dashboard", 
    "Prediction Lab"
])

# ============================================================================
# TAB 1: Data Upload
# ============================================================================
with tab1:
    st.markdown('<div class="section-header">Data Upload & Processing</div>', unsafe_allow_html=True)

    upload_col, demo_col = st.columns([2, 1])

    with upload_col:
        uploaded_file = st.file_uploader(
            "Upload your dataset",
            type=["xlsx", "csv", "xls"],
            help=f"Supported formats: CSV, Excel (.xlsx, .xls). Maximum {MAX_UPLOAD_MB} MB.",
        )

    with demo_col:
        st.markdown("**No dataset handy?**")
        st.caption("Load a synthetic housing dataset with categorical columns and missing values already in it.")
        if st.button("Load demo dataset", use_container_width=True):
            set_active_dataset(load_demo_dataset(), "demo")
            st.success("Demo dataset loaded. Continue in the Data Explorer tab.")

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        size_mb = len(file_bytes) / (1024 * 1024)

        if size_mb > MAX_UPLOAD_MB:
            st.error(
                f"File is {size_mb:.1f} MB, above the {MAX_UPLOAD_MB} MB limit. "
                "Upload a sample of the data instead."
            )
        else:
            try:
                with st.spinner("Loading data..."):
                    df = read_uploaded_file(file_bytes, uploaded_file.name)

                # Only reset downstream state when the dataset actually changed,
                # otherwise every rerun would wipe the user's trained models.
                if st.session_state.get("data_source") != uploaded_file.name:
                    set_active_dataset(df, uploaded_file.name)
            except Exception as exc:
                st.error(f"Could not read '{uploaded_file.name}': {exc}")
                st.info("Check that the file is a valid CSV or Excel workbook with a header row.")

    if st.session_state.get("data_loaded"):
        df = st.session_state.df
        source_label = (
            "Demo dataset" if st.session_state.get("data_source") == "demo"
            else st.session_state.get("data_source", "Uploaded file")
        )

        st.markdown(f"""
        <div class="success-box">
            <h4>Data Loaded Successfully</h4>
            <p><strong>Source:</strong> {source_label}</p>
            <p><strong>Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns</p>
            <p><strong>Memory:</strong> {df.memory_usage(deep=True).sum() / 1024:.2f} KB</p>
        </div>
        """, unsafe_allow_html=True)

        render_data_quality(df)

        st.download_button(
            "Download this dataset as CSV",
            data=to_csv_bytes(df),
            file_name="ai_regression_studio_dataset.csv",
            mime="text/csv",
        )

# ============================================================================
# TAB 2: Data Explorer
# ============================================================================
with tab2:
    if st.session_state.data_loaded:
        st.markdown('<div class="section-header">Data Explorer</div>', unsafe_allow_html=True)
        
        df = st.session_state.df
        
        # Data preview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(20), use_container_width=True)
        
        with col2:
            st.subheader("Target & Features Selection")
            
            # Target variable selection
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                # Smart target suggestion based on column names.
                suggested = suggest_target_column(numeric_cols)

                target_col = st.selectbox(
                    "Target Variable",
                    numeric_cols,
                    index=numeric_cols.index(suggested) if suggested else 0,
                    help="Select the variable you want to predict"
                )
                
                # Feature selection
                available_features = [col for col in df.columns if col != target_col]
                
                if st.checkbox("Smart Feature Selection"):
                    # Calculate feature importance preview using correlation
                    X_temp = df[available_features].select_dtypes(include=[np.number])
                    if len(X_temp.columns) > 0:
                        try:
                            correlations = X_temp.corrwith(df[target_col]).abs().sort_values(ascending=False)
                            top_features = correlations.head(min(10, len(correlations))).index.tolist()
                            
                            feature_cols = st.multiselect(
                                "Select Features",
                                available_features,
                                default=top_features,
                                help="Pre-selected based on correlation with target"
                            )
                        except (TypeError, ValueError) as exc:
                            # Correlation fails on non-numeric or constant columns;
                            # fall back to an unfiltered picker rather than hiding it.
                            st.caption(f"Correlation ranking unavailable ({exc}). Showing all features.")
                            feature_cols = st.multiselect("Select Features", available_features)
                    else:
                        feature_cols = st.multiselect("Select Features", available_features)
                else:
                    feature_cols = st.multiselect("Select Features", available_features)
                
                if target_col and feature_cols:
                    st.session_state.target_col = target_col
                    st.session_state.feature_cols = feature_cols
                    
                    st.markdown("""
                    <div class="success-box">
                        <p>Configuration saved!</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Data visualization
        if 'target_col' in st.session_state and 'feature_cols' in st.session_state:
            st.markdown('<div class="section-header">Data Visualization</div>', unsafe_allow_html=True)
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Target distribution
                fig_target = px.histogram(
                    df, 
                    x=st.session_state.target_col,
                    title=f"Target Distribution: {st.session_state.target_col}",
                    template="plotly_white"
                )
                fig_target.update_layout(showlegend=False)
                st.plotly_chart(fig_target, use_container_width=True)
            
            with viz_col2:
                # Correlation heatmap using utility function
                numeric_features = [
                    col for col in st.session_state.feature_cols 
                    if col in df.select_dtypes(include=[np.number]).columns
                ]
                if len(numeric_features) > 1:
                    corr_columns = numeric_features + [st.session_state.target_col]
                    fig_corr = create_correlation_heatmap(df, corr_columns)
                    st.plotly_chart(fig_corr, use_container_width=True)
            
            # Feature vs Target scatter plots
            st.subheader("Feature-Target Relationships")
            numeric_features = [
                col for col in st.session_state.feature_cols 
                if col in df.select_dtypes(include=[np.number]).columns
            ]
            
            if len(numeric_features) >= 2:
                # Plotting every row of a large dataset freezes the browser, and a
                # random sample shows the same relationship just as clearly.
                scatter_df = df
                if len(df) > MAX_SCATTER_POINTS:
                    scatter_df = df.sample(MAX_SCATTER_POINTS, random_state=RANDOM_STATE)
                    st.caption(
                        f"Showing a random sample of {MAX_SCATTER_POINTS:,} of {len(df):,} rows."
                    )

                scatter_cols = st.columns(2)
                for i, feature in enumerate(numeric_features[:MAX_SCATTER_FEATURES]):
                    with scatter_cols[i % 2]:
                        fig_scatter = px.scatter(
                            scatter_df,
                            x=feature,
                            y=st.session_state.target_col,
                            title=f"{feature} vs {st.session_state.target_col}",
                            template="plotly_white",
                            trendline="ols"
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Select at least two numeric features to see feature-target scatter plots.")
    else:
        st.info("Load a dataset in the Data Upload tab to explore it here.")

# ============================================================================
# TAB 3: Model Training
# ============================================================================
with tab3:
    if st.session_state.data_loaded and 'target_col' in st.session_state:
        st.markdown('<div class="section-header">Model Training Center</div>', unsafe_allow_html=True)
        
        if not selected_models:
            st.warning("Please select at least one model from the sidebar!")
        else:
            df = st.session_state.df
            target_col = st.session_state.target_col
            feature_cols = st.session_state.feature_cols
            
            # Data preprocessing using utility function
            st.subheader("Data Preprocessing Pipeline")
            
            prepared = None
            with st.expander("Preprocessing Steps", expanded=True):
                st.caption(
                    "Imputation, encoding and scaling are all fitted on the training "
                    "split only, so no test-set information reaches the models."
                )
                try:
                    prepared = preprocess_data(
                        df=df,
                        feature_cols=feature_cols,
                        target_col=target_col,
                        test_size=test_size / 100,
                        random_state=random_state,
                        scaling_method=scaling_method,
                        handle_missing=True,
                    )
                except ValueError as exc:
                    st.error(f"Preprocessing failed: {exc}")
                except Exception as exc:
                    st.error(f"Unexpected preprocessing error: {exc}")

                if prepared is not None:
                    X_train, X_test, y_train, y_test, scaler, encoders, preprocessing_steps = prepared
                    for i, step in enumerate(preprocessing_steps, 1):
                        st.write(f"{i}. {step}")

            # Model training
            if prepared is not None and st.button("Launch Model Training", type="primary", use_container_width=True):
                st.markdown('<div class="section-header">Training Progress</div>', unsafe_allow_html=True)
                
                model_config = get_model_config(random_state=random_state)
                models = {name: model_config[name] for name in selected_models if name in model_config}

                results = {}
                failures = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, (name, model) in enumerate(models.items()):
                    status_text.text(f"Training {name}...")

                    try:
                        results[name] = train_model(
                            model=model,
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            cv_folds=cv_folds,
                        )
                    except Exception as exc:
                        failures.append(name)
                        st.error(f"Error training {name}: {exc}")

                    progress_bar.progress((i + 1) / len(models))

                if not results:
                    status_text.empty()
                    st.error("No model trained successfully. Adjust the feature selection or test size and retry.")
                else:
                    st.session_state.training_results = results
                    # X_train is kept as the SHAP background distribution:
                    # attributions are measured relative to typical training input.
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.encoders = encoders
                    st.session_state.scaler = scaler
                    st.session_state.models_trained = True

                    trained_note = f"{len(results)} of {len(models)} models trained"
                    if failures:
                        trained_note += f"; failed: {', '.join(failures)}"
                    status_text.text(trained_note)

                    skipped_cv = [n for n, r in results.items() if not r.get('cv_folds_used')]
                    if skipped_cv:
                        st.warning(
                            "Cross-validation was skipped for "
                            f"{', '.join(skipped_cv)} — the training split is too small."
                        )

                    st.markdown(f"""
                    <div class="success-box">
                        <h4>Training Complete</h4>
                        <p>{trained_note}. Check the Results Dashboard for detailed analysis.</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("Load a dataset and pick a target plus features in the Data Explorer tab first.")

# ============================================================================
# TAB 4: Results Dashboard
# ============================================================================
with tab4:
    if st.session_state.get('models_trained', False):
        st.markdown('<div class="section-header">Results Dashboard</div>', unsafe_allow_html=True)
        
        results = st.session_state.training_results
        
        # Model comparison table
        st.subheader("Model Leaderboard")
        
        # Ranking is done on the raw floats. Sorting the pre-formatted strings
        # compares them lexicographically, which picks the wrong champion as
        # soon as scores are negative or reach double digits.
        comparison_df = pd.DataFrame([
            {
                'Model': name,
                'R² Score': result['r2'],
                'RMSE': result['rmse'],
                'MAE': result['mae'],
                'CV Mean': result['cv_mean'],
                'CV Std': result['cv_std'],
            }
            for name, result in results.items()
        ]).sort_values('R² Score', ascending=False, ignore_index=True)

        st.dataframe(
            comparison_df.style.format({
                'R² Score': '{:.4f}',
                'RMSE': '{:.4f}',
                'MAE': '{:.4f}',
                'CV Mean': '{:.4f}',
                'CV Std': '{:.4f}',
            }),
            use_container_width=True,
        )

        best_model_name = comparison_df.iloc[0]['Model']
        st.markdown(f"""
        <div class="success-box">
            <h4>Champion Model: {best_model_name}</h4>
            <p>Best performing model based on R² score</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Performance comparison charts
        st.subheader("Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # R² Score comparison
            r2_scores = [result['r2'] for result in results.values()]
            model_names = list(results.keys())
            
            fig_r2 = px.bar(
                x=model_names,
                y=r2_scores,
                title="R² Score Comparison",
                template="plotly_white",
                color=r2_scores,
                color_continuous_scale="Viridis"
            )
            fig_r2.update_layout(showlegend=False, xaxis_title="Model", yaxis_title="R² Score")
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            # RMSE comparison
            rmse_scores = [result['rmse'] for result in results.values()]
            
            fig_rmse = px.bar(
                x=model_names,
                y=rmse_scores,
                title="RMSE Comparison (Lower is Better)",
                template="plotly_white",
                color=rmse_scores,
                color_continuous_scale="Reds_r"
            )
            fig_rmse.update_layout(showlegend=False, xaxis_title="Model", yaxis_title="RMSE")
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        # Detailed analysis for best model
        st.subheader(f"Detailed Analysis: {best_model_name}")
        
        best_result = results[best_model_name]
        y_test = st.session_state.y_test
        y_pred = best_result['predictions']
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            # Actual vs Predicted using utility function
            fig_scatter = create_prediction_scatter(y_test, y_pred)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with analysis_col2:
            # Residuals plot using utility function
            fig_residuals = create_residual_plot(y_test, y_pred)
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        # Feature importance (if available)
        if hasattr(best_result['model'], 'feature_importances_'):
            st.subheader("Feature Importance Analysis")
            
            feature_importance = pd.DataFrame({
                'Feature': st.session_state.feature_cols,
                'Importance': best_result['model'].feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig_importance = px.bar(
                feature_importance,
                x='Importance',
                y='Feature',
                orientation='h',
                title=f"Feature Importance - {best_model_name}",
                template="plotly_white"
            )

            st.plotly_chart(fig_importance, use_container_width=True)

        # --- Global explainability -------------------------------------------
        st.subheader("Explainability (SHAP)")

        if not shap_is_available():
            st.info(
                "Install the optional `shap` package (`pip install shap`) to rank "
                "features by their mean impact on predictions."
            )
        else:
            kind = select_explainer_kind(best_result['model'])
            st.caption(
                "Mean absolute SHAP value per feature, in the target's own units. "
                "Unlike a tree's built-in importances this works for every model "
                f"type. Explainer selected for {best_model_name}: {kind}."
            )
            if kind == 'kernel':
                st.caption(
                    "This model has no exact SHAP algorithm, so values are "
                    "approximated by sampling and will take noticeably longer."
                )

            if st.button("Compute SHAP importance", key="shap_global"):
                X_train_bg = st.session_state.X_train
                X_explain = st.session_state.X_test[:MAX_SUMMARY_ROWS]

                with st.spinner(f"Explaining {len(X_explain)} test rows..."):
                    try:
                        shap_result = cached_shap_values(
                            best_result['model'],
                            X_train_bg,
                            X_explain,
                            cache_key=(best_model_name, "global", len(X_explain)),
                        )
                        ranked = summarize_global_importance(
                            shap_result, st.session_state.feature_cols
                        )
                        st.session_state.shap_global_ranked = ranked
                    except Exception as exc:
                        render_shap_error(exc)

            if st.session_state.get("shap_global_ranked"):
                ranked = st.session_state.shap_global_ranked
                st.plotly_chart(
                    create_shap_importance_bar(ranked), use_container_width=True
                )
                st.caption(
                    f"Computed over {min(MAX_SUMMARY_ROWS, len(st.session_state.X_test))} "
                    "test rows against a sample of the training set."
                )

        # --- Exports ---------------------------------------------------------
        st.subheader("Export")
        st.caption(
            "The model bundle carries the fitted scaler, encoders and column order "
            "alongside the estimator, so it can be reloaded and used as-is."
        )

        export_col1, export_col2, export_col3 = st.columns(3)

        with export_col1:
            st.download_button(
                "Leaderboard (CSV)",
                data=to_csv_bytes(comparison_df),
                file_name="model_leaderboard.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with export_col2:
            predictions_df = pd.DataFrame({
                'actual': np.asarray(y_test),
                'predicted': np.asarray(y_pred),
                'residual': np.asarray(y_test) - np.asarray(y_pred),
            })
            st.download_button(
                "Test predictions (CSV)",
                data=to_csv_bytes(predictions_df),
                file_name=f"predictions_{best_model_name.replace(' ', '_').lower()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with export_col3:
            bundle = export_model_bundle(
                model=best_result['model'],
                feature_cols=st.session_state.feature_cols,
                target_col=st.session_state.target_col,
                scaler=st.session_state.scaler,
                encoders=st.session_state.encoders,
                model_name=best_model_name,
                metrics_summary={
                    'r2': best_result['r2'],
                    'rmse': best_result['rmse'],
                    'mae': best_result['mae'],
                },
            )
            st.download_button(
                "Model bundle (.joblib)",
                data=bundle,
                file_name=f"{best_model_name.replace(' ', '_').lower()}_bundle.joblib",
                mime="application/octet-stream",
                use_container_width=True,
            )
    else:
        st.info("Train at least one model in the Model Training tab to see results here.")

# ============================================================================
# TAB 5: Prediction Lab
# ============================================================================
with tab5:
    if st.session_state.get('models_trained', False):
        st.markdown('<div class="section-header">Prediction Lab</div>', unsafe_allow_html=True)
        
        results = st.session_state.training_results
        
        # Model selection for prediction
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Select Prediction Model")
            
            model_options = list(results.keys())
            selected_model = st.selectbox(
                "Choose Model",
                model_options,
                index=0,
                help="Select which trained model to use for predictions"
            )
            
            # Display selected model performance
            selected_result = results[selected_model]
            
            st.markdown(f"""
            <div class="info-box">
                <h4>Model Performance</h4>
                <p><strong>R² Score:</strong> {selected_result['r2']:.4f}</p>
                <p><strong>RMSE:</strong> {selected_result['rmse']:.4f}</p>
                <p><strong>CV Score:</strong> {selected_result['cv_mean']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Input Features")
            
            # Create prediction input form
            new_data = {}
            feature_cols = st.session_state.feature_cols
            encoders = st.session_state.encoders
            df = st.session_state.df
            
            # Organize inputs in columns
            input_cols = st.columns(3)
            
            for i, col in enumerate(feature_cols):
                with input_cols[i % 3]:
                    if col in encoders:
                        # Categorical feature
                        available_values = list(encoders[col].classes_)
                        new_data[col] = st.selectbox(
                            col,
                            available_values,
                            key=f"pred_input_{col}",
                            help=f"Select value for {col}"
                        )
                    elif col in df.columns:
                        # Numerical feature. Bounds are widened defensively so a
                        # constant or single-row column cannot produce a
                        # degenerate range that number_input rejects.
                        low, high, default, step = safe_number_input_bounds(df[col])

                        new_data[col] = st.number_input(
                            col,
                            min_value=low,
                            max_value=high,
                            value=default,
                            step=step,
                            key=f"pred_input_{col}",
                            help=f"Allowed range: {low:.2f} - {high:.2f}, dataset mean: {default:.2f}",
                        )
        
        # Prediction button and results
        st.markdown("---")
        
        if st.button("Make Prediction", type="primary", use_container_width=True):
            try:
                # Prepare input data for prediction
                input_df = pd.DataFrame([new_data])
                
                # Encode categorical variables using the vocabularies learned at
                # training time. Values outside those vocabularies fall back to
                # the shared unseen-category code instead of raising.
                for col in feature_cols:
                    if col in encoders:
                        lookup = {label: code for code, label in enumerate(encoders[col].classes_)}
                        input_df[col] = lookup.get(str(new_data[col]), UNSEEN_CATEGORY_CODE)

                # Column order must match the order the model was fitted on.
                input_df = input_df[feature_cols]
                input_array = input_df.values

                scaler = st.session_state.scaler
                if scaler is not None:
                    input_array = scaler.transform(input_array)

                model = selected_result['model']
                prediction = model.predict(input_array)[0]

                # RMSE stands in for the prediction's standard error. This is an
                # approximation: it assumes errors are homoscedastic and normal,
                # so treat the band as indicative rather than calibrated.
                rmse = selected_result['rmse']
                confidence_interval = 1.96 * rmse

                # Display results
                st.markdown("### Prediction Results")
                
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.markdown(f"""
                    <div class="success-box">
                        <h3>Predicted Value</h3>
                        <h2 style="color: #28a745; margin: 1rem 0;">{prediction:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with result_col2:
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>Uncertainty Band (±1.96 × RMSE)</h4>
                        <p><strong>Lower:</strong> {prediction - confidence_interval:.2f}</p>
                        <p><strong>Upper:</strong> {prediction + confidence_interval:.2f}</p>
                        <p><strong>Range:</strong> ±{confidence_interval:.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.caption(
                    "The band is derived from the model's test-set RMSE, not from a "
                    "per-sample variance estimate, so it is the same width for every prediction."
                )

                # --- Per-prediction attribution ---------------------------
                st.markdown("### Why this prediction?")

                if not shap_is_available():
                    st.info(
                        "Install the optional `shap` package (`pip install shap`) "
                        "to see which features drove this specific prediction."
                    )
                else:
                    with st.spinner("Explaining this prediction..."):
                        try:
                            shap_result = cached_shap_values(
                                model,
                                st.session_state.X_train,
                                input_array,
                                cache_key=(
                                    selected_model,
                                    "single",
                                    tuple(np.asarray(input_array).ravel().tolist()),
                                ),
                            )

                            fig_waterfall = create_shap_waterfall(
                                feature_names=feature_cols,
                                shap_values=shap_result.values[0],
                                base_value=shap_result.base_value,
                                # Show the values the user actually entered, not
                                # the scaled numbers the model consumed.
                                feature_values=[new_data[c] for c in feature_cols],
                                title=f"Feature contributions - {selected_model}",
                            )
                            st.plotly_chart(fig_waterfall, use_container_width=True)

                            st.caption(
                                f"Starts from the model's average output over the "
                                f"training data ({shap_result.base_value:,.2f}) and "
                                "adds each feature's contribution to reach this "
                                "prediction. Red pushes the value up, blue pulls it "
                                f"down. Explainer: {shap_result.explainer_kind}."
                            )
                        except Exception as exc:
                            render_shap_error(exc)

                # Global feature importance for tree-based models. This describes
                # the model as a whole, not this particular row's contributions.
                if hasattr(model, 'feature_importances_'):
                    st.markdown("### Model Feature Importance")

                    feature_contributions = pd.DataFrame({
                        'Feature': feature_cols,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)

                    st.dataframe(feature_contributions, use_container_width=True)

                    fig_contrib = px.bar(
                        feature_contributions,
                        x='Feature',
                        y='Importance',
                        title=f"Feature Importance - {selected_model}",
                        template="plotly_white"
                    )
                    fig_contrib.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_contrib, use_container_width=True)

            except Exception as exc:
                st.error(f"Error making prediction: {exc}")
                st.info("Please ensure all input fields are filled correctly.")
    else:
        st.info("Train at least one model in the Model Training tab to make predictions here.")

