"""
AI Regression Studio - Utility Modules

This package contains utility functions for data processing, model training,
sample data generation, and visualization.
"""

from .data_processing import (
    UNSEEN_CATEGORY_CODE,
    apply_imputation,
    compute_imputation_values,
    encode_categorical,
    fit_categorical_encoders,
    handle_missing_values,
    preprocess_data,
    transform_categorical,
)
from .explainability import (
    compute_shap_values,
    select_explainer_kind,
    summarize_global_importance,
)
from .model_training import (
    export_model_bundle,
    get_model_config,
    resolve_cv_folds,
    train_model,
)
from .sample_data import generate_sample_dataset
from .ui_helpers import safe_number_input_bounds
from .visualization import (
    create_correlation_heatmap,
    create_prediction_scatter,
    create_residual_plot,
    create_shap_importance_bar,
    create_shap_waterfall,
)

__all__ = [
    # data_processing
    'UNSEEN_CATEGORY_CODE',
    'apply_imputation',
    'compute_imputation_values',
    'encode_categorical',
    'fit_categorical_encoders',
    'handle_missing_values',
    'preprocess_data',
    'transform_categorical',
    # explainability
    'compute_shap_values',
    'select_explainer_kind',
    'summarize_global_importance',
    # model_training
    'export_model_bundle',
    'get_model_config',
    'resolve_cv_folds',
    'train_model',
    # sample_data
    'generate_sample_dataset',
    # ui_helpers
    'safe_number_input_bounds',
    # visualization
    'create_correlation_heatmap',
    'create_prediction_scatter',
    'create_residual_plot',
    'create_shap_importance_bar',
    'create_shap_waterfall',
]
