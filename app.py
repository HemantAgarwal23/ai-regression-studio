import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.feature_selection import mutual_info_regression, f_regression
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for modern UI
st.set_page_config(
    page_title="AI Regression Studio", 
    layout="wide", 
    page_icon="🤖",
    initial_sidebar_state="expanded"
)

# Initialize theme in session state
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Theme switcher function
def toggle_theme():
    st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'

# Dynamic CSS based on theme
def get_theme_css(theme):
    if theme == 'dark':
        return """
        <style>
            /* Dark theme styles */
            .stApp {
                background: linear-gradient(135deg, #1e1e2e 0%, #2d1b69 100%);
                color: #ffffff;
            }
            
            .main-header {
                font-size: 3rem;
                font-weight: 700;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                margin-bottom: 2rem;
                text-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
            }
            
            .section-header {
                font-size: 1.5rem;
                font-weight: 600;
                color: #ffffff;
                border-bottom: 3px solid #667eea;
                padding-bottom: 0.5rem;
                margin: 2rem 0 1rem 0;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem;
                border-radius: 15px;
                color: white;
                text-align: center;
                box-shadow: 0 8px 32px 0 rgba(102, 126, 234, 0.4);
                margin: 0.5rem 0;
                border: 1px solid rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
            }
            
            .info-box {
                background: linear-gradient(135deg, #4c6ef5 0%, #364fc7 100%);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                margin: 1rem 0;
                box-shadow: 0 4px 16px rgba(76, 110, 245, 0.3);
                border: 1px solid rgba(255,255,255,0.1);
            }
            
            .success-box {
                background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                margin: 1rem 0;
                box-shadow: 0 4px 16px rgba(81, 207, 102, 0.3);
                border: 1px solid rgba(255,255,255,0.1);
            }
            
            .warning-box {
                background: linear-gradient(135deg, #ff8787 0%, #fa5252 100%);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                margin: 1rem 0;
                box-shadow: 0 4px 16px rgba(255, 135, 135, 0.3);
                border: 1px solid rgba(255,255,255,0.1);
            }
            
            .stTabs [data-baseweb="tab-list"] {
                gap: 2px;
                background: rgba(255,255,255,0.05);
                padding: 0.5rem;
                border-radius: 15px;
                backdrop-filter: blur(10px);
            }
            
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                padding-left: 20px;
                padding-right: 20px;
                background: linear-gradient(135deg, #495057 0%, #343a40 100%);
                border-radius: 10px;
                color: white;
                font-weight: 600;
                border: 1px solid rgba(255,255,255,0.1);
                transition: all 0.3s ease;
            }
            
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
                transform: translateY(-2px);
            }
            
            .stSidebar > div {
                background: linear-gradient(180deg, #2c2c54 0%, #1a1a2e 100%);
            }
            
            .theme-toggle {
                position: fixed;
                top: 1rem;
                right: 1rem;
                z-index: 1000;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                border-radius: 50px;
                padding: 0.5rem 1rem;
                color: white;
                font-weight: 600;
                box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
                transition: all 0.3s ease;
                cursor: pointer;
            }
            
            .theme-toggle:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
            }
        </style>
        """
    else:  # light theme
        return """
        <style>
            /* Light theme styles */
            .stApp {
                background: linear-gradient(135deg, #f8f9ff 0%, #e6f3ff 100%);
                color: #2d3748;
            }
            
            .main-header {
                font-size: 3rem;
                font-weight: 700;
                background: linear-gradient(90deg, #4c6ef5 0%, #5f3dc4 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-align: center;
                margin-bottom: 2rem;
                text-shadow: 0 0 20px rgba(76, 110, 245, 0.1);
            }
            
            .section-header {
                font-size: 1.5rem;
                font-weight: 600;
                color: #2d3748;
                border-bottom: 3px solid #4c6ef5;
                padding-bottom: 0.5rem;
                margin: 2rem 0 1rem 0;
            }
            
            .metric-card {
                background: linear-gradient(135deg, #4c6ef5 0%, #5f3dc4 100%);
                padding: 1.5rem;
                border-radius: 15px;
                color: white;
                text-align: center;
                box-shadow: 0 8px 32px 0 rgba(76, 110, 245, 0.2);
                margin: 0.5rem 0;
                border: 1px solid rgba(76, 110, 245, 0.1);
            }
            
            .info-box {
                background: linear-gradient(135deg, #339af0 0%, #228be6 100%);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                margin: 1rem 0;
                box-shadow: 0 4px 16px rgba(51, 154, 240, 0.2);
                border: 1px solid rgba(51, 154, 240, 0.1);
            }
            
            .success-box {
                background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                margin: 1rem 0;
                box-shadow: 0 4px 16px rgba(81, 207, 102, 0.2);
                border: 1px solid rgba(81, 207, 102, 0.1);
            }
            
            .warning-box {
                background: linear-gradient(135deg, #ff922b 0%, #fd7e14 100%);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                margin: 1rem 0;
                box-shadow: 0 4px 16px rgba(255, 146, 43, 0.2);
                border: 1px solid rgba(255, 146, 43, 0.1);
            }
            
            .stTabs [data-baseweb="tab-list"] {
                gap: 2px;
                background: rgba(76, 110, 245, 0.05);
                padding: 0.5rem;
                border-radius: 15px;
            }
            
            .stTabs [data-baseweb="tab"] {
                height: 50px;
                padding-left: 20px;
                padding-right: 20px;
                background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%);
                border-radius: 10px;
                color: #495057;
                font-weight: 600;
                border: 1px solid rgba(76, 110, 245, 0.1);
                transition: all 0.3s ease;
            }
            
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, #4c6ef5 0%, #5f3dc4 100%);
                color: white;
                box-shadow: 0 4px 16px rgba(76, 110, 245, 0.3);
                transform: translateY(-2px);
            }
            
            .stSidebar > div {
                background: linear-gradient(180deg, #ffffff 0%, #f8f9ff 100%);
                border-right: 1px solid rgba(76, 110, 245, 0.1);
            }
            
            .theme-toggle {
                position: fixed;
                top: 1rem;
                right: 1rem;
                z-index: 1000;
                background: linear-gradient(135deg, #4c6ef5 0%, #5f3dc4 100%);
                border: none;
                border-radius: 50px;
                padding: 0.5rem 1rem;
                color: white;
                font-weight: 600;
                box-shadow: 0 4px 16px rgba(76, 110, 245, 0.3);
                transition: all 0.3s ease;
                cursor: pointer;
            }
            
            .theme-toggle:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(76, 110, 245, 0.5);
            }
            
            /* Light theme specific overrides */
            .stMarkdown {
                color: #2d3748;
            }
            
            /* Ensure proper contrast for light theme */
            div[data-testid="metric-container"] {
                background: rgba(255, 255, 255, 0.8);
                border: 1px solid rgba(76, 110, 245, 0.1);
                border-radius: 10px;
                padding: 1rem;
                backdrop-filter: blur(10px);
            }
        </style>
        """

# Apply theme CSS
st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

# Theme toggle button (floating)
theme_icon = "🌙" if st.session_state.theme == 'light' else "☀️"
theme_text = "Dark" if st.session_state.theme == 'light' else "Light"

# Add theme toggle button in a container at the top
col1, col2, col3 = st.columns([6, 1, 1])
with col3:
    if st.button(f"{theme_icon} {theme_text}", key="theme_toggle", help=f"Switch to {theme_text.lower()} theme"):
        toggle_theme()
        st.rerun()

# Header
st.markdown('<h1 class="main-header">🤖 AI Regression Studio</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <h3>🚀 Next-Generation ML Platform</h3>
    <p>Advanced regression modeling with automated feature engineering, model comparison, and intelligent insights</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False

# Update plotly template based on theme
plotly_template = "plotly_dark" if st.session_state.theme == 'dark' else "plotly_white"

# Sidebar configuration
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        random_state = st.number_input("Random State", value=42, min_value=0, max_value=1000)
        test_size = st.slider("Test Size (%)", 10, 50, 20, 5)
        cv_folds = st.selectbox("Cross-Validation Folds", [3, 5, 10], index=1)
        scaling_method = st.selectbox("Scaling Method", ["StandardScaler", "RobustScaler", "None"])
    
    # Model selection with advanced options
    st.markdown("### 🤖 Model Arsenal")
    model_categories = {
        "Linear Models": ["Linear Regression", "Ridge", "Lasso", "ElasticNet"],
        "Tree Models": ["Decision Tree", "Random Forest", "Gradient Boosting"],
        "Advanced": ["Support Vector Regression"]
    }
    
    selected_models = []
    for category, models in model_categories.items():
        st.write(f"**{category}**")
        for model in models:
            if st.checkbox(model, key=f"model_{model}", value=model in ["Linear Regression", "Random Forest"]):
                selected_models.append(model)

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📤 Data Upload", "🔍 Data Explorer", "🤖 Model Training", "📊 Results Dashboard", "🔮 Prediction Lab"])

with tab1:
    st.markdown('<div class="section-header">📤 Data Upload & Processing</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Drop your dataset here", 
        type=["xlsx", "csv", "xls"],
        help="Supported formats: CSV, Excel"
    )
    
    if uploaded_file:
        try:
            # Load data with progress bar
            with st.spinner("🔄 Loading and preprocessing data..."):
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.df = df
                st.session_state.data_loaded = True
            
            st.markdown(f"""
            <div class="success-box">
                <h4>✅ Data Loaded Successfully!</h4>
                <p><strong>Shape:</strong> {df.shape[0]} rows × {df.shape[1]} columns</p>
                <p><strong>Size:</strong> {df.memory_usage(deep=True).sum() / 1024:.2f} KB</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick data quality check
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{missing_pct:.1f}%</h3>
                    <p>Missing Data</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{numeric_cols}</h3>
                    <p>Numeric Features</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                cat_cols = len(df.select_dtypes(include=['object', 'category']).columns)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{cat_cols}</h3>
                    <p>Categorical Features</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                duplicates = df.duplicated().sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{duplicates}</h3>
                    <p>Duplicates</p>
                </div>
                """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")

with tab2:
    if st.session_state.data_loaded:
        st.markdown('<div class="section-header">🔍 Intelligent Data Explorer</div>', unsafe_allow_html=True)
        
        df = st.session_state.df
        
        # Data preview with interactive filtering
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 Dataset Preview")
            
            # Interactive filtering
            if st.checkbox("🔍 Enable interactive filtering"):
                filter_col = st.selectbox("Filter by column", df.columns)
                if df[filter_col].dtype in ['object', 'category']:
                    filter_values = st.multiselect("Select values", df[filter_col].unique())
                    if filter_values:
                        df_filtered = df[df[filter_col].isin(filter_values)]
                    else:
                        df_filtered = df
                else:
                    min_val, max_val = float(df[filter_col].min()), float(df[filter_col].max())
                    filter_range = st.slider("Select range", min_val, max_val, (min_val, max_val))
                    df_filtered = df[(df[filter_col] >= filter_range[0]) & (df[filter_col] <= filter_range[1])]
                
                st.dataframe(df_filtered.head(20), use_container_width=True)
            else:
                st.dataframe(df.head(20), use_container_width=True)
        
        with col2:
            st.subheader("🎯 Target & Features Selection")
            
            # Smart target suggestion
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                # Suggest target based on column names and data characteristics
                suggested_targets = []
                common_target_keywords = ['price', 'cost', 'value', 'amount', 'target', 'y', 'label', 'outcome']
                
                for col in numeric_cols:
                    if any(keyword in col.lower() for keyword in common_target_keywords):
                        suggested_targets.append(col)
                
                target_col = st.selectbox(
                    "🎯 Target Variable", 
                    numeric_cols,
                    index=0 if not suggested_targets else numeric_cols.index(suggested_targets[0]),
                    help="💡 AI suggested the best target variable"
                )
                
                # Feature selection with importance preview
                available_features = [col for col in df.columns if col != target_col]
                
                if st.checkbox("🧠 Smart Feature Selection"):
                    # Calculate feature importance preview
                    X_temp = df[available_features].select_dtypes(include=[np.number])
                    if len(X_temp.columns) > 0:
                        try:
                            # Simple correlation-based feature importance
                            correlations = X_temp.corrwith(df[target_col]).abs().sort_values(ascending=False)
                            top_features = correlations.head(min(10, len(correlations))).index.tolist()
                            
                            feature_cols = st.multiselect(
                                "Select Features",
                                available_features,
                                default=top_features,
                                help="🎯 Pre-selected based on correlation with target"
                            )
                        except:
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
                        <p>✅ Configuration saved!</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Advanced data visualization
        if 'target_col' in st.session_state and 'feature_cols' in st.session_state:
            st.markdown('<div class="section-header">📊 Advanced Data Visualization</div>', unsafe_allow_html=True)
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Target distribution
                fig_target = px.histogram(df, x=st.session_state.target_col, 
                                        title=f"Target Distribution: {st.session_state.target_col}",
                                        template="plotly_white")
                fig_target.update_layout(showlegend=False)
                st.plotly_chart(fig_target, use_container_width=True)
            
            with viz_col2:
                # Correlation heatmap
                numeric_features = [col for col in st.session_state.feature_cols if col in df.select_dtypes(include=[np.number]).columns]
                if len(numeric_features) > 1:
                    corr_data = df[numeric_features + [st.session_state.target_col]].corr()
                    fig_corr = px.imshow(corr_data, 
                                       title="Feature Correlation Matrix",
                                       template=plotly_template,
                                       color_continuous_scale="RdBu")
                    st.plotly_chart(fig_corr, use_container_width=True)
            
            # Feature vs Target scatter plots
            st.subheader("🎯 Feature-Target Relationships")
            
            numeric_features = [col for col in st.session_state.feature_cols if col in df.select_dtypes(include=[np.number]).columns]
            
            if len(numeric_features) >= 2:
                scatter_cols = st.columns(2)
                for i, feature in enumerate(numeric_features[:4]):  # Show up to 4 scatter plots
                    with scatter_cols[i % 2]:
                        fig_scatter = px.scatter(df, x=feature, y=st.session_state.target_col,
                                               title=f"{feature} vs {st.session_state.target_col}",
                                               template=plotly_template,
                                               trendline="ols")
                        st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    if st.session_state.data_loaded and 'target_col' in st.session_state:
        st.markdown('<div class="section-header">🤖 AI Model Training Center</div>', unsafe_allow_html=True)
        
        if not selected_models:
            st.warning("⚠️ Please select at least one model from the sidebar!")
        else:
            df = st.session_state.df
            target_col = st.session_state.target_col
            feature_cols = st.session_state.feature_cols
            
            # Data preprocessing pipeline
            st.subheader("🔧 Data Preprocessing Pipeline")
            
            with st.expander("📋 Preprocessing Steps", expanded=True):
                X = df[feature_cols].copy()
                Y = df[target_col].copy()
                
                preprocessing_steps = []
                
                # Handle missing values
                if X.isnull().sum().sum() > 0:
                    st.info("🔄 Handling missing values...")
                    preprocessing_steps.append("Missing value imputation")
                    for col in X.columns:
                        if X[col].dtype in ['object', 'category']:
                            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')
                        else:
                            X[col] = X[col].fillna(X[col].median())
                
                # Encode categorical variables
                encoders = {}
                categorical_cols = X.select_dtypes(include=['object', 'category']).columns
                
                if len(categorical_cols) > 0:
                    st.info(f"🏷️ Encoding {len(categorical_cols)} categorical features...")
                    preprocessing_steps.append(f"Label encoding for {len(categorical_cols)} features")
                    
                    for col in categorical_cols:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                        encoders[col] = le
                
                # Remove target missing values
                if Y.isnull().sum() > 0:
                    mask = Y.notnull()
                    X = X[mask]
                    Y = Y[mask]
                    preprocessing_steps.append("Removed rows with missing target values")
                
                # Display preprocessing summary
                for i, step in enumerate(preprocessing_steps, 1):
                    st.write(f"{i}. {step}")
            
            # Train/test split
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, test_size=test_size/100, random_state=random_state, shuffle=True
            )
            
            # Feature scaling
            scaler = None
            if scaling_method != "None":
                if scaling_method == "StandardScaler":
                    scaler = StandardScaler()
                else:
                    scaler = RobustScaler()
                
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Model training
            if st.button("🚀 Launch Model Training", type="primary", use_container_width=True):
                st.markdown('<div class="section-header">🎯 Training Progress</div>', unsafe_allow_html=True)
                
                # Define models
                models = {}
                if "Linear Regression" in selected_models:
                    models["Linear Regression"] = LinearRegression()
                if "Ridge" in selected_models:
                    models["Ridge"] = Ridge(alpha=1.0)
                if "Lasso" in selected_models:
                    models["Lasso"] = Lasso(alpha=1.0)
                if "ElasticNet" in selected_models:
                    models["ElasticNet"] = ElasticNet(alpha=1.0)
                if "Decision Tree" in selected_models:
                    models["Decision Tree"] = DecisionTreeRegressor(random_state=random_state, max_depth=10)
                if "Random Forest" in selected_models:
                    models["Random Forest"] = RandomForestRegressor(random_state=random_state, n_estimators=100)
                if "Gradient Boosting" in selected_models:
                    models["Gradient Boosting"] = GradientBoostingRegressor(random_state=random_state)
                if "Support Vector Regression" in selected_models:
                    models["Support Vector Regression"] = SVR()
                
                # Train models with progress tracking
                results = {}
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (name, model) in enumerate(models.items()):
                    status_text.text(f"🔄 Training {name}...")
                    
                    try:
                        # Train model
                        model.fit(X_train_scaled, Y_train)
                        Y_pred = model.predict(X_test_scaled)
                        
                        # Calculate metrics
                        mae = metrics.mean_absolute_error(Y_test, Y_pred)
                        mse = metrics.mean_squared_error(Y_test, Y_pred)
                        rmse = np.sqrt(mse)
                        r2 = metrics.r2_score(Y_test, Y_pred)
                        
                        # Cross-validation
                        cv_scores = cross_val_score(model, X_train_scaled, Y_train, cv=cv_folds, scoring='r2')
                        
                        results[name] = {
                            'model': model,
                            'predictions': Y_pred,
                            'mae': mae,
                            'mse': mse,
                            'rmse': rmse,
                            'r2': r2,
                            'cv_mean': cv_scores.mean(),
                            'cv_std': cv_scores.std()
                        }
                        
                        status_text.text(f"✅ {name} completed!")
                        
                    except Exception as e:
                        st.error(f"❌ Error training {name}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / len(models))
                
                # Store results in session state
                st.session_state.training_results = results
                st.session_state.X_test = X_test
                st.session_state.Y_test = Y_test
                st.session_state.encoders = encoders
                st.session_state.scaler = scaler
                st.session_state.models_trained = True
                
                status_text.text("🎉 All models trained successfully!")
                
                st.markdown("""
                <div class="success-box">
                    <h4>🎉 Training Complete!</h4>
                    <p>All models have been trained and evaluated. Check the Results Dashboard for detailed analysis.</p>
                </div>
                """, unsafe_allow_html=True)

with tab4:
    if st.session_state.get('models_trained', False):
        st.markdown('<div class="section-header">📊 Advanced Results Dashboard</div>', unsafe_allow_html=True)
        
        results = st.session_state.training_results
        
        # Model comparison table
        st.subheader("🏆 Model Leaderboard")
        
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Model': name,
                'R² Score': f"{result['r2']:.4f}",
                'RMSE': f"{result['rmse']:.4f}",
                'MAE': f"{result['mae']:.4f}",
                'CV Score': f"{result['cv_mean']:.4f} (±{result['cv_std']:.4f})"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('R² Score', ascending=False)
        
        # Highlight best model
        st.dataframe(comparison_df, use_container_width=True)
        
        best_model_name = comparison_df.iloc[0]['Model']
        st.markdown(f"""
        <div class="success-box">
            <h4>🏆 Champion Model: {best_model_name}</h4>
            <p>Best performing model based on R² score</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Interactive model comparison charts
        st.subheader("📊 Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # R² Score comparison
            r2_scores = [result['r2'] for result in results.values()]
            model_names = list(results.keys())
            
            fig_r2 = px.bar(x=model_names, y=r2_scores, 
                           title="R² Score Comparison",
                           template=plotly_template,
                           color=r2_scores,
                           color_continuous_scale="Viridis")
            fig_r2.update_layout(showlegend=False)
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            # RMSE comparison
            rmse_scores = [result['rmse'] for result in results.values()]
            
            fig_rmse = px.bar(x=model_names, y=rmse_scores,
                             title="RMSE Comparison (Lower is Better)",
                             template=plotly_template,
                             color=rmse_scores,
                             color_continuous_scale="Reds_r")
            fig_rmse.update_layout(showlegend=False)
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        # Detailed analysis for best model
        st.subheader(f"🔍 Detailed Analysis: {best_model_name}")
        
        best_result = results[best_model_name]
        Y_test = st.session_state.Y_test
        Y_pred = best_result['predictions']
        
        # Create subplots for detailed analysis
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            # Actual vs Predicted
            fig_scatter = px.scatter(x=Y_test, y=Y_pred,
                                   title="Actual vs Predicted Values",
                                   labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                                   template=plotly_template)
            
            # Add perfect prediction line
            min_val = min(Y_test.min(), Y_pred.min())
            max_val = max(Y_test.max(), Y_pred.max())
            fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                           mode='lines', name='Perfect Prediction',
                                           line=dict(dash='dash', color='red')))
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with analysis_col2:
            # Residuals plot
            residuals = Y_test - Y_pred
            fig_residuals = px.scatter(x=Y_pred, y=residuals,
                                     title="Residuals Plot",
                                     labels={'x': 'Predicted Values', 'y': 'Residuals'},
                                     template=plotly_template)
            
            # Add horizontal line at y=0
            fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
            
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        # Feature importance (if available)
        if hasattr(best_result['model'], 'feature_importances_'):
            st.subheader("🎯 Feature Importance Analysis")
            
            feature_importance = pd.DataFrame({
                'Feature': st.session_state.feature_cols,
                'Importance': best_result['model'].feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig_importance = px.bar(feature_importance, x='Importance', y='Feature',
                                  orientation='h',
                                  title=f"Feature Importance - {best_model_name}",
                                  template=plotly_template)
            
            st.plotly_chart(fig_importance, use_container_width=True)

with tab5:
    if st.session_state.get('models_trained', False):
        st.markdown('<div class="section-header">🔮 AI Prediction Laboratory</div>', unsafe_allow_html=True)
        
        results = st.session_state.training_results
        
        # Model selection for prediction
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🎯 Select Prediction Model")
            
            model_options = list(results.keys())
            selected_model = st.selectbox("Choose Model", model_options, 
                                        index=0,  # Default to best model
                                        help="Select which trained model to use for predictions")
            
            # Display selected model performance
            selected_result = results[selected_model]
            
            st.markdown(f"""
            <div class="info-box">
                <h4>📊 Model Performance</h4>
                <p><strong>R² Score:</strong> {selected_result['r2']:.4f}</p>
                <p><strong>RMSE:</strong> {selected_result['rmse']:.4f}</p>
                <p><strong>CV Score:</strong> {selected_result['cv_mean']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🎛️ Input Features")
            
            # Create prediction input form
            new_data = {}
            feature_cols = st.session_state.feature_cols
            encoders = st.session_state.encoders
            df = st.session_state.df
            
            # Organize inputs in a more user-friendly way
            input_cols = st.columns(3)
            
            for i, col in enumerate(feature_cols):
                with input_cols[i % 3]:
                    if col in encoders:
                        # Categorical feature
                        available_values = list(encoders[col].classes_)
                        default_val = available_values[0] if available_values else ""
                        new_data[col] = st.selectbox(
                            f"🏷️ {col}", 
                            available_values,
                            key=f"pred_input_{col}",
                            help=f"Select value for {col}"
                        )
                    else:
                        # Numerical feature
                        if col in df.columns:
                            col_min = float(df[col].min())
                            col_max = float(df[col].max())
                            col_mean = float(df[col].mean())
                            col_std = float(df[col].std())
                            
                            new_data[col] = st.number_input(
                                f"🔢 {col}",
                                min_value=col_min - 2*col_std,  # Allow some extrapolation
                                max_value=col_max + 2*col_std,
                                value=col_mean,
                                step=col_std/10,
                                key=f"pred_input_{col}",
                                help=f"Range: {col_min:.2f} - {col_max:.2f}, Mean: {col_mean:.2f}"
                            )