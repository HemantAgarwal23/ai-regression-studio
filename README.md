# 🤖 AI Regression Studio

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Issues](https://img.shields.io/github/issues/yourusername/ai-regression-studio.svg)](https://github.com/yourusername/ai-regression-studio/issues)

> **Next-Generation Machine Learning Platform for Regression Analysis**
> 
> A sophisticated, interactive web application that democratizes machine learning by providing an intuitive interface for regression modeling, automated feature engineering, and intelligent predictions.

![AI Regression Studio Demo](https://via.placeholder.com/800x400/667eea/ffffff?text=AI+Regression+Studio+Demo)

## 🌟 Key Features

### 🎯 **Core Capabilities**
- **Smart Data Upload**: Support for CSV, Excel (.xlsx, .xls) files with automatic preprocessing
- **Intelligent Feature Engineering**: Automated categorical encoding, missing value handling, and feature scaling
- **Multiple ML Algorithms**: Linear Regression, Ridge, Lasso, ElasticNet, Decision Trees, Random Forest, Gradient Boosting, SVR
- **Automated Model Comparison**: Side-by-side performance evaluation with statistical metrics
- **Interactive Predictions**: Real-time prediction interface with confidence intervals

### 🎨 **Modern User Experience**
- **Dual Theme Support**: Professional light and dark themes with smooth transitions
- **Interactive Visualizations**: Powered by Plotly for dynamic, responsive charts
- **Tab-based Navigation**: Intuitive workflow organized into logical steps
- **Real-time Progress Tracking**: Visual feedback during model training
- **Responsive Design**: Optimized for both desktop and mobile viewing

### 🧠 **Advanced Analytics**
- **Smart Target Suggestion**: AI-powered recommendation for target variables
- **Feature Importance Analysis**: Understand which features drive predictions
- **Cross-Validation**: Robust model evaluation with configurable K-fold validation
- **Residual Analysis**: Comprehensive error pattern detection
- **Prediction Confidence Assessment**: Statistical uncertainty quantification

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ai-regression-studio.git
   cd ai-regression-studio
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser:**
   Navigate to `http://localhost:8501`

### Docker Installation (Alternative)

```bash
# Build the image
docker build -t ai-regression-studio .

# Run the container
docker run -p 8501:8501 ai-regression-studio
```

## 📋 Requirements

```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.3.0
plotly>=5.15.0
matplotlib>=3.5.0
seaborn>=0.11.0
openpyxl>=3.0.0
```

## 🎯 Usage Guide

### 1. **Data Upload & Processing**
- Upload your dataset in CSV or Excel format
- Review automatic data quality assessment
- Handle missing values and data types

### 2. **Exploratory Data Analysis**
- Interactive data filtering and exploration
- Smart target variable suggestion
- Correlation analysis and feature relationships
- Distribution visualizations

### 3. **Model Training**
- Select from 8 different regression algorithms
- Configure hyperparameters and preprocessing options
- Real-time training progress with performance metrics
- Automated cross-validation evaluation

### 4. **Results Dashboard**
- Comprehensive model leaderboard
- Interactive performance comparisons
- Detailed residual analysis
- Feature importance visualization

### 5. **Prediction Laboratory**
- Interactive prediction interface
- Multiple scenario testing
- Confidence interval assessment
- Feature contribution analysis

## 🏗️ Architecture

```
ai-regression-studio/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── Dockerfile            # Docker configuration
├── .gitignore           # Git ignore rules
│
├── assets/              # Static assets
│   ├── screenshots/     # Application screenshots
│   └── sample_data/     # Sample datasets
│
├── utils/               # Utility functions
│   ├── __init__.py
│   ├── data_processing.py
│   ├── model_training.py
│   └── visualization.py
│
└── tests/               # Unit tests
    ├── __init__.py
    ├── test_data_processing.py
    └── test_models.py
```

## 🔧 Configuration

### Theme Customization
The application supports both light and dark themes. Toggle between themes using the button in the top-right corner.

### Model Parameters
Customize model hyperparameters through the sidebar:
- **Train/Test Split**: 10-50% test size
- **Cross-Validation**: 3, 5, or 10 folds
- **Feature Scaling**: StandardScaler, RobustScaler, or None
- **Model-specific parameters**: Alpha for regularized models, tree depth, etc.

## 📊 Supported Models

| Model | Type | Use Case | Hyperparameters |
|-------|------|----------|------------------|
| **Linear Regression** | Linear | Simple, interpretable models | None |
| **Ridge Regression** | Regularized Linear | High multicollinearity | Alpha (regularization) |
| **Lasso Regression** | Regularized Linear | Feature selection | Alpha (regularization) |
| **ElasticNet** | Regularized Linear | Combined L1/L2 penalty | Alpha, L1 ratio |
| **Decision Tree** | Tree-based | Non-linear relationships | Max depth, min samples |
| **Random Forest** | Ensemble | Robust, feature importance | N estimators, max depth |
| **Gradient Boosting** | Ensemble | High performance | N estimators, learning rate |
| **Support Vector Regression** | Kernel-based | Non-linear patterns | Kernel, C parameter |

## 📈 Sample Datasets

The application works with any regression dataset. Here are some examples:

### Real Estate Price Prediction
```csv
bedrooms,bathrooms,area_sqft,location,age_years,price
3,2,1500,downtown,10,300000
4,3,2000,suburb,5,450000
2,1,1200,downtown,15,250000
```

### Sales Forecasting
```csv
advertising_spend,season,product_category,region,sales
5000,summer,electronics,north,75000
8000,winter,clothing,south,120000
3000,spring,electronics,west,45000
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/ai-regression-studio.git

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run with hot reload
streamlit run app.py --server.runOnSave true
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=utils tests/

# Run specific test file
pytest tests/test_models.py -v
```

## 🐛 Troubleshooting

### Common Issues

**Issue: "ModuleNotFoundError: No module named 'plotly'"**
```bash
Solution: pip install plotly
```

**Issue: "File upload fails"**
```bash
Solution: Ensure file is CSV/Excel format and under 200MB
```

**Issue: "Model training takes too long"**
```bash
Solution: Reduce dataset size or use simpler models for large datasets
```

### Performance Optimization

- For datasets > 10k rows: Use Random Forest or Gradient Boosting
- For high-dimensional data: Use Lasso for feature selection
- For real-time predictions: Use Linear Regression or Decision Trees

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit Team**: For the amazing web app framework
- **Scikit-learn Contributors**: For comprehensive ML algorithms
- **Plotly Team**: For interactive visualization library
- **Open Source Community**: For continuous inspiration and support



**⭐ Star this repository if it helped you!**

Made with ❤️ by HEMANT AGARWAL(https://github.com/HemantAgarwal23)



</div>
