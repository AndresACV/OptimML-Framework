# OptimML Framework
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

<p align="center">
  <img src="https://raw.githubusercontent.com/AndresACV/OptimML-Framework/main/assets/logo.png" alt="OptimML Logo" width="400"/>
</p>

> A comprehensive machine learning framework for automated model selection and hyperparameter optimization using both genetic algorithms and exhaustive search methods. This interactive dashboard allows users to analyze datasets, train multiple ML models, and visualize results through an intuitive Streamlit interface.

## ✨ Features

- **📊 Automated Data Processing**: Load and preprocess CSV datasets with automatic handling of data types
- **🧬 Dual Optimization Approaches**: Compare genetic algorithm search vs. exhaustive grid search for model selection
- **🤖 Multiple ML Algorithms**: Supports various regression models including Linear Regression, Decision Trees, Random Forests, Lasso, Ridge, KNN, and XGBoost
- **📈 Interactive Dashboard**: Visualize model performance, dataset correlations, and detailed results
- **📋 Exportable Results**: Save analysis results to Excel for further examination
- **💾 SQL Integration**: Optional database connectivity for larger datasets
- **🖨️ Enhanced Console Output**: Standardized, colorful, and informative print utilities for better user experience

## 🏗️ Architecture

The framework follows a modular component-based architecture:

| Component | Description |
|-----------|-------------|
| **CGenerator** | Handles data loading, preprocessing, and train/test splitting |
| **CEvaluator** | Implements genetic and exhaustive search algorithms for model optimization |
| **CPredictor** | Provides prediction capabilities using optimized models |
| **CVisualizer** | Creates the interactive dashboard with multiple visualization options |
| **print_utils** | Provides standardized and visually appealing console output functions |

## 📋 Requirements

- Python 3.8+
- Dependencies listed in requirements.txt:
  - streamlit
  - pandas
  - numpy
  - scikit-learn
  - scikit-learn-genetic
  - xgboost
  - matplotlib
  - seaborn
  - plotly
  - sqlalchemy
  - pymssql (for SQL Server connectivity)

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AndresACV/OptimML-Framework.git
   cd OptimML-Framework
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

1. Place your CSV datasets in the `datasets` folder

2. Run the application using Streamlit:
   ```bash
   streamlit run app.py
   ```
   
   Alternatively, on Windows, you can use the provided batch file:
   ```bash
   run_app.bat
   ```

3. The dashboard will open in your default web browser, allowing you to:
   - Explore dataset statistics and correlations
   - View model performance comparisons
   - Analyze the best hyperparameters for each algorithm
   - Export results to Excel

## 📁 Project Structure

```
OptimML-Framework/
├── app.py                          # Streamlit application entry point
├── components/                     # Framework components
│   ├── __init__.py                 # Package initialization
│   ├── CMain.py                    # Main application logic
│   ├── CGenerator.py               # Data loading and preprocessing component
│   ├── CEvaluator.py               # Model evaluation and optimization component
│   ├── CPredictor.py               # Prediction component
│   ├── CVisualizer.py              # Dashboard and visualization component
│   └── utils/                      # Utility modules
│       ├── __init__.py             # Utils package initialization
│       └── print_utils.py          # Print formatting utilities
├── datasets/                       # Directory for input CSV datasets
├── results/                        # Directory for output results
├── assets/                         # Images and other static assets
├── requirements.txt                # Python dependencies
└── run_app.bat                     # Windows batch file for easy startup
```

## 📊 Dashboard Screenshots

<p align="center">
  <!-- These placeholder comments will be replaced with actual screenshots once available -->
  <!-- <img src="https://raw.githubusercontent.com/AndresACV/OptimML-Framework/main/assets/dashboard-main.png" alt="Dashboard Main View" width="700"/> -->
</p>

<p align="center">
  <!-- <img src="https://raw.githubusercontent.com/AndresACV/OptimML-Framework/main/assets/model-comparison.png" alt="Model Comparison" width="700"/> -->
</p>

## 🔧 Customization

### Adding New Machine Learning Algorithms

Extend the `models` dictionary in `components/CEvaluator.py`:

```python
self.models = {
    'LinearRegression': LinearRegression(),
    'DecisionTreeRegressor': DecisionTreeRegressor(),
    # Add your custom model here:
    'YourCustomModel': YourCustomModel()
}
```

### Customizing the Dashboard Layout

Modify the `create_dashboard` method in `components/CVisualizer.py`:

```python
def create_dashboard(self):
    st.title('OptimML Framework Dashboard')
    # Add your custom dashboard components here
```

### Adjusting Hyperparameter Search Spaces

Modify the parameter search spaces in the `_get_param_grids_genetic` and `_get_param_grids_exhaustive` methods in `components/CEvaluator.py`.

## 🖨️ Enhanced Print Utilities

The framework includes a comprehensive print utilities module (`components/utils/print_utils.py`) that provides standardized and visually appealing console output. The utilities include:

- **Section Headers**: Clear visual separation between major sections
- **Subsection Headers**: Visual separation for subsections
- **Info Messages**: Standard formatted informational messages
- **Success Messages**: Green-highlighted success notifications
- **Warning Messages**: Yellow-highlighted warning notifications
- **Error Messages**: Red-highlighted error notifications
- **Progress Indicators**: Visual progress bars for long-running operations
- **Timestamp Messages**: Messages with timestamps for chronological reference
- **Model Training Notifications**: Specialized formatting for model training events
- **Data Loading Indicators**: Specialized formatting for data loading operations
- **Summary Statistics**: Tabular formatting for statistical summaries
- **Table Formatting**: Consistent tabular data presentation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

Andrés Calvo - [GitHub Profile](https://github.com/AndresACV)
