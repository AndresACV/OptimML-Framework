# ML Framework Dashboard

A comprehensive machine learning framework for automated model selection and hyperparameter optimization using both genetic algorithms and exhaustive search methods. This interactive dashboard allows users to analyze datasets, train multiple ML models, and visualize results through an intuitive Streamlit interface.

## Features

- **Automated Data Processing**: Load and preprocess CSV datasets with automatic handling of data types
- **Dual Optimization Approaches**: Compare genetic algorithm search vs. exhaustive grid search for model selection
- **Multiple ML Algorithms**: Supports various regression models including Linear Regression, Decision Trees, Random Forests, Lasso, Ridge, KNN, and XGBoost
- **Interactive Dashboard**: Visualize model performance, dataset correlations, and detailed results
- **Exportable Results**: Save analysis results to Excel for further examination
- **SQL Integration**: Optional database connectivity for larger datasets

## Architecture

The framework follows a modular component-based architecture:

- **CGenerator**: Handles data loading, preprocessing, and train/test splitting
- **CEvaluator**: Implements genetic and exhaustive search algorithms for model optimization
- **CPredictor**: Provides prediction capabilities using optimized models
- **CVisualizer**: Creates the interactive dashboard with multiple visualization options

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ml-framework-dashboard.git
   cd ml-framework-dashboard
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Place your CSV datasets in the `datasets` folder

2. Run the application using Streamlit:
   ```
   streamlit run app.py
   ```
   
   Alternatively, on Windows, you can use the provided batch file:
   ```
   run_app.bat
   ```

3. The dashboard will open in your default web browser, allowing you to:
   - Explore dataset statistics and correlations
   - View model performance comparisons
   - Analyze the best hyperparameters for each algorithm
   - Export results to Excel

## Project Structure

```
├── app.py                  # Streamlit application entry point
├── CMain.py                # Main application logic
├── CGenerator.py           # Data loading and preprocessing component
├── CEvaluator.py           # Model evaluation and optimization component
├── CPredictor.py           # Prediction component
├── CVisualizer.py          # Dashboard and visualization component
├── datasets/               # Directory for input CSV datasets
├── results/                # Directory for output results
├── requirements.txt        # Python dependencies
└── run_app.bat             # Windows batch file for easy startup
```

## Customization

- Add new machine learning algorithms by extending the `models` dictionary in `CEvaluator.py`
- Customize the dashboard layout by modifying the `create_dashboard` method in `CVisualizer.py`
- Adjust hyperparameter search spaces in the `_get_param_grids_genetic` and `_get_param_grids_exhaustive` methods

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
