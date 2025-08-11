import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import time
from .utils.print_utils import (
    print_subsection_header, print_info, print_success, 
    print_warning, print_error, print_table
)

"""
Model Prediction and Evaluation Module

This module is responsible for making predictions using trained models and evaluating
their performance using various metrics such as R2 score, MSE, and MAE.
"""

class Predictor:
    """
    Predictor class responsible for making predictions and evaluating model performance.
    
    This class provides methods for making predictions using trained models and
    evaluating their performance using various metrics such as R2 score, MSE, and MAE.
    
    Attributes:
        X_test (DataFrame): Testing feature set
        y_test (Series): Testing target values
    """
    
    def __init__(self, X_test, y_test):
        """
        Initialize the Predictor with testing data.
        
        Args:
            X_test (DataFrame): Testing feature set
            y_test (Series): Testing target values
        """
        self.X_test = X_test
        self.y_test = y_test
        
    def predict(self, model):
        """
        Make predictions using a trained model.
        
        Args:
            model: Trained model with a predict method
            
        Returns:
            array: Predicted values
        """
        print_info(f"Making predictions with {model.__class__.__name__}")
        return model.predict(self.X_test)
        
    def evaluate(self, y_pred):
        """
        Evaluate predictions using various metrics.
        
        Args:
            y_pred (array): Predicted values
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        print_info("Evaluating model performance...")
        
        r2 = r2_score(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        
        metrics = {
            'r2': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }
        
        # Print metrics in a table format
        metrics_data = [
            ['R² Score', f"{r2:.4f}"],
            ['Mean Squared Error', f"{mse:.4f}"],
            ['Root Mean Squared Error', f"{rmse:.4f}"],
            ['Mean Absolute Error', f"{mae:.4f}"]
        ]
        
        print_table(['Metric', 'Value'], metrics_data)
        
        return metrics
        
    def evaluate_results(self, results):
        """
        Evaluate results from model training.
        
        This method evaluates the performance of trained models and returns
        a dictionary of evaluation metrics for each model.
        
        Args:
            results (dict): Dictionary of model training results
            
        Returns:
            dict: Dictionary of evaluation metrics for each model
        """
        print_subsection_header("Model Evaluation")
        
        evaluation = {}
        model_count = len(results)
        model_index = 0
        
        for model_name, model_data in results.items():
            model_index += 1
            print_info(f"Evaluating {model_name} ({model_index}/{model_count})...")
            
            model = model_data['estimator']
            
            # Make predictions
            start_time = time.time()
            y_pred = self.predict(model)
            prediction_time = time.time() - start_time
            
            # Evaluate predictions
            metrics = self.evaluate(y_pred)
            
            # Add training time and prediction time
            metrics['training_time'] = model_data['training_time']
            metrics['prediction_time'] = prediction_time
            
            # Add best parameters
            metrics['best_params'] = model_data['best_params']
            
            # Store evaluation results
            evaluation[model_name] = metrics
            
            print_success(f"Evaluation complete for {model_name}")
            
        print_success(f"Evaluation complete for all {model_count} models")
        
        return evaluation
