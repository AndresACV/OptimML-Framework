import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Categorical, Continuous
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.feature_selection import SelectFromModel 
import time
from sklearn.linear_model import LassoCV 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from .utils.print_utils import (
    print_model_training_start, print_model_training_complete,
    print_subsection_header, print_info, print_success
)

"""
Model Evaluation and Hyperparameter Optimization Module

This module is responsible for evaluating machine learning models and optimizing their
hyperparameters using both genetic algorithms and exhaustive grid search methods.
It supports multiple regression algorithms and provides a comprehensive framework for
comparing their performance under different optimization strategies.
"""

class ModelEvaluator:
    """
    Model Evaluator class responsible for hyperparameter optimization and model evaluation.
    
    This class implements two approaches for hyperparameter optimization:
    1. Genetic algorithm search using GASearchCV
    2. Exhaustive grid search using GridSearchCV
    
    It supports multiple regression algorithms including Linear Regression, Decision Trees,
    Random Forests, Lasso, Ridge, KNN, and XGBoost. Feature selection is performed using
    LassoCV before model training to improve performance.
    
    Attributes:
        X_train (DataFrame): Training feature set
        X_test (DataFrame): Testing feature set
        y_train (Series): Training target values
        y_test (Series): Testing target values
        models (dict): Dictionary of regression models to evaluate
        param_grids_genetic (dict): Parameter search spaces for genetic algorithm
        param_grids_exhaustive (dict): Parameter search spaces for exhaustive search
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        """
        Initialize the ModelEvaluator with training and testing data.
        
        Args:
            X_train (DataFrame): Training feature set
            X_test (DataFrame): Testing feature set
            y_train (Series): Training target values
            y_test (Series): Testing target values
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models = {
            'LinearRegression': LinearRegression(),
            'DecisionTreeRegressor': DecisionTreeRegressor(),
            'RandomForestRegressor': RandomForestRegressor(),
            'Lasso': Lasso(),
            'Ridge': Ridge(),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'XGBRegressor': XGBRegressor(random_state=42)
        }
        self.param_grids_genetic = self._get_param_grids_genetic()
        self.param_grids_exhaustive = self._get_param_grids_exhaustive()

    def _get_param_grids_genetic(self):
        """
        Define parameter search spaces for genetic algorithm optimization.
        
        This method creates a dictionary of parameter search spaces for each model
        using the specific format required by GASearchCV. Parameters include:
        - Continuous: For floating-point parameters with a range
        - Integer: For integer parameters with a range
        - Categorical: For parameters with discrete choices
        
        Returns:
            dict: Parameter search spaces for each model for genetic algorithm
        """
        return {
            'LinearRegression': {
                "clf__copy_X": Categorical([True, False]),
                "clf__fit_intercept": Categorical([True, False]),
                "clf__positive": Categorical([True, False])
            },
            'DecisionTreeRegressor': {
                "clf__max_depth": Integer(3, 10),
                'clf__min_samples_split': Integer(2, 10),
                'clf__min_samples_leaf': Integer(1, 4),
                'clf__random_state': Categorical([42])
            },
            'RandomForestRegressor': {
                "clf__n_estimators": Integer(50, 100),
                "clf__max_depth": Integer(5, 10),
                'clf__min_samples_split': Integer(2, 5),
                'clf__random_state': Categorical([42])
            },
            'Lasso': {
                'clf__alpha': Continuous(1.0, 1.0),
                'clf__fit_intercept': Categorical([True, False]),
                'clf__max_iter': Integer(1000, 2000),
                'clf__tol': Continuous(0.0001, 0.001),
                'clf__selection': Categorical(['cyclic', 'random'])
            },
            'Ridge': {
                'clf__alpha': Continuous(1.0, 1.0),
                'clf__fit_intercept': Categorical([True, False]),
                'clf__tol': Continuous(0.0001, 0.001),
                'clf__solver': Categorical(['auto', 'svd', 'cholesky'])
            },
            'KNeighborsRegressor': {
                'clf__n_neighbors': Integer(3, 7),
                'clf__weights': Categorical(['uniform', 'distance']),
                'clf__algorithm': Categorical(['auto', 'ball_tree', 'kd_tree'])
            },
            'XGBRegressor': {
                'clf__learning_rate': Continuous(0.01, 0.1),
                'clf__n_estimators': Integer(50, 100),
                'clf__max_depth': Integer(3, 5),
                'clf__subsample': Continuous(0.8, 1.0),
                'clf__colsample_bytree': Continuous(0.8, 1.0)
            }
        }
    
    def _get_param_grids_exhaustive(self):
        """
        Define parameter search spaces for exhaustive grid search optimization.
        
        This method creates a dictionary of parameter search spaces for each model
        using the format required by GridSearchCV. Unlike the genetic algorithm approach,
        this uses discrete values for all parameters, which makes it more computationally
        intensive but potentially more thorough.
        
        Returns:
            dict: Parameter search spaces for each model for exhaustive grid search
        """
        return {
            
            'LinearRegression': {
                "clf__copy_X": [True, False],
                "clf__fit_intercept": [True, False],
                "clf__positive": [True, False]
            },
            'DecisionTreeRegressor': {
                "clf__max_depth": [3, 5, 7, 10],
                'clf__min_samples_split': [2, 5, 10],
                'clf__min_samples_leaf': [1, 2, 4],
                'clf__random_state': [42]
            },
            'RandomForestRegressor': {
                "clf__n_estimators": [50, 100],
                "clf__max_depth": [5, 10],
                'clf__min_samples_split': [2, 5],
                'clf__random_state': [42]
            },
            'Lasso': {
                'clf__alpha': [1.0],
                'clf__fit_intercept': [True, False],
                'clf__max_iter': [1000, 2000],
                'clf__tol': [0.0001, 0.001],
                'clf__selection': ['cyclic', 'random']
            },
            'Ridge': {
                'clf__alpha': [1.0],
                'clf__fit_intercept': [True, False],
                'clf__tol': [0.0001, 0.001],
                'clf__solver': ['auto', 'svd', 'cholesky']
            },
            'KNeighborsRegressor': {
                'clf__n_neighbors': [3, 5, 7],
                'clf__weights': ['uniform', 'distance'],
                'clf__algorithm': ['auto', 'ball_tree', 'kd_tree']
            },
            'XGBRegressor': {
                'clf__learning_rate': [0.01, 0.1],
                'clf__n_estimators': [50, 100],
                'clf__max_depth': [3, 5],
                'clf__subsample': [0.8, 1.0],
                'clf__colsample_bytree': [0.8, 1.0]
            }
        }

    def genetic_search(self):
        """
        Perform genetic algorithm-based hyperparameter optimization for each model.
        
        This method:
        1. Applies feature selection using LassoCV
        2. Creates a pipeline with feature selection and the model
        3. Uses GASearchCV to find optimal hyperparameters through genetic evolution
        4. Measures and records training time for each model
        5. Returns the best parameters, best estimator, and training time for each model
        
        Returns:
            dict: Results dictionary containing best parameters, estimator, and training time for each model
        """
        print_subsection_header("Genetic Algorithm Hyperparameter Optimization")
        print_info(f"Training {len(self.models)} models using genetic algorithm approach")
        
        results = {}
        model_count = len(self.models)
        model_index = 0
        
        for name, model in self.models.items():
            model_index += 1
            print_model_training_start(name, "genetic")
            
            # Feature selection with LassoCV
            lasso_cv = LassoCV(cv=5) 
            lasso_cv.fit(self.X_train, self.y_train)
            f_selection = SelectFromModel(lasso_cv)
            self.X_train = f_selection.transform(self.X_train)
            self.X_test = f_selection.transform(self.X_test)
            
            # Create pipeline
            pl = Pipeline([
              ('fs', f_selection), 
              ('clf', model), 
            ])            
            
            # Measure start time
            start_time = time.time()
            
            # Train model using genetic algorithm
            evolved_estimator = GASearchCV(
                estimator=pl,
                cv=5,
                scoring="neg_mean_squared_error",
                population_size=10,
                generations=5,
                tournament_size=3,
                elitism=True,
                crossover_probability=0.8,
                mutation_probability=0.1,
                param_grid=self.param_grids_genetic[name],
                algorithm="eaSimple",
                n_jobs=-1,
                error_score='raise',
                verbose=True
            )
            evolved_estimator.fit(self.X_train, self.y_train)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Store results
            results[name] = {
                'best_params': evolved_estimator.best_params_,
                'estimator': evolved_estimator.best_estimator_,
                'training_time': training_time
            }
            
            # Print completion message
            print_model_training_complete(
                name, 
                "genetic", 
                training_time, 
                {"neg_mse": evolved_estimator.best_score_}
            )
            
        print_success(f"Completed genetic algorithm training for {model_count} models")
        return results

    def exhaustive_search(self):
        """
        Perform exhaustive grid search-based hyperparameter optimization for each model.
        
        This method:
        1. Applies feature selection using LassoCV
        2. Creates a pipeline with the model
        3. Uses GridSearchCV to find optimal hyperparameters through exhaustive search
        4. Measures and records training time for each model
        5. Returns the best parameters, best estimator, and training time for each model
        
        Returns:
            dict: Results dictionary containing best parameters, estimator, and training time for each model
        """
        print_subsection_header("Exhaustive Grid Search Hyperparameter Optimization")
        print_info(f"Training {len(self.models)} models using exhaustive grid search approach")
        
        results = {}
        model_count = len(self.models)
        model_index = 0
        
        for name, model in self.models.items():
            model_index += 1
            print_model_training_start(name, "exhaustive")
            
            # Feature selection with LassoCV
            lasso_cv = LassoCV(cv=5) 
            lasso_cv.fit(self.X_train, self.y_train)
            f_selection = SelectFromModel(lasso_cv)
            self.X_train = f_selection.transform(self.X_train)
            self.X_test = f_selection.transform(self.X_test)
            
            # Create pipeline
            pl = Pipeline([
              ('clf', model), 
            ])
            
            # Measure start time
            start_time = time.time()
            
            # Train model using grid search
            grid_search = GridSearchCV(
                estimator=pl,
                param_grid=self.param_grids_exhaustive[name],
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Store results
            results[name] = {
                'best_params': grid_search.best_params_,
                'estimator': grid_search.best_estimator_,
                'training_time': training_time
            }
            
            # Print completion message
            print_model_training_complete(
                name, 
                "exhaustive", 
                training_time, 
                {"neg_mse": grid_search.best_score_}
            )
            
        print_success(f"Completed exhaustive grid search training for {model_count} models")
        return results