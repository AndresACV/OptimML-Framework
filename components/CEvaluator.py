import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn_genetic import GASearchCV
from sklearn_genetic.space import Integer, Categorical, Continuous
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.feature_selection import SelectFromModel 
import time
from sklearn.linear_model import LassoCV 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.base import clone
import random
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
            'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
            'Lasso': Lasso(),
            'Ridge': Ridge(),
            'ElasticNet': ElasticNet(random_state=42),
            'KNeighborsRegressor': KNeighborsRegressor(),
            'SVR': SVR(),
            'MLPRegressor': MLPRegressor(random_state=42, max_iter=300),
            'XGBRegressor': XGBRegressor(random_state=42)
        }
        self.param_grids_genetic = self._get_param_grids_genetic()
        self.param_grids_exhaustive = self._get_param_grids_exhaustive()
        self.param_grids_bioinspired = self.param_grids_exhaustive

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
            'SVR': {
                'clf__kernel': Categorical(['rbf', 'linear']),
                'clf__C': Continuous(0.1, 10.0),
                'clf__epsilon': Continuous(0.01, 0.2),
                'clf__gamma': Categorical(['scale', 'auto'])
            },
            'ElasticNet': {
                'clf__alpha': Continuous(0.001, 1.0),
                'clf__l1_ratio': Continuous(0.1, 0.9),
                'clf__fit_intercept': Categorical([True, False])
            },
            'MLPRegressor': {
                'clf__hidden_layer_sizes': Categorical([(50,), (100,)]),
                'clf__activation': Categorical(['relu', 'tanh']),
                'clf__alpha': Continuous(0.0001, 0.01),
                'clf__learning_rate_init': Continuous(0.001, 0.01)
            },
            'GradientBoostingRegressor': {
                'clf__n_estimators': Integer(50, 100),
                'clf__learning_rate': Continuous(0.05, 0.2),
                'clf__max_depth': Integer(3, 5),
                'clf__subsample': Continuous(0.7, 1.0)
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
            'SVR': {
                'clf__kernel': ['rbf', 'linear'],
                'clf__C': [0.1, 1.0, 10.0],
                'clf__epsilon': [0.01, 0.1, 0.2],
                'clf__gamma': ['scale', 'auto']
            },
            'ElasticNet': {
                'clf__alpha': [0.001, 0.01, 0.1],
                'clf__l1_ratio': [0.1, 0.5, 0.9],
                'clf__fit_intercept': [True, False]
            },
            'MLPRegressor': {
                'clf__hidden_layer_sizes': [(50,), (100,)],
                'clf__activation': ['relu', 'tanh'],
                'clf__alpha': [0.0001, 0.001],
                'clf__learning_rate_init': [0.001, 0.01]
            },
            'GradientBoostingRegressor': {
                'clf__n_estimators': [50, 100],
                'clf__learning_rate': [0.05, 0.1],
                'clf__max_depth': [3, 5],
                'clf__subsample': [0.7, 1.0]
            },
            'XGBRegressor': {
                'clf__learning_rate': [0.01, 0.1],
                'clf__n_estimators': [50, 100],
                'clf__max_depth': [3, 5],
                'clf__subsample': [0.8, 1.0],
                'clf__colsample_bytree': [0.8, 1.0]
            }
        }

    def _normalize_param_grid(self, param_grid):
        return {key.replace('clf__', ''): value for key, value in param_grid.items()}

    def _add_clf_prefix(self, params):
        return {f"clf__{key}": value for key, value in params.items()}

    def _prepare_feature_selection(self):
        lasso_cv = LassoCV(cv=5, random_state=42)
        lasso_cv.fit(self.X_train, self.y_train)
        selector = SelectFromModel(lasso_cv, prefit=True)
        X_train_fs = selector.transform(self.X_train)
        X_test_fs = selector.transform(self.X_test)
        return selector, X_train_fs, X_test_fs

    def _sample_params(self, param_grid):
        return {key: random.choice(values) for key, values in param_grid.items()}

    def _evaluate_candidate(self, model, params, X_train, y_train):
        estimator = clone(model)
        estimator.set_params(**params)
        scores = cross_val_score(estimator, X_train, y_train, cv=5,
                                 scoring='neg_mean_squared_error', n_jobs=-1)
        return np.mean(scores)

    def _mutate_one_param(self, candidate, param_grid):
        mutated = candidate.copy()
        key = random.choice(list(param_grid.keys()))
        mutated[key] = random.choice(param_grid[key])
        return mutated

    def _create_new_candidate(self, current, best, global_best, param_grid):
        candidate = {}
        for key, values in param_grid.items():
            choice = random.random()
            if choice < 0.3:
                candidate[key] = current[key]
            elif choice < 0.6:
                candidate[key] = best[key]
            elif choice < 0.9:
                candidate[key] = global_best[key]
            else:
                candidate[key] = random.choice(values)
        return candidate

    def _fit_best_estimator(self, model, best_params, selector):
        pipeline = Pipeline([('fs', selector), ('clf', clone(model))])
        pipeline.set_params(**self._add_clf_prefix(best_params))
        pipeline.fit(self.X_train, self.y_train)
        return pipeline

    def genetic_search(self):
        """
        Perform genetic algorithm-based hyperparameter optimization for each model.
        """
        print_subsection_header("Genetic Algorithm Hyperparameter Optimization")
        print_info(f"Training {len(self.models)} models using genetic algorithm approach")

        results = {}
        model_count = len(self.models)

        for name, model in self.models.items():
            print_model_training_start(name, "genetic")
            selector, X_train_fs, _ = self._prepare_feature_selection()
            raw_grid = self._normalize_param_grid(self.param_grids_genetic[name])
            estimator = clone(model)

            start_time = time.time()
            evolved_estimator = GASearchCV(
                estimator=estimator,
                cv=5,
                scoring="neg_mean_squared_error",
                population_size=10,
                generations=5,
                tournament_size=3,
                elitism=True,
                crossover_probability=0.8,
                mutation_probability=0.1,
                param_grid=raw_grid,
                algorithm="eaSimple",
                n_jobs=-1,
                error_score='raise',
                verbose=True
            )
            evolved_estimator.fit(X_train_fs, self.y_train)
            training_time = time.time() - start_time

            best_params = self._add_clf_prefix(evolved_estimator.best_params_)
            best_estimator = self._fit_best_estimator(model, evolved_estimator.best_params_, selector)

            results[name] = {
                'best_params': best_params,
                'estimator': best_estimator,
                'training_time': training_time
            }

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
        """
        print_subsection_header("Exhaustive Grid Search Hyperparameter Optimization")
        print_info(f"Training {len(self.models)} models using exhaustive grid search approach")

        results = {}
        model_count = len(self.models)

        for name, model in self.models.items():
            print_model_training_start(name, "exhaustive")
            selector, X_train_fs, _ = self._prepare_feature_selection()
            raw_grid = self._normalize_param_grid(self.param_grids_exhaustive[name])
            estimator = clone(model)

            start_time = time.time()
            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=raw_grid,
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train_fs, self.y_train)
            training_time = time.time() - start_time

            best_params = self._add_clf_prefix(grid_search.best_params_)
            best_estimator = self._fit_best_estimator(model, grid_search.best_params_, selector)

            results[name] = {
                'best_params': best_params,
                'estimator': best_estimator,
                'training_time': training_time
            }

            print_model_training_complete(
                name,
                "exhaustive",
                training_time,
                {"neg_mse": grid_search.best_score_}
            )

        print_success(f"Completed exhaustive grid search training for {model_count} models")
        return results

    def pso_search(self):
        """
        Perform particle swarm optimization (PSO) search for model hyperparameters.
        """
        print_subsection_header("Particle Swarm Optimization Hyperparameter Optimization")
        print_info(f"Training {len(self.models)} models using PSO")

        results = {}
        for name, model in self.models.items():
            print_model_training_start(name, "pso")
            selector, X_train_fs, _ = self._prepare_feature_selection()
            raw_grid = self._normalize_param_grid(self.param_grids_bioinspired[name])
            swarm = [self._sample_params(raw_grid) for _ in range(6)]
            personal_best = list(swarm)
            personal_scores = [self._evaluate_candidate(model, p, X_train_fs, self.y_train) for p in personal_best]
            global_best = personal_best[int(np.argmax(personal_scores))]
            global_score = max(personal_scores)
            start_time = time.time()

            for iteration in range(4):
                for i, particle in enumerate(swarm):
                    trial = self._create_new_candidate(particle, personal_best[i], global_best, raw_grid)
                    score = self._evaluate_candidate(model, trial, X_train_fs, self.y_train)
                    if score > personal_scores[i]:
                        personal_best[i] = trial
                        personal_scores[i] = score
                    if score > global_score:
                        global_best = trial
                        global_score = score
                    swarm[i] = trial

            best_params = self._add_clf_prefix(global_best)
            best_estimator = self._fit_best_estimator(model, global_best, selector)
            training_time = time.time() - start_time
            results[name] = {
                'best_params': best_params,
                'estimator': best_estimator,
                'training_time': training_time
            }
            print_model_training_complete(name, "pso", training_time, {"neg_mse": global_score})

        print_success(f"Completed PSO training for {len(self.models)} models")
        return results

    def cdea_search(self):
        """
        Perform CDEA-like search for model hyperparameters.
        """
        print_subsection_header("CDEA Hyperparameter Optimization")
        print_info(f"Training {len(self.models)} models using CDEA")

        results = {}
        for name, model in self.models.items():
            print_model_training_start(name, "cdea")
            selector, X_train_fs, _ = self._prepare_feature_selection()
            raw_grid = self._normalize_param_grid(self.param_grids_bioinspired[name])
            population = [self._sample_params(raw_grid) for _ in range(6)]
            fitness = [self._evaluate_candidate(model, ind, X_train_fs, self.y_train) for ind in population]
            start_time = time.time()

            for iteration in range(4):
                for idx in range(len(population)):
                    a, b = random.sample(population, 2)
                    trial = {}
                    for key, values in raw_grid.items():
                        if random.random() < 0.5:
                            trial[key] = a[key]
                        else:
                            trial[key] = b[key]
                        if random.random() < 0.15:
                            trial[key] = random.choice(values)
                    score = self._evaluate_candidate(model, trial, X_train_fs, self.y_train)
                    if score > fitness[idx]:
                        population[idx] = trial
                        fitness[idx] = score

            best_idx = int(np.argmax(fitness))
            best_params = self._add_clf_prefix(population[best_idx])
            best_estimator = self._fit_best_estimator(model, population[best_idx], selector)
            training_time = time.time() - start_time
            results[name] = {
                'best_params': best_params,
                'estimator': best_estimator,
                'training_time': training_time
            }
            print_model_training_complete(name, "cdea", training_time, {"neg_mse": fitness[best_idx]})

        print_success(f"Completed CDEA training for {len(self.models)} models")
        return results

    def abc_search(self):
        """
        Perform artificial bee colony search for model hyperparameters.
        """
        print_subsection_header("Artificial Bee Colony Hyperparameter Optimization")
        print_info(f"Training {len(self.models)} models using ABC")

        results = {}
        for name, model in self.models.items():
            print_model_training_start(name, "abc")
            selector, X_train_fs, _ = self._prepare_feature_selection()
            raw_grid = self._normalize_param_grid(self.param_grids_bioinspired[name])
            colony = [self._sample_params(raw_grid) for _ in range(6)]
            fitness = [self._evaluate_candidate(model, bee, X_train_fs, self.y_train) for bee in colony]
            trial_counters = [0] * len(colony)
            start_time = time.time()

            for iteration in range(4):
                for idx, bee in enumerate(colony):
                    candidate = self._mutate_one_param(bee, raw_grid)
                    score = self._evaluate_candidate(model, candidate, X_train_fs, self.y_train)
                    if score > fitness[idx]:
                        colony[idx] = candidate
                        fitness[idx] = score
                        trial_counters[idx] = 0
                    else:
                        trial_counters[idx] += 1

                total_fitness = sum(max(f - min(fitness) + 1e-6, 0.0) for f in fitness)
                probabilities = [max(f - min(fitness) + 1e-6, 0.0) / total_fitness if total_fitness > 0 else 1.0 / len(fitness) for f in fitness]

                for _ in range(len(colony)):
                    selected_idx = random.choices(range(len(colony)), weights=probabilities, k=1)[0]
                    candidate = self._mutate_one_param(colony[selected_idx], raw_grid)
                    score = self._evaluate_candidate(model, candidate, X_train_fs, self.y_train)
                    if score > fitness[selected_idx]:
                        colony[selected_idx] = candidate
                        fitness[selected_idx] = score
                        trial_counters[selected_idx] = 0

                for idx in range(len(colony)):
                    if trial_counters[idx] >= 3:
                        colony[idx] = self._sample_params(raw_grid)
                        fitness[idx] = self._evaluate_candidate(model, colony[idx], X_train_fs, self.y_train)
                        trial_counters[idx] = 0

            best_idx = int(np.argmax(fitness))
            best_params = self._add_clf_prefix(colony[best_idx])
            best_estimator = self._fit_best_estimator(model, colony[best_idx], selector)
            training_time = time.time() - start_time
            results[name] = {
                'best_params': best_params,
                'estimator': best_estimator,
                'training_time': training_time
            }
            print_model_training_complete(name, "abc", training_time, {"neg_mse": fitness[best_idx]})

        print_success(f"Completed ABC training for {len(self.models)} models")
        return results

    def pso_search(self):
        """
        Perform particle swarm optimization (PSO) search for model hyperparameters.
        """
        print_subsection_header("Particle Swarm Optimization Hyperparameter Optimization")
        print_info(f"Training {len(self.models)} models using PSO")

        results = {}
        for name, model in self.models.items():
            print_model_training_start(name, "pso")
            selector, X_train_fs, _ = self._prepare_feature_selection()
            raw_grid = self._normalize_param_grid(self.param_grids_bioinspired[name])
            swarm = [self._sample_params(raw_grid) for _ in range(6)]
            personal_best = list(swarm)
            personal_scores = [self._evaluate_candidate(model, p, X_train_fs, self.y_train) for p in personal_best]
            global_best = personal_best[int(np.argmax(personal_scores))]
            global_score = max(personal_scores)
            start_time = time.time()

            for iteration in range(4):
                for i, particle in enumerate(swarm):
                    trial = self._create_new_candidate(particle, personal_best[i], global_best, raw_grid)
                    score = self._evaluate_candidate(model, trial, X_train_fs, self.y_train)
                    if score > personal_scores[i]:
                        personal_best[i] = trial
                        personal_scores[i] = score
                    if score > global_score:
                        global_best = trial
                        global_score = score
                    swarm[i] = trial

            best_params = self._add_clf_prefix(global_best)
            best_estimator = self._fit_best_estimator(model, global_best, selector)
            training_time = time.time() - start_time
            results[name] = {
                'best_params': best_params,
                'estimator': best_estimator,
                'training_time': training_time
            }
            print_model_training_complete(name, "pso", training_time, {"neg_mse": global_score})

        print_success(f"Completed PSO training for {len(self.models)} models")
        return results

    def cdea_search(self):
        """
        Perform CDEA-like search for model hyperparameters.
        """
        print_subsection_header("CDEA Hyperparameter Optimization")
        print_info(f"Training {len(self.models)} models using CDEA")

        results = {}
        for name, model in self.models.items():
            print_model_training_start(name, "cdea")
            selector, X_train_fs, _ = self._prepare_feature_selection()
            raw_grid = self._normalize_param_grid(self.param_grids_bioinspired[name])
            population = [self._sample_params(raw_grid) for _ in range(6)]
            fitness = [self._evaluate_candidate(model, ind, X_train_fs, self.y_train) for ind in population]
            start_time = time.time()

            for iteration in range(4):
                for idx in range(len(population)):
                    a, b = random.sample(population, 2)
                    trial = {}
                    for key, values in raw_grid.items():
                        if random.random() < 0.5:
                            trial[key] = a[key]
                        else:
                            trial[key] = b[key]
                        if random.random() < 0.15:
                            trial[key] = random.choice(values)
                    score = self._evaluate_candidate(model, trial, X_train_fs, self.y_train)
                    if score > fitness[idx]:
                        population[idx] = trial
                        fitness[idx] = score

            best_idx = int(np.argmax(fitness))
            best_params = self._add_clf_prefix(population[best_idx])
            best_estimator = self._fit_best_estimator(model, population[best_idx], selector)
            training_time = time.time() - start_time
            results[name] = {
                'best_params': best_params,
                'estimator': best_estimator,
                'training_time': training_time
            }
            print_model_training_complete(name, "cdea", training_time, {"neg_mse": fitness[best_idx]})

        print_success(f"Completed CDEA training for {len(self.models)} models")
        return results

    def abc_search(self):
        """
        Perform artificial bee colony search for model hyperparameters.
        """
        print_subsection_header("Artificial Bee Colony Hyperparameter Optimization")
        print_info(f"Training {len(self.models)} models using ABC")

        results = {}
        for name, model in self.models.items():
            print_model_training_start(name, "abc")
            selector, X_train_fs, _ = self._prepare_feature_selection()
            raw_grid = self._normalize_param_grid(self.param_grids_bioinspired[name])
            colony = [self._sample_params(raw_grid) for _ in range(6)]
            fitness = [self._evaluate_candidate(model, bee, X_train_fs, self.y_train) for bee in colony]
            trial_counters = [0] * len(colony)
            start_time = time.time()

            for iteration in range(4):
                for idx, bee in enumerate(colony):
                    candidate = self._mutate_one_param(bee, raw_grid)
                    score = self._evaluate_candidate(model, candidate, X_train_fs, self.y_train)
                    if score > fitness[idx]:
                        colony[idx] = candidate
                        fitness[idx] = score
                        trial_counters[idx] = 0
                    else:
                        trial_counters[idx] += 1

                total_fitness = sum(max(f - min(fitness) + 1e-6, 0.0) for f in fitness)
                probabilities = [max(f - min(fitness) + 1e-6, 0.0) / total_fitness if total_fitness > 0 else 1.0 / len(fitness) for f in fitness]

                for _ in range(len(colony)):
                    selected_idx = random.choices(range(len(colony)), weights=probabilities, k=1)[0]
                    candidate = self._mutate_one_param(colony[selected_idx], raw_grid)
                    score = self._evaluate_candidate(model, candidate, X_train_fs, self.y_train)
                    if score > fitness[selected_idx]:
                        colony[selected_idx] = candidate
                        fitness[selected_idx] = score
                        trial_counters[selected_idx] = 0

                for idx in range(len(colony)):
                    if trial_counters[idx] >= 3:
                        colony[idx] = self._sample_params(raw_grid)
                        fitness[idx] = self._evaluate_candidate(model, colony[idx], X_train_fs, self.y_train)
                        trial_counters[idx] = 0

            best_idx = int(np.argmax(fitness))
            best_params = self._add_clf_prefix(colony[best_idx])
            best_estimator = self._fit_best_estimator(model, colony[best_idx], selector)
            training_time = time.time() - start_time
            results[name] = {
                'best_params': best_params,
                'estimator': best_estimator,
                'training_time': training_time
            }
            print_model_training_complete(name, "abc", training_time, {"neg_mse": fitness[best_idx]})

        print_success(f"Completed ABC training for {len(self.models)} models")
        return results
