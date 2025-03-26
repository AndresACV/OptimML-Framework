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


class ModelEvaluator:
    """Componente evaluador: Se encarga del algoritmo genético y del exhaustivo."""

    def __init__(self, X_train, X_test, y_train, y_test):
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
        """Define los espacios de búsqueda de parámetros para cada modelo."""
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
        """Define los espacios de búsqueda de parámetros para el método exhaustivo."""
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
        """Realiza la búsqueda genética de hiperparámetros para cada modelo."""
        results = {}
        for name, model in self.models.items():
            lasso_cv = LassoCV(cv=5) 
            lasso_cv.fit(self.X_train, self.y_train)
            f_selection = SelectFromModel(lasso_cv)
            self.X_train = f_selection.transform(self.X_train)
            self.X_test = f_selection.transform(self.X_test)
            pl = Pipeline([
              ('fs', f_selection), 
              ('clf', model), 
            ])            
            print(f"Entrenando {name} con método genético...")
            
            # Medir el tiempo de inicio
            start_time = time.time()
            
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
            
            # Calcular el tiempo de entrenamiento
            training_time = time.time() - start_time
            
            results[name] = {
                'best_params': evolved_estimator.best_params_,
                'estimator': evolved_estimator.best_estimator_,
                'training_time': training_time  # Añadir el tiempo de entrenamiento
            }
        return results

    def exhaustive_search(self):
        """Realiza la búsqueda exhaustiva de hiperparámetros para cada modelo."""
        results = {}
        for name, model in self.models.items():
            lasso_cv = LassoCV(cv=5) 
            lasso_cv.fit(self.X_train, self.y_train)
            f_selection = SelectFromModel(lasso_cv)
            self.X_train = f_selection.transform(self.X_train)
            self.X_test = f_selection.transform(self.X_test)
            pl = Pipeline([
              ('clf', model), 
            ])
            print(f"Entrenando {name} con método exhaustivo...")
            
            # Medir el tiempo de inicio
            start_time = time.time()
            
            grid_search = GridSearchCV(
                estimator=pl,
                param_grid=self.param_grids_exhaustive[name],
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            # Calcular el tiempo de entrenamiento
            training_time = time.time() - start_time
            
            results[name] = {
                'best_params': grid_search.best_params_,
                'estimator': grid_search.best_estimator_,
                'training_time': training_time  # Añadir el tiempo de entrenamiento
            }
        return results