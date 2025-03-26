import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


class Predictor:
    """Componente predictor: Conecta el evaluador con las métricas de predicción."""

    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def evaluate_model(self, model):
        """Evalúa el modelo utilizando R2 y RMSE."""
        y_pred = model.predict(self.X_test)
        r2 = r2_score(self.y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        return r2, rmse

    def evaluate_results(self, results):
        """Evalúa los resultados de múltiples modelos."""
        evaluation = {}
        for name, result in results.items():
            r2, rmse = self.evaluate_model(result['estimator'])
            evaluation[name] = {
                'r2': r2,
                'rmse': rmse,
                'best_params': result['best_params'],
                'training_time': result.get('training_time', 0)  # Incluir el tiempo de entrenamiento
            }
        return evaluation
