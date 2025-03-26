# Make the components directory a proper Python package
"""
OptimML Framework Components Package

This package contains the core components of the OptimML Framework, a comprehensive
machine learning framework for automated model selection and hyperparameter optimization
using both genetic algorithms and exhaustive search methods.

Components:
- CMain: Main orchestration module with the entry point for the application
- CGenerator: Data loading, preprocessing, and train/test splitting
- CEvaluator: Model evaluation and hyperparameter optimization
- CPredictor: Prediction capabilities using optimized models
- CVisualizer: Visualization and dashboard creation
"""

__version__ = '1.0.0'

# Export main function
from .CMain import main

# Export component classes
from .CGenerator import DataGenerator
from .CEvaluator import ModelEvaluator
from .CPredictor import Predictor
from .CVisualizer import Visualizer

# Define what symbols are exported when using "from components import *"
__all__ = [
    'main',
    'DataGenerator',
    'ModelEvaluator',
    'Predictor',
    'Visualizer'
]
