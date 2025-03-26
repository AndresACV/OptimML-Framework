import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
from sklearn.exceptions import ConvergenceWarning
import warnings
import threading
import seaborn as sns
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, URL
from sqlalchemy_utils import database_exists, create_database
import pymssql
import time
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import os 
import matplotlib.pyplot as plt
import seaborn as sns
from . import CGenerator as cG
from . import CEvaluator as cE
from . import CPredictor as cP
from . import CVisualizer as cV
import streamlit as st
import sys
from .utils.print_utils import (
    print_section_header, print_subsection_header, print_info, 
    print_success, print_warning, print_error, print_progress,
    print_timestamp, print_data_loading, print_application_header
)

"""
Main Controller Module

This module serves as the main controller for the OptimML Framework. It orchestrates
the data loading, model training, evaluation, and visualization processes by coordinating
the interactions between the different components of the framework.
"""

def main():
    """
    Main function that orchestrates the entire application workflow.
    
    This function:
    1. Sets up database parameters (if SQL is used)
    2. Locates and loads datasets from the datasets directory
    3. Processes the data to ensure compatibility with Arrow/Streamlit
    4. Trains models using both genetic and exhaustive search methods
    5. Creates and displays the interactive dashboard
    
    Returns:
        None
    """
    # Print application header
    print_application_header()
    
    # Print section header for initialization
    print_section_header("Initializing OptimML Framework")
    
    db_params = {
        "username": "sa",
        "password": "sa",
        "host": "DESKTOP-FTU949F",
        "database": "framework_db",
        "port": 1433
    }

    start_time = time.time()
    use_sql = False
    
    # Print database configuration
    if use_sql:
        print_info("Using SQL database for data storage")
        print_info(f"Database: {db_params['database']} on {db_params['host']}")
    else:
        print_info("Using CSV files for data storage")
    
    # Update path to point to datasets in the project root, not in the components directory
    file_paths = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
    print_info(f"Dataset directory: {file_paths}")
    
    # Print section header for data loading
    print_section_header("Loading Datasets")
    
    csv_files = []
    csv_file_count = 0
    
    # Count CSV files first
    for filename in os.listdir(file_paths):
        if filename.endswith(".csv"):
            csv_file_count += 1
    
    # Then load them with progress tracking
    file_index = 0
    for filename in os.listdir(file_paths):
        if filename.endswith(".csv"):
            full_path = os.path.join(file_paths, filename)
            base_name = os.path.splitext(filename)[0]
            file_index += 1
            print_data_loading(base_name, file_index, csv_file_count)
            csv_files.append(full_path)
            
    if not csv_files:
        print_error("No datasets found in the datasets directory")
        st.error("No datasets found")
        return
    
    print_success(f"Successfully loaded {len(csv_files)} datasets")

    pd.set_option('future.infer_string', False)  

    # Print section header for data processing
    print_section_header("Processing Data")
    print_info("Loading and preprocessing datasets...")
    
    X_trains, X_tests, y_trains, y_tests, dfs = cV.load_and_process_data(csv_files, use_sql, db_params)
    
    print_success(f"Data processing complete for {len(dfs)} datasets")
    
    # Print subsection header for Arrow compatibility processing
    print_subsection_header("Ensuring Arrow Compatibility")
    print_info("Converting data types for Streamlit/Arrow compatibility...")

    # Process each dataframe to ensure Arrow compatibility
    dataset_count = len(dfs.keys())
    for i, dataset_name in enumerate(dfs.keys()):
        # Create a new DataFrame with explicit types
        new_df = pd.DataFrame()
        
        # Process each column with appropriate type conversion
        for col in dfs[dataset_name].columns:
            col_data = dfs[dataset_name][col]
            
            # Convert based on data type
            if pd.api.types.is_integer_dtype(col_data):
                new_df[col] = col_data.astype('int64')
            elif pd.api.types.is_float_dtype(col_data):
                new_df[col] = col_data.astype('float64')
            elif pd.api.types.is_bool_dtype(col_data):
                new_df[col] = col_data.astype('bool')
            else:
                new_df[col] = col_data.astype(str)
        
        # Replace the original dataframe with the new one
        dfs[dataset_name] = new_df
        
        # Print progress
        print_progress(i + 1, dataset_count, f"Processing {dataset_name}")
    
    print_success("Arrow compatibility processing complete")
    
    # Print section header for model training
    print_section_header("Training Models")
    print_info("Starting model training with genetic and exhaustive search methods...")
    print_info("This may take some time depending on the dataset size and complexity")
    
    model_training_start = time.time()
    results_genetic, results_exhaustive, genetic_evaluation, exhaustive_evaluation = cV.train_models(X_trains, X_tests, y_trains, y_tests)
    model_training_duration = time.time() - model_training_start
    
    print_success(f"Model training complete in {model_training_duration:.2f} seconds")
    
    # Print section header for visualization
    print_section_header("Creating Visualization Dashboard")
    print_info("Initializing Streamlit dashboard...")

    visualizer = cV.Visualizer(genetic_evaluation, exhaustive_evaluation, dfs)
    visualizer.create_dashboard()
    
    # Print execution summary
    total_duration = time.time() - start_time
    print_section_header("Execution Summary")
    print_info(f"Total execution time: {total_duration:.2f} seconds")
    print_success("OptimML Framework execution completed successfully")


if __name__ == "__main__":
    is_streamlit = any(arg.endswith('streamlit') for arg in sys.argv) or 'streamlit.cli' in sys.modules
    
    if not is_streamlit:
        print_warning("For the best experience with the dashboard, run this script using:")
        print_info("streamlit run CMain.py")
        print_warning("Running in non-interactive mode...\n")
        
    main()