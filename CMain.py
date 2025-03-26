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
import CGenerator as cG
import CEvaluator as cE
import CPredictor as cP
import CVisualizer as cV
import streamlit as st
import sys

def main():
    db_params = {
        "username": "sa",
        "password": "sa",
        "host": "DESKTOP-FTU949F",
        "database": "framework_db",
        "port": 1433
    }

    start_time = time.time()
    use_sql = False
    file_paths = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    
    csv_files = []

    for filename in os.listdir(file_paths):
        if filename.endswith(".csv"):
            full_path = os.path.join(file_paths, filename)
            base_name = os.path.splitext(filename)[0]
            print(f"Reading file: {base_name}")
            csv_files.append(full_path)
            
    if not csv_files:
        st.error("No dataset encontrado")
        return

    pd.set_option('future.infer_string', False)  

    X_trains, X_tests, y_trains, y_tests, dfs = cV.load_and_process_data(csv_files, use_sql, db_params)

    # Process each dataframe to ensure Arrow compatibility
    for dataset_name in dfs.keys():
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

    results_genetic, results_exhaustive, genetic_evaluation, exhaustive_evaluation = cV.train_models(X_trains, X_tests, y_trains, y_tests)

    visualizer = cV.Visualizer(genetic_evaluation, exhaustive_evaluation, dfs)
    visualizer.create_dashboard()


if __name__ == "__main__":
    is_streamlit = any(arg.endswith('streamlit') for arg in sys.argv) or 'streamlit.cli' in sys.modules
    
    if not is_streamlit:
        print("Note: For the best experience with the dashboard, run this script using:")
        print("streamlit run CMain.py")
        print("\nRunning in non-interactive mode...\n")
        
    main()