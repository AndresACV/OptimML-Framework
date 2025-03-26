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
import streamlit as st
from streamlit_option_menu import option_menu
from datetime import datetime
import json
from .utils.print_utils import (
    print_section_header, print_subsection_header, print_info, 
    print_success, print_warning, print_error, print_progress
)

"""
Data Visualization and Dashboard Module

This module provides functionality for visualizing model evaluation results and creating
interactive dashboards. It includes utilities for loading and processing data, training models,
and creating various visualizations to compare model performance across different algorithms
and optimization approaches.
"""

@st.cache_data
def load_and_process_data(csv_files, use_sql, db_params):
    """
    Load and process data from CSV files or SQL database.
    
    This function uses the DataGenerator to load data from either CSV files or a SQL database,
    process it, and prepare train/test splits for model training.
    
    Args:
        csv_files (list): List of paths to CSV files
        use_sql (bool): Whether to use SQL database for data storage
        db_params (dict): Database connection parameters if use_sql is True
        
    Returns:
        tuple: (X_trains, X_tests, y_trains, y_tests, dfs) containing:
            - X_trains (dict): Dictionary of training feature sets by dataset name
            - X_tests (dict): Dictionary of testing feature sets by dataset name
            - y_trains (dict): Dictionary of training target values by dataset name
            - y_tests (dict): Dictionary of testing target values by dataset name
            - dfs (dict): Dictionary of original dataframes by dataset name
    """
    print_subsection_header("Data Loading and Processing")
    
    data_gen = cG.DataGenerator(csv_files, use_sql=use_sql, db_params=db_params)
    data_gen.process()
    X_trains, X_tests, y_trains, y_tests = data_gen.get_data()
    
    print_success("Data loading and processing complete")
    
    return X_trains, X_tests, y_trains, y_tests, data_gen.dfs

@st.cache_data
def train_models(X_trains, X_tests, y_trains, y_tests):
    """
    Train models using both genetic and exhaustive search methods.
    
    This function trains models for each dataset using both genetic algorithm and
    exhaustive search approaches for hyperparameter optimization, and evaluates
    their performance.
    
    Args:
        X_trains (dict): Dictionary of training feature sets by dataset name
        X_tests (dict): Dictionary of testing feature sets by dataset name
        y_trains (dict): Dictionary of training target values by dataset name
        y_tests (dict): Dictionary of testing target values by dataset name
        
    Returns:
        tuple: (results_genetic, results_exhaustive, genetic_evaluation, exhaustive_evaluation) containing:
            - results_genetic (dict): Results from genetic algorithm optimization
            - results_exhaustive (dict): Results from exhaustive search optimization
            - genetic_evaluation (dict): Evaluation metrics for genetic algorithm models
            - exhaustive_evaluation (dict): Evaluation metrics for exhaustive search models
    """
    print_section_header("Model Training")
    
    results_genetic = {}
    results_exhaustive = {}

    for dataset_name in X_trains.keys():
        print_subsection_header(f"Training models for dataset: {dataset_name}")
        
        evaluator = cE.ModelEvaluator(X_trains[dataset_name], X_tests[dataset_name], 
                                   y_trains[dataset_name], y_tests[dataset_name])
        results_genetic[dataset_name] = evaluator.genetic_search()
        X_tests[dataset_name] = evaluator.X_train
        X_tests[dataset_name] = evaluator.X_test      
        results_exhaustive[dataset_name] = evaluator.exhaustive_search()

    genetic_evaluation = {}
    exhaustive_evaluation = {}
    for dataset_name in X_trains.keys():
        predictor = cP.Predictor(X_tests[dataset_name], y_tests[dataset_name])
        genetic_evaluation[dataset_name] = predictor.evaluate_results(results_genetic[dataset_name])
        exhaustive_evaluation[dataset_name] = predictor.evaluate_results(results_exhaustive[dataset_name])

    print_success("Model training complete")
    
    return results_genetic, results_exhaustive, genetic_evaluation, exhaustive_evaluation


class Visualizer:
    """
    Visualizer class responsible for creating interactive dashboards and visualizations.
    
    This class provides methods to create various visualizations and dashboards to compare
    model performance across different algorithms and optimization approaches. It handles
    the presentation of results in a user-friendly format using Streamlit.
    
    Attributes:
        results_genetic (dict): Results from genetic algorithm optimization
        results_exhaustive (dict): Results from exhaustive search optimization
        dfs (dict): Dictionary of original dataframes by dataset name
    """
    
    def __init__(self, results_genetic, results_exhaustive, dfs):
        """
        Initialize the Visualizer with model results and data.
        
        Args:
            results_genetic (dict): Results from genetic algorithm optimization
            results_exhaustive (dict): Results from exhaustive search optimization
            dfs (dict): Dictionary of original dataframes by dataset name
        """
        self.results_genetic = results_genetic
        self.results_exhaustive = results_exhaustive
        self.dfs = dfs
        
    def ensure_arrow_compatible(self, df):
        """
        Ensure dataframe is compatible with Arrow for Streamlit.
        
        This method converts DataFrame columns to types that are compatible with
        Apache Arrow, which is used by Streamlit for data transfer. This helps
        prevent errors when displaying DataFrames in the Streamlit interface.
        
        Args:
            df (DataFrame): Input DataFrame to convert
            
        Returns:
            DataFrame: New DataFrame with Arrow-compatible data types
        """
        # Create a new DataFrame with explicit types that Arrow can handle
        new_df = pd.DataFrame()
        
        # Process each column with the appropriate type conversion
        for col in df.columns:
            # Get the column data
            col_data = df[col]
            
            # Check data type and convert appropriately
            if pd.api.types.is_integer_dtype(col_data):
                # Convert integers to standard Python int
                new_df[col] = col_data.astype('int64')
            elif pd.api.types.is_float_dtype(col_data):
                # Convert floats to standard Python float
                new_df[col] = col_data.astype('float64')
            elif pd.api.types.is_bool_dtype(col_data):
                # Convert to standard Python bool
                new_df[col] = col_data.astype('bool')
            else:
                # Convert everything else to string
                new_df[col] = col_data.astype(str)
                
        return new_df

    def create_results_dataframe(self):
        """
        Create a summary DataFrame with model performance results.
        
        This method compiles the results from both genetic and exhaustive search
        approaches into a single DataFrame for easy comparison, including RMSE
        and training time for each algorithm and dataset.
        
        Returns:
            DataFrame: Summary DataFrame with model performance metrics
        """
        data = []
        for dataset_name in self.results_genetic.keys():
            for alg in self.results_genetic[dataset_name].keys():
                # Get training times
                genetic_time = self.results_genetic[dataset_name][alg].get('training_time', 0)
                exhaustive_time = self.results_exhaustive[dataset_name][alg].get('training_time', 0)
                
                # Format times in seconds with 2 decimals
                genetic_time_formatted = f"{genetic_time:.2f} s"
                exhaustive_time_formatted = f"{exhaustive_time:.2f} s"
                
                data.append({
                    'Dataset': dataset_name,
                    'Algorithm': alg,
                    'RMSE (Genetic)': self.results_genetic[dataset_name][alg]['rmse'],
                    'Time (G)': genetic_time_formatted,
                    'RMSE (Exhaustive)': self.results_exhaustive[dataset_name][alg]['rmse'],
                    'Time (E)': exhaustive_time_formatted
                })
        df = pd.DataFrame(data)
        df.sort_values(by=['RMSE (Genetic)', 'RMSE (Exhaustive)'], ascending=[True, True], inplace=True)
        # Ensure the dataframe is Arrow-compatible
        df = self.ensure_arrow_compatible(df)
        return df
    
    def create_detailed_results_dataframe(self):
        """
        Create a detailed DataFrame with all results for export to Excel.
        
        This method creates a comprehensive DataFrame with detailed information
        about each model's performance, including hyperparameters, RMSE, and
        training time for both genetic and exhaustive search approaches.
        
        Returns:
            DataFrame: Detailed DataFrame with model performance and hyperparameters
        """
        data = []
        
        for dataset_name in self.results_genetic.keys():
            for alg in self.results_genetic[dataset_name].keys():
                # Convert parameters to text format for Excel
                genetic_params = json.dumps(self.results_genetic[dataset_name][alg]['best_params'])
                exhaustive_params = json.dumps(self.results_exhaustive[dataset_name][alg]['best_params'])
                
                # Get training times
                genetic_time = self.results_genetic[dataset_name][alg].get('training_time', 0)
                exhaustive_time = self.results_exhaustive[dataset_name][alg].get('training_time', 0)
                
                # Format times in seconds with 2 decimals
                genetic_time_formatted = f"{genetic_time:.2f} seconds"
                exhaustive_time_formatted = f"{exhaustive_time:.2f} seconds"
                
                data.append({
                    'Dataset': dataset_name,
                    'Algorithm': alg,
                    'RMSE (Genetic)': self.results_genetic[dataset_name][alg]['rmse'],
                    'Hyperparameters (Genetic)': genetic_params,
                    'Training Time (Genetic)': genetic_time_formatted,
                    'RMSE (Exhaustive)': self.results_exhaustive[dataset_name][alg]['rmse'],
                    'Hyperparameters (Exhaustive)': exhaustive_params,
                    'Training Time (Exhaustive)': exhaustive_time_formatted
                })
        
        df = pd.DataFrame(data)
        # Ensure the dataframe is Arrow-compatible
        df = self.ensure_arrow_compatible(df)
        return df
    
    def create_best_hyperparams_dataframe(self):
        """
        Create a DataFrame with hyperparameters of the best model by RMSE.
        
        This method identifies the dataset and algorithm with the best overall RMSE
        performance and creates a DataFrame with the hyperparameters for both genetic
        and exhaustive search approaches for that best model.
        
        Returns:
            tuple: (DataFrame, str) - DataFrame with hyperparameters of the best performing model,
                   and the name of the best algorithm
        """
        data = []
        best_dataset = None
        best_algorithm = None
        best_rmse = float('inf')
        
        # Find the dataset and algorithm with the best global RMSE
        for dataset_name in self.results_genetic.keys():
            for alg in self.results_genetic[dataset_name].keys():
                genetic_rmse = self.results_genetic[dataset_name][alg]['rmse']
                exhaustive_rmse = self.results_exhaustive[dataset_name][alg]['rmse']
                
                # Use the best between genetic and exhaustive
                current_best_rmse = min(genetic_rmse, exhaustive_rmse)
                
                if current_best_rmse < best_rmse:
                    best_rmse = current_best_rmse
                    best_dataset = dataset_name
                    best_algorithm = alg
        
        if best_dataset and best_algorithm:
            # Get hyperparameters and RMSE for the best genetic model
            genetic_best = self.results_genetic[best_dataset][best_algorithm]
            genetic_time = genetic_best.get('training_time', 0)
            genetic_params = genetic_best['best_params']
            
            # Get hyperparameters and RMSE for the best exhaustive model
            exhaustive_best = self.results_exhaustive[best_dataset][best_algorithm]
            exhaustive_time = exhaustive_best.get('training_time', 0)
            exhaustive_params = exhaustive_best['best_params']
            
            # Create entry for genetic approach
            genetic_entry = {
                'Dataset': best_dataset,
                'Approach': 'Genetic',
                'Algorithm': best_algorithm,
                'RMSE': genetic_best['rmse'],
                'Time (seconds)': genetic_time
            }
            
            # Add each hyperparameter as a separate column
            for param, value in genetic_params.items():
                genetic_entry[param] = value
            
            # Create entry for exhaustive approach
            exhaustive_entry = {
                'Dataset': best_dataset,
                'Approach': 'Exhaustive',
                'Algorithm': best_algorithm,
                'RMSE': exhaustive_best['rmse'],
                'Time (seconds)': exhaustive_time
            }
            
            # Add each hyperparameter as a separate column
            for param, value in exhaustive_params.items():
                exhaustive_entry[param] = value
            
            data.append(genetic_entry)
            data.append(exhaustive_entry)
        
        df = pd.DataFrame(data)
        # Ensure the dataframe is Arrow-compatible
        df = self.ensure_arrow_compatible(df)
        return df, best_algorithm
    
    def create_best_algorithms_dataframe(self):
        """
        Create a DataFrame with the 3 best algorithms by RMSE overall.
        
        This method collects all algorithm results from all datasets, sorts them by RMSE,
        and returns the top 3 algorithms.
        
        Returns:
            DataFrame: DataFrame with the top 3 algorithms by RMSE
        """
        data = []
        
        # Collect all algorithm results from all datasets
        all_results = []
        for dataset_name in self.results_genetic.keys():
            for alg in self.results_genetic[dataset_name].keys():
                genetic_rmse = self.results_genetic[dataset_name][alg]['rmse']
                exhaustive_rmse = self.results_exhaustive[dataset_name][alg]['rmse']
                
                # Use the best between genetic and exhaustive
                best_rmse = min(genetic_rmse, exhaustive_rmse)
                best_approach = 'Genetic' if genetic_rmse <= exhaustive_rmse else 'Exhaustive'
                
                all_results.append({
                    'Dataset': dataset_name,
                    'Algorithm': alg,
                    'RMSE': best_rmse,
                    'Approach': best_approach
                })
        
        # Sort by RMSE (ascending) and get the top 3
        all_results.sort(key=lambda x: x['RMSE'])
        top_algorithms = all_results[:3] if len(all_results) >= 3 else all_results
        
        # Create entries for the best algorithms
        for result in top_algorithms:
            data.append({
                'Dataset': result['Dataset'],
                'Algorithm': result['Algorithm'],
                'RMSE': result['RMSE'],
                'Approach': result['Approach']
            })
        
        df = pd.DataFrame(data)
        # Ensure the dataframe is Arrow-compatible
        df = self.ensure_arrow_compatible(df)
        return df
    
    def export_results_to_excel(self):
        """
        Export the results to an Excel file.
        
        This method creates an Excel file with three sheets: General Results, Best Model,
        and Top 3 Algorithms.
        
        Returns:
            str: Path to the generated Excel file
        """
        df_general = self.create_detailed_results_dataframe()
        df_best_hyperparams, best_algorithm = self.create_best_hyperparams_dataframe()
        df_best_algorithms = self.create_best_algorithms_dataframe()
        
        # Create directory for results if it doesn't exist
        # Adjust path to be relative to the project root, not the components directory
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(results_dir, f"framework_results_{timestamp}.xlsx")
        
        # Create an ExcelWriter for the main file
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Write the main table
            df_general.to_excel(writer, sheet_name='General Results', index=False)
            
            # Write the best hyperparameters table
            best_hyperparams_sheet = f"Best Model - {best_algorithm}"
            df_best_hyperparams.to_excel(writer, sheet_name=best_hyperparams_sheet, index=False)
            
            # Write the best algorithms table
            df_best_algorithms.to_excel(writer, sheet_name='Top 3 Algorithms', index=False)
            
            # Adjust the column width for each sheet
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                df = None
                
                if sheet_name == 'General Results':
                    df = df_general
                elif sheet_name == best_hyperparams_sheet:
                    df = df_best_hyperparams
                elif sheet_name == 'Top 3 Algorithms':
                    df = df_best_algorithms
                
                if df is not None:
                    for i, col in enumerate(df.columns):
                        max_length = max(df[col].astype(str).map(len).max(), len(col)) + 2
                        worksheet.column_dimensions[chr(65 + i)].width = max_length
        
        return filename

    def create_correlation_heatmap(self, dataset_name):
        """
        Create a correlation heatmap for a specific dataset.
        
        This method generates a heatmap showing the correlation between features in the
        specified dataset.
        
        Args:
            dataset_name (str): Name of the dataset to create the heatmap for
        
        Returns:
            plotly.graph_objs.Figure: Heatmap figure
        """
        if dataset_name not in self.dfs:
            return go.Figure()  
        
        corr = self.dfs[dataset_name].corr()
        fig = px.imshow(corr, 
                        labels=dict(color="Correlation"),
                        x=corr.columns,
                        y=corr.columns,
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1)
        fig.update_layout(title=f"Correlation Heatmap - {dataset_name}")
        st.plotly_chart(fig)
        
    def create_dashboard(self):        
        with st.sidebar:
            selected = option_menu(
                menu_title="Main Menu",
                options=["Exploratory Data Analysis", "Results"],
                menu_icon="cast",
                default_index=0,
            )

        if selected == "Exploratory Data Analysis":
            self.page_eda()
        elif selected == "Results":
            self.page_model_results()
            
    def page_eda(self):
        """
        Create the Exploratory Data Analysis page.
        
        This method generates a page with various visualizations and statistics for
        exploratory data analysis.
        """
        st.title("Exploratory Data Analysis")

        selected_dataset = st.selectbox("Select the dataset:", list(self.dfs.keys()))

        if selected_dataset:
            data = self.dfs[selected_dataset]
            melted_data = data.melt()

            st.subheader("Data Description")
            st.write("Data Types:")
            st.write(data.dtypes)
            st.write("Missing Values:")
            st.write(data.isnull().sum())
            st.write("Statistical Description:")
            st.write(data.describe())


            st.subheader("Histogram")
            fig = px.histogram(melted_data, x="value", color="variable", marginal="box", hover_data=['variable'])
            st.plotly_chart(fig)

            st.subheader("Boxplot")
            fig = px.box(melted_data, y="value", color="variable", points="all", hover_data=['variable'])
            st.plotly_chart(fig)

            st.subheader("Correlation Map")
            self.create_correlation_heatmap(selected_dataset)
            
    def page_model_results(self):
        """
        Create the Model Results page.
        
        This method generates a page with various visualizations and statistics for
        model results.
        """
        st.title("Results")

        df_results = self.create_results_dataframe()
        all_datasets = list(self.dfs.keys())
        all_algorithms = list(self.results_genetic[all_datasets[0]].keys())
        selected_dataset = st.selectbox("Select the dataset:", all_datasets)
        selected_algorithm = st.selectbox("Select the algorithm", all_algorithms)

        # Display RMSE comparison bar chart
        df_results_filtered = df_results[df_results['Dataset'] == selected_dataset]
        comparison_fig = go.Figure()
        comparison_fig.add_trace(go.Bar(
            x=df_results_filtered['Algorithm'],
            y=df_results_filtered['RMSE (Genetic)'],
            name='Genetic',
            marker_color='blue'
        ))
        comparison_fig.add_trace(go.Bar(
            x=df_results_filtered['Algorithm'],
            y=df_results_filtered['RMSE (Exhaustive)'],
            name='Exhaustive',
            marker_color='red'
        ))
        comparison_fig.update_layout(
            title=f'RMSE Comparison: Genetic vs Exhaustive ({selected_dataset})',
            xaxis_title='Algorithm',
            yaxis_title='RMSE',
            barmode='group'
        )
        st.plotly_chart(comparison_fig)

        st.subheader(f"Best parameters for {selected_algorithm} ({selected_dataset}):")
        st.markdown("**Genetic Method:**")
        st.json(self.results_genetic[selected_dataset][selected_algorithm]['best_params'])
        st.markdown("**Exhaustive Method:**")
        st.json(self.results_exhaustive[selected_dataset][selected_algorithm]['best_params'])

        st.subheader("Results Table")
        st.dataframe(df_results)
        
        # Button to export to Excel
        if st.button("Export Results to Excel"):
            with st.spinner("Exporting results..."):
                filename = self.export_results_to_excel()
                
                # Get the best algorithm to display in the message
                _, best_algorithm = self.create_best_hyperparams_dataframe()
                
                st.success(f"Results exported successfully:")
                
                # Display the generated file
                st.info(f" Excel file created with the following reports:")
                st.markdown(f"""
                - **General Results**: Table with all model results
                - **Best Model - {best_algorithm}**: Table with hyperparameters of the best model according to RMSE
                - **Top 3 Algorithms**: Table with the 3 best algorithms according to RMSE
                """)
                
                # Verify that the file exists before trying to open it
                if os.path.exists(filename):
                    # Create a download link for the file
                    with open(filename, "rb") as f:
                        st.download_button(
                            label=f"Download {os.path.basename(filename)}",
                            data=f,
                            file_name=os.path.basename(filename),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=filename  # Use the file name as a unique key
                        )
                else:
                    st.error(f"File not found: {filename}")
