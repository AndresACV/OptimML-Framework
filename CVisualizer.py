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
from streamlit_option_menu import option_menu
from datetime import datetime
import json

@st.cache_data
def load_and_process_data(csv_files, use_sql, db_params):
    data_gen = cG.DataGenerator(csv_files, use_sql=use_sql, db_params=db_params)
    data_gen.process()
    X_trains, X_tests, y_trains, y_tests = data_gen.get_data()
    return X_trains, X_tests, y_trains, y_tests, data_gen.dfs

@st.cache_data
def train_models(X_trains, X_tests, y_trains, y_tests):
    results_genetic = {}
    results_exhaustive = {}

    for dataset_name in X_trains.keys():
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

    return results_genetic, results_exhaustive, genetic_evaluation, exhaustive_evaluation


class Visualizer:
    """Componente visualizador: Se encarga de crear el dashboard."""
    def __init__(self, results_genetic, results_exhaustive, dfs):
        self.results_genetic = results_genetic
        self.results_exhaustive = results_exhaustive
        self.dfs = dfs
        
    def ensure_arrow_compatible(self, df):
        """Ensure dataframe is compatible with Arrow for Streamlit."""
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
        data = []
        for dataset_name in self.results_genetic.keys():
            for alg in self.results_genetic[dataset_name].keys():
                # Obtener los tiempos de entrenamiento
                genetic_time = self.results_genetic[dataset_name][alg].get('training_time', 0)
                exhaustive_time = self.results_exhaustive[dataset_name][alg].get('training_time', 0)
                
                # Formatear los tiempos en segundos con 2 decimales
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
        """Crea un DataFrame detallado con todos los resultados para exportar a Excel."""
        data = []
        
        for dataset_name in self.results_genetic.keys():
            for alg in self.results_genetic[dataset_name].keys():
                # Convertir los parámetros a formato de texto para Excel
                genetic_params = json.dumps(self.results_genetic[dataset_name][alg]['best_params'])
                exhaustive_params = json.dumps(self.results_exhaustive[dataset_name][alg]['best_params'])
                
                # Obtener los tiempos de entrenamiento
                genetic_time = self.results_genetic[dataset_name][alg].get('training_time', 0)
                exhaustive_time = self.results_exhaustive[dataset_name][alg].get('training_time', 0)
                
                # Formatear los tiempos en segundos con 2 decimales
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
        """Crea un DataFrame con los hiperparámetros del mejor modelo según RMSE, solo para el dataset con mejor resultado."""
        data = []
        best_dataset = None
        best_algorithm = None
        best_rmse = float('inf')
        
        # Encontrar el dataset y algoritmo con el mejor RMSE global
        for dataset_name in self.results_genetic.keys():
            for alg in self.results_genetic[dataset_name].keys():
                genetic_rmse = self.results_genetic[dataset_name][alg]['rmse']
                exhaustive_rmse = self.results_exhaustive[dataset_name][alg]['rmse']
                
                # Usar el mejor entre genético y exhaustivo
                current_best_rmse = min(genetic_rmse, exhaustive_rmse)
                
                if current_best_rmse < best_rmse:
                    best_rmse = current_best_rmse
                    best_dataset = dataset_name
                    best_algorithm = alg
        
        if best_dataset and best_algorithm:
            # Obtener los hiperparámetros y RMSE para el mejor modelo genético
            genetic_best = self.results_genetic[best_dataset][best_algorithm]
            genetic_time = genetic_best.get('training_time', 0)
            genetic_params = genetic_best['best_params']
            
            # Obtener los hiperparámetros y RMSE para el mejor modelo exhaustivo
            exhaustive_best = self.results_exhaustive[best_dataset][best_algorithm]
            exhaustive_time = exhaustive_best.get('training_time', 0)
            exhaustive_params = exhaustive_best['best_params']
            
            # Crear entrada para aproximación genética
            genetic_entry = {
                'Dataset': best_dataset,
                'Approach': 'Genetic',
                'Algorithm': best_algorithm,
                'RMSE': genetic_best['rmse'],
                'Time (seconds)': genetic_time
            }
            
            # Añadir cada hiperparámetro como columna separada
            for param, value in genetic_params.items():
                genetic_entry[param] = value
            
            # Crear entrada para aproximación exhaustiva
            exhaustive_entry = {
                'Dataset': best_dataset,
                'Approach': 'Exhaustive',
                'Algorithm': best_algorithm,
                'RMSE': exhaustive_best['rmse'],
                'Time (seconds)': exhaustive_time
            }
            
            # Añadir cada hiperparámetro como columna separada
            for param, value in exhaustive_params.items():
                exhaustive_entry[param] = value
            
            data.append(genetic_entry)
            data.append(exhaustive_entry)
        
        df = pd.DataFrame(data)
        # Ensure the dataframe is Arrow-compatible
        df = self.ensure_arrow_compatible(df)
        return df, best_algorithm
    
    def create_best_algorithms_dataframe(self):
        """Crea un DataFrame con los 3 mejores algoritmos según RMSE en general."""
        data = []
        all_algorithms = []
        
        # Recopilar todos los resultados de algoritmos de todos los datasets
        for dataset_name in self.results_genetic.keys():
            for alg in self.results_genetic[dataset_name].keys():
                genetic_rmse = self.results_genetic[dataset_name][alg]['rmse']
                exhaustive_rmse = self.results_exhaustive[dataset_name][alg]['rmse']
                
                # Usar el mejor entre genético y exhaustivo para cada algoritmo
                best_approach = 'Genetic' if genetic_rmse <= exhaustive_rmse else 'Exhaustive'
                best_rmse = min(genetic_rmse, exhaustive_rmse)
                
                all_algorithms.append({
                    'dataset': dataset_name,
                    'algorithm': alg,
                    'approach': best_approach,
                    'rmse': best_rmse
                })
        
        # Ordenar todos los algoritmos por RMSE (menor a mayor)
        all_algorithms.sort(key=lambda x: x['rmse'])
        
        # Tomar los 3 mejores algoritmos en general
        best_algorithms = all_algorithms[:3]
        
        # Crear entradas para los mejores algoritmos
        for entry in best_algorithms:
            dataset_name = entry['dataset']
            alg = entry['algorithm']
            approach = entry['approach']
            
            if approach == 'Genetic':
                model_data = self.results_genetic[dataset_name][alg]
            else:
                model_data = self.results_exhaustive[dataset_name][alg]
            
            training_time = model_data.get('training_time', 0)
            params = model_data['best_params']
            
            result_entry = {
                'Dataset': dataset_name,
                'Approach': approach,
                'Algorithm': alg,
                'RMSE': model_data['rmse'],
                'Time (seconds)': training_time
            }
            
            # Añadir cada hiperparámetro como columna separada
            for param, value in params.items():
                result_entry[param] = value
            
            data.append(result_entry)
        
        df = pd.DataFrame(data)
        # Ensure the dataframe is Arrow-compatible
        df = self.ensure_arrow_compatible(df)
        return df
    
    def export_results_to_excel(self):
        """Exporta los resultados a un archivo Excel."""
        df_general = self.create_detailed_results_dataframe()
        df_best_hyperparams, best_algorithm = self.create_best_hyperparams_dataframe()
        df_best_algorithms = self.create_best_algorithms_dataframe()
        
        # Crear directorio para resultados si no existe
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Nombre del archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(results_dir, f"framework_results_{timestamp}.xlsx")
        
        # Crear un ExcelWriter para el archivo principal
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Escribir la tabla principal
            df_general.to_excel(writer, sheet_name='General Results', index=False)
            
            # Escribir la tabla de mejores hiperparámetros
            best_hyperparams_sheet = f"Best Model - {best_algorithm}"
            df_best_hyperparams.to_excel(writer, sheet_name=best_hyperparams_sheet, index=False)
            
            # Escribir la tabla de mejores algoritmos
            df_best_algorithms.to_excel(writer, sheet_name='Top 3 Algorithms', index=False)
            
            # Ajustar el ancho de las columnas para cada hoja
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
        """Crea un heatmap de correlación para un dataset específico."""
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
                options=["EDA", "Results"],
                menu_icon="cast",
                default_index=0,
            )

        if selected == "EDA":
            self.page_eda()
        elif selected == "Results":
            self.page_model_results()
            
    def page_eda(self):
        st.title("Exploratory Data Analysis (EDA)")

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
