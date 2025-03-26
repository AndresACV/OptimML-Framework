import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, URL
from sqlalchemy_utils import database_exists, create_database
import os
import warnings
from .utils.print_utils import (
    print_subsection_header, print_info, print_success, 
    print_warning, print_error, print_table, print_summary_statistics
)

"""
Data Generation and Preprocessing Module

This module handles all data-related operations including loading datasets from files or SQL databases,
preprocessing the data, feature scaling, and preparing train/test splits for model training.
It serves as the foundation for the machine learning pipeline in the OptimML Framework.
"""

class DataGenerator:
    """
    Data Generator class responsible for loading, preprocessing, and preparing datasets.
    
    This class handles loading data from CSV files or SQL databases, performs basic
    exploratory data analysis, preprocesses the data (including handling missing values),
    scales features, and splits the data into training and testing sets.
    
    Attributes:
        file_paths (list): List of paths to the CSV files to be processed
        use_sql (bool): Flag indicating whether to use SQL database for data storage
        db_params (dict): Database connection parameters if use_sql is True
        dfs (dict): Dictionary of pandas DataFrames containing the loaded datasets
        X_trains (dict): Dictionary of training feature sets for each dataset
        X_tests (dict): Dictionary of testing feature sets for each dataset
        y_trains (dict): Dictionary of training target values for each dataset
        y_tests (dict): Dictionary of testing target values for each dataset
        engine: SQLAlchemy engine for database operations if use_sql is True
    """
    
    def __init__(self, file_paths, use_sql=False, db_params=None):
        """
        Initialize the DataGenerator with file paths and database configuration.
        
        Args:
            file_paths (list): List of paths to the CSV files to be processed
            use_sql (bool, optional): Whether to use SQL database. Defaults to False.
            db_params (dict, optional): Database connection parameters. Defaults to None.
        """
        self.file_paths = file_paths  
        self.use_sql = use_sql
        self.db_params = db_params
        self.dfs = {}  
        self.X_trains = {}
        self.X_tests = {}
        self.y_trains = {}
        self.y_tests = {}
        self.engine = None

    def create_db_engine(self):
        """
        Creates and returns the database connection engine.
        
        This method establishes a connection to the SQL database using the provided
        connection parameters. If the database doesn't exist, it creates a new one.
        
        Returns:
            None: The engine is stored as an instance attribute
        """
        connection_string = (
            f"mysql+mysqlconnector://{self.db_params['username']}:{self.db_params['password']}@"
            f"{self.db_params['host']}:{self.db_params['port']}/{self.db_params['database']}"
        )
        self.engine = create_engine(connection_string, echo=False)

        if not database_exists(self.engine.url):
            print_info(f"Creating database '{self.db_params['database']}'...")
            create_database(self.engine.url)
            print_success(f"Database '{self.db_params['database']}' created successfully")
        else:
            print_info(f"Using existing database '{self.db_params['database']}'")

    def load_data(self):
        """
        Load data from CSV files or SQL database.
        
        If use_sql is True, the method first loads data from CSV files into SQL tables,
        then reads the data from those tables. Otherwise, it reads directly from CSV files.
        Each dataset is stored in the dfs dictionary with the dataset name as the key.
        
        Returns:
            None
        """
        print_subsection_header("Loading Datasets")
        
        if self.use_sql:
            if self.engine is None:
                self.create_db_engine()
            
            for file_path in self.file_paths:
                # Use os.path to extract the base name without extension
                dataset_name = os.path.splitext(os.path.basename(file_path))[0]
                print_info(f"Loading dataset '{dataset_name}' from CSV file...")
                df_local = pd.read_csv(file_path, delimiter=',', decimal=".", index_col=0)
                print_success(f"Successfully loaded dataset '{dataset_name}' ({len(df_local)} rows)")
                
                print_info(f"Saving dataset '{dataset_name}' to SQL database...")
                df_local.to_sql(f'{dataset_name}_table', self.engine, if_exists='replace', index=False)
                print_success(f"Dataset '{dataset_name}' saved to database")
            
            for file_path in self.file_paths:
                dataset_name = os.path.splitext(os.path.basename(file_path))[0]
                print_info(f"Loading dataset '{dataset_name}' from SQL database...")
                self.dfs[dataset_name] = pd.read_sql_table(f'{dataset_name}_table', self.engine)
                print_success(f"Successfully loaded dataset '{dataset_name}' ({len(self.dfs[dataset_name])} rows)")
        else:
            for file_path in self.file_paths:
                dataset_name = os.path.splitext(os.path.basename(file_path))[0]
                print_info(f"Loading dataset '{dataset_name}' from CSV file...")
                self.dfs[dataset_name] = pd.read_csv(file_path, delimiter=',', decimal=".", index_col=0)
                print_success(f"Successfully loaded dataset '{dataset_name}' ({len(self.dfs[dataset_name])} rows)")
        
        for dataset_name, df in self.dfs.items():
            self.dfs[dataset_name] = df.dropna()
            print_info(f"Removed {df.shape[0] - self.dfs[dataset_name].shape[0]} rows with missing values from '{dataset_name}'")
        
        print_success(f"Loaded {len(self.dfs)} datasets")

    def preprocess_data(self):
        """
        Preprocess data and create train/test splits.
        
        For each dataset, this method:
        1. Separates features (X) and target variable (y)
        2. Applies log transformation to the target variable (defects)
        3. Creates train/test splits with 70/30 ratio
        4. Stores the splits in their respective dictionaries
        
        Returns:
            None
        """
        print_subsection_header("Preprocessing Data")
        
        for dataset_name, df in self.dfs.items():
            print_info(f"Preprocessing dataset '{dataset_name}'...")
            
            X = df.drop('defects', axis=1)
            y = np.log(df['defects'] + 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
            print_info(f"Split '{dataset_name}' into {len(X_train)} training and {len(X_test)} testing samples")
            
            self.X_trains[dataset_name] = X_train
            self.X_tests[dataset_name] = X_test
            self.y_trains[dataset_name] = y_train
            self.y_tests[dataset_name] = y_test
            
            print_success(f"Preprocessing complete for dataset '{dataset_name}'")
            
        print_success(f"Preprocessing complete for all {len(self.dfs)} datasets")

    def scale_features(self):
        """
        Scale features using StandardScaler.
        
        For each dataset, this method:
        1. Creates a StandardScaler instance
        2. Fits the scaler on the training data and transforms both training and test data
        3. Converts the scaled arrays back to DataFrames with the original column names
        
        Returns:
            None
        """
        print_subsection_header("Scaling Features")
        
        for dataset_name in self.dfs.keys():
            print_info(f"Scaling features for dataset '{dataset_name}'...")
            scaler = StandardScaler()
            self.X_trains[dataset_name] = pd.DataFrame(scaler.fit_transform(self.X_trains[dataset_name]), 
                                                       columns=self.X_trains[dataset_name].columns)
            self.X_tests[dataset_name] = pd.DataFrame(scaler.transform(self.X_tests[dataset_name]), 
                                                      columns=self.X_tests[dataset_name].columns)
            print_success(f"Scaling complete for dataset '{dataset_name}'")
            
        print_success(f"Scaling complete for all {len(self.dfs)} datasets")

    def get_data(self):
        """
        Get the prepared data splits.
        
        Returns:
            tuple: A tuple containing dictionaries of (X_trains, X_tests, y_trains, y_tests)
        """
        return self.X_trains, self.X_tests, self.y_trains, self.y_tests

    def print_eda(self):
        """
        Print exploratory data analysis information for each dataset.
        
        For each dataset, this method prints:
        1. Statistical summary
        2. Dataset information
        3. Count of null values per column
        4. Correlations with the target variable
        5. Distribution of the target variable
        
        Returns:
            None
        """
        print_subsection_header("Exploratory Data Analysis")
        
        for dataset_name, df in self.dfs.items():
            # Extract just the base name without path
            base_name = dataset_name.split('/')[-1] if '/' in dataset_name else dataset_name
            base_name = base_name.split('\\')[-1] if '\\' in base_name else base_name
            
            print_subsection_header(f"EDA for dataset: {base_name}")
            
            # Statistical summary
            print_summary_statistics("Statistical Summary", {
                "Rows": len(df),
                "Columns": len(df.columns),
                "Missing Values": df.isnull().sum().sum(),
                "Target Mean": df['defects'].mean(),
                "Target Std": df['defects'].std()
            })
            
            # Column information
            col_info = []
            for col in df.columns:
                col_info.append([
                    col, 
                    df[col].dtype, 
                    df[col].isnull().sum(),
                    df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else "N/A"
                ])
            
            print_info("Column Information:")
            print_table(["Column", "Type", "Null Count", "Mean"], col_info)
            
            # Correlations with target
            if 'defects' in df.columns:
                print_info("\nCorrelations with target variable 'defects':")
                corrs = df.corr()['defects'].sort_values(ascending=False)
                corr_data = [[col, corrs[col]] for col in corrs.index if col != 'defects']
                print_table(["Feature", "Correlation"], corr_data)
            
            # Target distribution
            if 'defects' in df.columns:
                print_info("\nDistribution of the target variable 'defects':")
                value_counts = df['defects'].value_counts(normalize=True)
                dist_data = [[val, f"{count:.2%}"] for val, count in value_counts.items()]
                print_table(["Value", "Percentage"], dist_data)
            
            print("\n" + "=" * 50)

    def process(self):
        """
        Execute the entire data loading and preprocessing pipeline.
        
        This method orchestrates the complete data processing workflow by calling:
        1. load_data() - Load data from files or database
        2. print_eda() - Print exploratory data analysis
        3. preprocess_data() - Preprocess and split the data
        4. scale_features() - Scale the features
        
        Returns:
            None
        """
        self.load_data()
        self.print_eda()
        self.preprocess_data()
        self.scale_features()