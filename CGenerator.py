import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, URL
from sqlalchemy_utils import database_exists, create_database
import os

class DataGenerator:
    def __init__(self, file_paths, use_sql=False, db_params=None):
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
        """Crea y retorna el motor de conexión a la base de datos."""
        connection_string = (
            f"mysql+mysqlconnector://{self.db_params['username']}:{self.db_params['password']}@"
            f"{self.db_params['host']}:{self.db_params['port']}/{self.db_params['database']}"
        )
        self.engine = create_engine(connection_string, echo=False)

        if not database_exists(self.engine.url):
            create_database(self.engine.url)


    def load_data(self):
        if self.use_sql:
            if self.engine is None:
                self.create_db_engine()
            
            for file_path in self.file_paths:
                # Use os.path to extract the base name without extension
                dataset_name = os.path.splitext(os.path.basename(file_path))[0]
                df_local = pd.read_csv(file_path, delimiter=',', decimal=".", index_col=0)
                df_local.to_sql(f'{dataset_name}_table', self.engine, if_exists='replace', index=False)
            
            for file_path in self.file_paths:
                dataset_name = os.path.splitext(os.path.basename(file_path))[0]
                self.dfs[dataset_name] = pd.read_sql_table(f'{dataset_name}_table', self.engine)
        else:
            for file_path in self.file_paths:
                dataset_name = os.path.splitext(os.path.basename(file_path))[0]
                self.dfs[dataset_name] = pd.read_csv(file_path, delimiter=',', decimal=".", index_col=0)
        
        for dataset_name, df in self.dfs.items():
            self.dfs[dataset_name] = df.dropna()

    def preprocess_data(self):
        for dataset_name, df in self.dfs.items():
            X = df.drop('defects', axis=1)
            y = np.log(df['defects'] + 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
            self.X_trains[dataset_name] = X_train
            self.X_tests[dataset_name] = X_test
            self.y_trains[dataset_name] = y_train
            self.y_tests[dataset_name] = y_test

    def scale_features(self):
        for dataset_name in self.dfs.keys():
            scaler = StandardScaler()
            self.X_trains[dataset_name] = pd.DataFrame(scaler.fit_transform(self.X_trains[dataset_name]), 
                                                       columns=self.X_trains[dataset_name].columns)
            self.X_tests[dataset_name] = pd.DataFrame(scaler.transform(self.X_tests[dataset_name]), 
                                                      columns=self.X_tests[dataset_name].columns)

    def get_data(self):
        return self.X_trains, self.X_tests, self.y_trains, self.y_tests

    def print_eda(self):
        for dataset_name, df in self.dfs.items():
            # Extract just the base name without path
            base_name = dataset_name.split('/')[-1] if '/' in dataset_name else dataset_name
            base_name = base_name.split('\\')[-1] if '\\' in base_name else base_name
            
            print(f"\nEDA for dataset: {base_name}")
            print("\nResumen Estadístico:")
            print(df.describe())
            print("\nInformación del Dataset:")
            print(df.info())
            print("\nConteo de valores nulos por columna:")
            print(df.isnull().sum())
            print("\nCorrelaciones:")
            print(df.corr()['defects'].sort_values(ascending=False))
            print("\nDistribución de la variable objetivo 'defects':")
            print(df['defects'].value_counts(normalize=True))
            print("="*50)

    def process(self):
        """Ejecuta todo el proceso de carga y preprocesamiento de datos."""
        self.load_data()
        self.print_eda()
        self.preprocess_data()
        self.scale_features()