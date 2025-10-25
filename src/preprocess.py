"""
Módulo de preprocesamiento de datos - Heart Disease Dataset
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Clase para preprocesar datos del pipeline de ML"""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """Cargar datos desde CSV"""
        logger.info(f"Cargando datos desde {file_path}")
        
        # Nombres de columnas para Heart Disease dataset
        column_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
            'ca', 'thal', 'target'
        ]
        
        # Leer CSV
        df = pd.read_csv(file_path, names=column_names, na_values='?')
        logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Mostrar info del target
        logger.info(f"Distribución del target:\n{df['target'].value_counts().sort_index()}")
        
        return df
    
    def handle_missing_values(self, df):
        """Manejar valores faltantes"""
        strategy = self.config['preprocessing']['handle_missing']
        
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            logger.info("No se encontraron valores faltantes")
            return df
        
        logger.info(f"Valores faltantes encontrados: {missing_count}")
        logger.info(f"Manejando valores faltantes con estrategia: {strategy}")
        
        if strategy == "mean":
            return df.fillna(df.mean())
        elif strategy == "median":
            return df.fillna(df.median())
        elif strategy == "drop":
            df_clean = df.dropna()
            logger.info(f"Filas después de eliminar NaN: {len(df_clean)}")
            return df_clean
        else:
            return df
    
    def encode_target(self, df, target_column):
        """Convertir variable objetivo a binaria"""
        if target_column == 'target':
            # Heart Disease: 0 = no disease, 1-4 = disease present
            # Convertir a binario: 0 = no disease, 1 = disease
            df['target_binary'] = (df[target_column] > 0).astype(int)
            
            logger.info("Target codificado:")
            logger.info(f"  0 = No enfermedad cardíaca")
            logger.info(f"  1 = Enfermedad cardíaca presente")
            logger.info(f"Distribución binaria:\n{df['target_binary'].value_counts()}")
            
            return df, 'target_binary'
        return df, target_column
    
    def split_data(self, df, target_column):
        """Dividir datos en train y test"""
        X = df.drop(columns=[target_column, 'target'] if 'target' in df.columns else [target_column])
        y = df[target_column]
        
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Datos divididos: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
        logger.info(f"Features utilizados: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Escalar características"""
        if not self.config['preprocessing']['scale_features']:
            return X_train, X_test
        
        logger.info("Escalando características...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return pd.DataFrame(X_train_scaled, columns=X_train.columns), \
               pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    def preprocess(self, file_path, target_column='target'):
        """Pipeline completo de preprocesamiento"""
        # Cargar datos
        df = self.load_data(file_path)
        
        # Manejar valores faltantes
        df = self.handle_missing_values(df)
        
        # Codificar target
        df, target_col = self.encode_target(df, target_column)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = self.split_data(df, target_col)
        
        # Escalar características
        X_train, X_test = self.scale_features(X_train, X_test)
        
        logger.info("Preprocesamiento completado exitosamente")
        
        return X_train, X_test, y_train, y_test