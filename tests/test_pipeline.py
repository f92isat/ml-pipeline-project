"""
Tests básicos para el pipeline de ML
"""
import pytest
import pandas as pd
import yaml
import sys
from pathlib import Path

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocess import DataPreprocessor
from evaluate import ModelEvaluator


def load_config():
    """Cargar configuración de prueba"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


def test_config_file_exists():
    """Verificar que existe el archivo de configuración"""
    assert Path('config.yaml').exists()


def test_data_file_exists():
    """Verificar que existe el archivo de datos"""
    config = load_config()
    data_path = Path(config['data']['raw_path'])
    assert data_path.exists(), f"No se encuentra el archivo: {data_path}"


def test_preprocessor_initialization():
    """Test de inicialización del preprocesador"""
    config = load_config()
    preprocessor = DataPreprocessor(config)
    assert preprocessor is not None
    assert preprocessor.config == config


def test_data_loading():
    """Test de carga de datos"""
    config = load_config()
    preprocessor = DataPreprocessor(config)
    df = preprocessor.load_data(config['data']['raw_path'])
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert len(df.columns) > 0


def test_evaluator_initialization():
    """Test de inicialización del evaluador"""
    config = load_config()
    evaluator = ModelEvaluator(config)
    assert evaluator is not None


def test_config_structure():
    """Test de estructura del archivo de configuración"""
    config = load_config()
    
    # Verificar secciones principales
    assert 'data' in config
    assert 'preprocessing' in config
    assert 'model' in config
    assert 'mlflow' in config
    assert 'evaluation' in config
    
    # Verificar parámetros del modelo
    assert 'type' in config['model']
    assert 'params' in config['model']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])