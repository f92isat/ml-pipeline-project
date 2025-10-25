"""
Script principal de entrenamiento del pipeline de ML
"""
import yaml
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import logging
import sys
from pathlib import Path

# Importar módulos locales
from preprocess import DataPreprocessor
from evaluate import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path='config.yaml'):
    """Cargar configuración desde YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config):
    """Crear modelo según configuración"""
    model_type = config['model']['type']
    params = config['model']['params']
    
    logger.info(f"Creando modelo tipo: {model_type}")
    
    if model_type == "xgboost":
        model = XGBClassifier(**params)
    elif model_type == "random_forest":
        model = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")
    
    return model


def main():
    """Pipeline principal de ML"""
    try:
        # 1. Cargar configuración
        logger.info("=" * 60)
        logger.info("INICIANDO PIPELINE DE MACHINE LEARNING")
        logger.info("=" * 60)
        
        config = load_config()
        
        # 2. Configurar MLflow
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        mlflow.set_experiment(config['mlflow']['experiment_name'])
        
        # 3. Iniciar run de MLflow
        with mlflow.start_run():
            
            # 4. Preprocesar datos
            logger.info("\n[1/4] PREPROCESAMIENTO DE DATOS")
            preprocessor = DataPreprocessor(config)
            X_train, X_test, y_train, y_test = preprocessor.preprocess(
                config['data']['raw_path']
            )
            
            # Log de información del dataset
            mlflow.log_param("dataset_name", "heart-disease")
            mlflow.log_param("n_samples_train", len(X_train))
            mlflow.log_param("n_samples_test", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            
            # 5. Entrenar modelo
            logger.info("\n[2/4] ENTRENAMIENTO DEL MODELO")
            model = create_model(config)
            
            # Log de hiperparámetros
            mlflow.log_params(config['model']['params'])
            mlflow.log_param("model_type", config['model']['type'])
            
            model.fit(X_train, y_train)
            logger.info("Modelo entrenado exitosamente")
            
            # 6. Evaluar modelo
            logger.info("\n[3/4] EVALUACIÓN DEL MODELO")
            evaluator = ModelEvaluator(config)
            metrics = evaluator.evaluate(model, X_test, y_test)
            
            # Log de métricas
            mlflow.log_metrics(metrics)
            
            # 7. Registrar modelo en MLflow
            logger.info("\n[4/4] REGISTRO DEL MODELO")
            
            # Crear signature y ejemplo de entrada
            from mlflow.models.signature import infer_signature
            signature = infer_signature(X_train, model.predict(X_train))
            
            # Guardar modelo
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.iloc[:5],
                registered_model_name=config['mlflow']['registered_model_name']
            )
            
            logger.info(f"Modelo registrado como: {config['mlflow']['registered_model_name']}")
            
            # Guardar artefactos adicionales
            import pandas as pd
            results_df = pd.DataFrame([metrics])
            results_df.to_csv("results.csv", index=False)
            mlflow.log_artifact("results.csv")
            
            logger.info("\n" + "=" * 60)
            logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
            logger.info("=" * 60)
            logger.info(f"\nMétricas finales:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            return 0
            
    except Exception as e:
        logger.error(f"Error en el pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())