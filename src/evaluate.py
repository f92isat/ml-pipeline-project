"""
Módulo de evaluación de modelos
"""
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Clase para evaluar modelos de ML"""
    
    def __init__(self, config):
        self.config = config
    
    def evaluate(self, model, X_test, y_test):
        """Evaluar modelo con múltiples métricas"""
        logger.info("Evaluando modelo...")
        
        # Predicciones
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calcular métricas
        metrics = {}
        
        if "accuracy" in self.config['evaluation']['metrics']:
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
        
        if "f1_score" in self.config['evaluation']['metrics']:
            metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted')
        
        if "roc_auc" in self.config['evaluation']['metrics']:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        # Log de resultados
        logger.info("Métricas de evaluación:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        # Reporte de clasificación
        logger.info("\nReporte de clasificación:")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        return metrics