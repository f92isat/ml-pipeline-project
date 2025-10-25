"""
Script de diagnóstico para detectar overfitting
"""
import yaml
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import logging

from preprocess import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    logger.info("=" * 70)
    logger.info("DIAGNÓSTICO DE OVERFITTING - WINE QUALITY DATASET")
    logger.info("=" * 70)
    
    # Cargar config
    config = load_config()
    
    # Preprocesar datos
    logger.info("\n[1] CARGANDO Y PREPROCESANDO DATOS")
    preprocessor = DataPreprocessor(config)
    X_train, X_test, y_train, y_test = preprocessor.preprocess(
        config['data']['raw_path']
    )
    
    logger.info(f"\nTamaño del conjunto de entrenamiento: {len(X_train)}")
    logger.info(f"Tamaño del conjunto de prueba: {len(X_test)}")
    logger.info(f"Número de features: {X_train.shape[1]}")
    
    # Verificar distribución de clases
    logger.info("\n[2] DISTRIBUCIÓN DE CLASES")
    logger.info(f"Train - Clase 0: {(y_train == 0).sum()} | Clase 1: {(y_train == 1).sum()}")
    logger.info(f"Test  - Clase 0: {(y_test == 0).sum()} | Clase 1: {(y_test == 1).sum()}")
    
    # Crear modelo
    logger.info("\n[3] ENTRENANDO MODELO")
    model = XGBClassifier(**config['model']['params'])
    model.fit(X_train, y_train)
    
    # Evaluar en TRAIN (detectar memorización)
    logger.info("\n[4] EVALUACIÓN EN CONJUNTO DE ENTRENAMIENTO")
    train_score = model.score(X_train, y_train)
    logger.info(f"Accuracy en TRAIN: {train_score:.4f} ({train_score*100:.2f}%)")
    
    # Evaluar en TEST
    logger.info("\n[5] EVALUACIÓN EN CONJUNTO DE PRUEBA")
    test_score = model.score(X_test, y_test)
    logger.info(f"Accuracy en TEST: {test_score:.4f} ({test_score*100:.2f}%)")
    
    # Diferencia (señal de overfitting)
    diff = train_score - test_score
    logger.info(f"\n[6] DIFERENCIA (Train - Test): {diff:.4f}")
    
    if diff > 0.1:
        logger.warning("⚠️ OVERFITTING DETECTADO (diferencia > 10%)")
    elif train_score > 0.95 and test_score > 0.95:
        logger.warning("⚠️ POSIBLE DATA LEAKAGE (ambos scores > 95%)")
    else:
        logger.info("✅ No se detectó overfitting severo")
    
    # Validación cruzada
    logger.info("\n[7] VALIDACIÓN CRUZADA (5-Fold)")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    logger.info(f"CV Scores: {cv_scores}")
    logger.info(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Predicciones detalladas
    logger.info("\n[8] ANÁLISIS DE PREDICCIONES")
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    # Contar predicciones correctas
    correct_train = (y_pred_train == y_train).sum()
    correct_test = (y_pred_test == y_test).sum()
    
    logger.info(f"Predicciones correctas en TRAIN: {correct_train}/{len(y_train)}")
    logger.info(f"Predicciones correctas en TEST: {correct_test}/{len(y_test)}")
    
    # DIAGNÓSTICO FINAL
    logger.info("\n" + "=" * 70)
    logger.info("DIAGNÓSTICO FINAL")
    logger.info("=" * 70)
    
    if train_score == 1.0 and test_score == 1.0:
        logger.error("🚨 PROBLEMA CRÍTICO: Accuracy 100% en ambos conjuntos")
        logger.error("Posibles causas:")
        logger.error("  1. Data leakage (información del test en train)")
        logger.error("  2. Dataset muy simple para el modelo")
        logger.error("  3. Error en el preprocesamiento")
        logger.error("\n💡 RECOMENDACIÓN: Cambiar a otro dataset")
    elif train_score > 0.95:
        logger.warning("⚠️ OVERFITTING: Modelo memorizó los datos de entrenamiento")
        logger.warning("Soluciones:")
        logger.warning("  1. Reducir max_depth")
        logger.warning("  2. Aumentar learning_rate")
        logger.warning("  3. Reducir n_estimators")
        logger.warning("  4. Añadir regularización")
    else:
        logger.info("✅ Modelo parece estar funcionando correctamente")


if __name__ == "__main__":
    main()