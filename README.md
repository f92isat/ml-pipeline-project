# 🤖 Pipeline Automatizado de Machine Learning

Pipeline reproducible de machine learning con automatización CI/CD usando GitHub Actions y MLflow.

## 📋 Descripción del Proyecto

Este proyecto implementa un pipeline completo de MLOps que incluye:
- ✅ Preprocesamiento automatizado de datos
- ✅ Entrenamiento de modelos de ML
- ✅ Evaluación con múltiples métricas
- ✅ Tracking con MLflow
- ✅ CI/CD con GitHub Actions

## 🎯 Dataset

**Dataset:** Heart Disease UCI Dataset  
**Fuente:** UCI Machine Learning Repository  
**Tarea:** Clasificación binaria (predicción de enfermedad cardíaca)  
**Características:** 13 features clínicos  
**Target:** 0 = No enfermedad, 1 = Enfermedad presente  
**Muestras:** 303 pacientes

### Features del Dataset:

1. **age:** Edad en años
2. **sex:** Sexo (1 = masculino, 0 = femenino)
3. **cp:** Tipo de dolor de pecho (0-3)
4. **trestbps:** Presión arterial en reposo (mm Hg)
5. **chol:** Colesterol sérico (mg/dl)
6. **fbs:** Glucemia en ayunas > 120 mg/dl
7. **restecg:** Resultados electrocardiográficos
8. **thalach:** Frecuencia cardíaca máxima alcanzada
9. **exang:** Angina inducida por ejercicio
10. **oldpeak:** Depresión del ST inducida por ejercicio
11. **slope:** Pendiente del segmento ST
12. **ca:** Número de vasos principales (0-3)
13. **thal:** Thalassemia

**Accuracy esperada:** 78-85%

### Sobre el Dataset:

Este dataset proviene de la Cleveland Clinic Foundation y ha sido utilizado en múltiples 
investigaciones científicas desde 1988. Es considerado un benchmark estándar para problemas 
de clasificación en diagnóstico médico.

**Referencia:**
> Detrano, R. (1989). Heart Disease Data Set. UCI Machine Learning Repository.

## 🏗️ Estructura del Proyecto
```
ml-pipeline-project/
├── .github/
│   └── workflows/
│       └── ml.yml          # CI/CD con GitHub Actions
├── src/
│   ├── train.py           # Script principal
│   ├── preprocess.py      # Preprocesamiento
│   ├── evaluate.py        # Evaluación
│   └── __init__.py
├── tests/
│   ├── test_pipeline.py   # Tests unitarios
│   └── __init__.py
├── data/
│   └── heart.csv          # Dataset
├── mlruns/                # Experimentos MLflow
├── config.yaml            # Configuración
├── Makefile               # Automatización
├── requirements.txt       # Dependencias
├── .gitignore
└── README.md
```

## 🚀 Instalación (Windows)

### Requisitos Previos
- Python 3.9+
- pip
- Git

### Pasos de Instalación

1. **Clonar el repositorio:**
```powershell
git clone https://github.com/TU-USUARIO/ml-pipeline-project.git
cd ml-pipeline-project
```

2. **Crear entorno virtual:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

3. **Instalar dependencias:**
```powershell
pip install -r requirements.txt
```

4. **Descargar el dataset:**
```powershell
cd data
curl -o heart.csv https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
cd ..
```

## 💻 Uso

### Entrenamiento Local
```powershell
# Ejecutar pipeline completo
python src\train.py

# O con Make (si está instalado)
make train
```

### Ejecutar Tests
```powershell
# Ejecutar tests
pytest tests -v

# O con Make
make test
```

### Ver Experimentos en MLflow
```powershell
mlflow ui
```
Abrir en navegador: http://localhost:5000

## 🔧 Configuración

Editar `config.yaml` para modificar:
```yaml
model:
  type: "xgboost"        # xgboost o random_forest
  params:
    max_depth: 4
    learning_rate: 0.05
    n_estimators: 150
```

### Parámetros Configurables:

- **data.test_size:** Proporción del conjunto de prueba (default: 0.2)
- **preprocessing.scale_features:** Activar/desactivar escalado (default: true)
- **model.type:** Tipo de modelo (xgboost, random_forest)
- **model.params:** Hiperparámetros del modelo
- **evaluation.metrics:** Métricas a calcular

## 🤖 CI/CD con GitHub Actions

El pipeline se ejecuta automáticamente en cada push a `main` o `develop`.

### Pasos del Workflow:

1. ✅ Checkout del código
2. 🐍 Configuración de Python 3.10
3. 📦 Instalación de dependencias
4. 📥 Descarga del dataset
5. 🔍 Lint con Flake8
6. 🧪 Ejecución de tests
7. 🎯 Entrenamiento del modelo
8. 💾 Guardado de artefactos (30 días)

### Ver el Workflow:

1. Ve a tu repositorio en GitHub
2. Click en la pestaña "Actions"
3. Selecciona el workflow más reciente

## 📊 Métricas del Modelo

El modelo es evaluado con:

- **Accuracy:** Precisión general de clasificación
- **F1-Score:** Balance entre precisión y recall
- **ROC-AUC:** Capacidad de discriminación del modelo

### Resultados Típicos:

- Accuracy: 78-85%
- F1-Score: 75-83%
- ROC-AUC: 82-88%

## 📦 Características Principales

### Preprocesamiento
- Manejo automático de valores faltantes
- Escalado de características (StandardScaler)
- División estratificada train/test (80/20)
- Codificación de target binario

### Modelo
- XGBoost Classifier (por defecto)
- Hiperparámetros optimizados
- Soporte para Random Forest
- Regularización para prevenir overfitting

### MLflow Tracking
- Registro automático de parámetros e hiperparámetros
- Tracking de métricas de evaluación
- Guardado de modelos con signature y metadata
- Artifacts: modelo, métricas, resultados CSV

### Testing
- Tests unitarios con pytest
- Validación de configuración
- Verificación de carga de datos
- Cobertura de código

## 🛠️ Tecnologías Utilizadas

- **Python 3.10**
- **scikit-learn:** Preprocesamiento y métricas
- **XGBoost:** Modelo de clasificación
- **MLflow:** Tracking y registro de modelos
- **GitHub Actions:** CI/CD
- **pytest:** Testing
- **flake8:** Linting
- **black:** Formateo de código
- **PyYAML:** Gestión de configuración

## 📝 Comandos Útiles
```powershell
# Instalar dependencias
pip install -r requirements.txt

# Entrenar modelo
python src\train.py

# Ejecutar tests
pytest tests -v

# Ver MLflow UI
mlflow ui

# Limpiar archivos temporales
make clean

# Formatear código
black src tests

# Verificar código
flake8 src
```

## 🐛 Solución de Problemas

### Error: "No module named 'preprocess'"
```powershell
# Asegúrate de ejecutar desde la raíz del proyecto
cd ml-pipeline-project
python src\train.py
```

### Error: "Dataset not found"
```powershell
# Verificar que el dataset existe
ls data\heart.csv

# Si no existe, descargarlo
cd data
curl -o heart.csv https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
cd ..
```

### Error al activar entorno virtual
```powershell
# Dar permisos de ejecución
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# O usar activate.bat
venv\Scripts\activate.bat
```

### Error: "xgboost not found"
```powershell
# Activar entorno virtual e instalar
.\venv\Scripts\Activate.ps1
pip install xgboost
```

## 📚 Recursos y Referencias

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

## 🎓 Para la Sustentación

### Puntos Clave:

1. **Arquitectura del Pipeline:**
   - Diseño modular con separación de responsabilidades
   - Configuración centralizada en YAML
   - Flujo de datos claro y reproducible

2. **Preprocesamiento:**
   - Manejo robusto de valores faltantes
   - Escalado necesario para features de diferentes magnitudes
   - División estratificada para mantener proporciones de clases

3. **Modelo:**
   - XGBoost elegido por su rendimiento en clasificación
   - Hiperparámetros configurados para evitar overfitting
   - Resultados realistas (78-85% accuracy)

4. **MLflow:**
   - Tracking completo de experimentos
   - Registro de modelos con metadata
   - Reproducibilidad garantizada

5. **CI/CD:**
   - Automatización completa con GitHub Actions
   - Tests antes de cada entrenamiento
   - Artefactos disponibles para descarga

6. **Resultados:**
   - Métricas consistentes con literatura científica
   - No hay evidencia de overfitting
   - Modelo generaliza bien a datos no vistos

## 👥 Autor

**Tu Nombre**  
Proyecto de MLOps - Pipeline Automatizado de ML  
Universidad - Año 2025

## 📄 Licencia

MIT License

---

⭐ Si este proyecto te fue útil, dale una estrella en GitHub!

## 🔗 Enlaces

- [Repositorio GitHub](https://github.com/TU-USUARIO/ml-pipeline-project)
- [Issues](https://github.com/TU-USUARIO/ml-pipeline-project/issues)
- [Pull Requests](https://github.com/TU-USUARIO/ml-pipeline-project/pulls)