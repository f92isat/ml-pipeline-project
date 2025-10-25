.PHONY: help install train test lint format clean all

help:
	@echo Comandos disponibles:
	@echo   make install  - Instalar dependencias
	@echo   make train    - Entrenar el modelo
	@echo   make test     - Ejecutar tests
	@echo   make lint     - Verificar código con flake8
	@echo   make format   - Formatear código con black
	@echo   make clean    - Limpiar archivos temporales
	@echo   make all      - Ejecutar todo el pipeline

install:
	@echo Instalando dependencias...
	python -m pip install --upgrade pip
	pip install -r requirements.txt

train:
	@echo Ejecutando pipeline de entrenamiento...
	python src\train.py

test:
	@echo Ejecutando tests...
	pytest tests -v --tb=short

lint:
	@echo Verificando código...
	flake8 src --max-line-length=100 --exclude=__pycache__

format:
	@echo Formateando código...
	black src tests

clean:
	@echo Limpiando archivos temporales...
	@if exist __pycache__ rmdir /s /q __pycache__
	@if exist src\__pycache__ rmdir /s /q src\__pycache__
	@if exist tests\__pycache__ rmdir /s /q tests\__pycache__
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist results.csv del /q results.csv
	@echo Limpieza completada

all: install lint test train