# Makefile

.PHONY: clean install freeze

# Variable para el comando Python
PYTHON = python

# Script de limpieza
CLEAN_SCRIPT = etl/clean_data.py

# Objetivos del ETL
clean:
	$(PYTHON) $(CLEAN_SCRIPT) --train-input=data/raw/train.csv --test-input=data/raw/test.csv --train-output=data/clean/train_clean.csv --test-output=data/clean/test_clean.csv

# Instalar dependencias
install:
	pip install -r requirements.txt

# Congelar dependencias
freeze:
	pip freeze > requirements.txt

