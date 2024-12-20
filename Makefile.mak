PYTHON = python
CLEAN_SCRIPT = etl/clean_data.py
TRAIN_INPUT = data/train.csv
TEST_INPUT = data/test.csv
TRAIN_OUTPUT = data/train_clean.csv
TEST_OUTPUT = data/test_clean.csv

.PHONY: all
all: etl

.PHONY: etl
etl:
	$(PYTHON) $(CLEAN_SCRIPT) --train_input $(TRAIN_INPUT) --test_input $(TEST_INPUT) --train_output $(TRAIN_OUTPUT) --test_output $(TEST_OUTPUT)

.PHONY: install
install:
	pip install -r requirements.txt

.PHONY: freeze
freeze:
	pip freeze > requirements.txt
