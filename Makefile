.PHONY: help install train test lint format cyclo clean

# Set Python interpreter (adjust if needed for Windows or specific environment)
PYTHON := python

# Set the root directory for source files
SRC := src.adc_testdatascience_1

help:
	@echo "Available commands:"
	@echo "  install      Install dependencies"
	@echo "  train        Train the model"
	@echo "  test         Run model evaluation"
	@echo "  lint         Run ruff and flake8"
	@echo "  format       Auto-format code (black, isort)"
	@echo "  cyclo        Check cyclomatic complexity"
	@echo "  clean        Remove __pycache__ and .pyc files"

install:
	poetry install

train:
	$(PYTHON) -m $(SRC).scripts.train_model --model=logistic

test:
	$(PYTHON) -m $(SRC).scripts.test_model --model=rotcnn

lint:
	ruff check tests
	flake8 tests

format:
	black . && isort .

cyclo:
	radon cc $(SRC) -a

optimize:
	$(PYTHON) -m $(SRC).optimize.optimize --model LogisticRegression

clean:
	find . -type d -name "__pycache__" -exec rm -r {} \; && find . -name "*.pyc" -delete

