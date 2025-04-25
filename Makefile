.PHONY: help install train test lint format cyclo clean docker_build docker_run

# Set Python interpreter (adjust if needed for Windows or specific environment)
PYTHON := python

# Set the root directory for source files
SRC := src.adc_testdatascience_2

help:
	@echo "Available commands:"
	@echo "  install        Install dependencies"
	@echo "  train          Train the model"
	@echo "  test           Run model evaluation"
	@echo "  lint           Run ruff and flake8"
	@echo "  format         Auto-format code (black, isort)"
	@echo "  cyclo          Check cyclomatic complexity"
	@echo "  clean          Remove __pycache__ and .pyc files"
	@echo "  docker_build   Build the Docker image"
	@echo "  docker_run     Run the Docker container"

install:
	# Install dependencies using poetry
	poetry install

train:
	# Train the model
	$(PYTHON) -m $(SRC).scripts.train_model --model=rnn

test:
	# Run model evaluation
	$(PYTHON) -m $(SRC).scripts.test_model --model=lstmvae

lint:
	# Run ruff and flake8 to check code quality
	ruff check $(SRC)
	flake8 $(SRC)

format:
	# Auto-format code using black and isort
	black . && isort .

cyclo:
	# Check cyclomatic complexity using radon
	radon cc $(SRC) -a

clean:
	# Clean __pycache__ and .pyc files
	find . -type d -name "__pycache__" -exec rm -r {} \; && find . -name "*.pyc" -delete

docker_build:
	# Build the Docker image for the project
	docker build -t adc-testdatascience-2 .

docker_run:
	# Run the Docker container for the project
	docker run --rm -v $(pwd):/app adc-testdatascience-2
