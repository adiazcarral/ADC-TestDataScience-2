# ADC-TestDataScience-2

[![image](https://img.shields.io/pypi/v/adc_testdatascience_2.svg)](https://pypi.python.org/pypi/adc_testdatascience_2)

[![image](https://img.shields.io/travis/adiazcarral/adc_testdatascience_2.svg)](https://travis-ci.com/adiazcarral/adc_testdatascience_2)

[![Documentation Status](https://readthedocs.org/projects/adc-testdatascience-2/badge/?version=latest)](https://adc-testdatascience-2.readthedocs.io/en/latest/?version=latest)

## Test 2 - Time Series Forecasting (Regression)

This repository includes code and models for time series forecasting, specifically targeting the Appliances Energy dataset. The project compares deterministic and probabilistic models, including an RNN-based model and a probabilistic LSTM-VAE model for forecasting appliance energy consumption.

- Free software: MIT license
- Documentation: <https://adc-testdatascience-2.readthedocs.io>.

## Deliverables

The following deliverables are included in this repository to replicate the analysis and deploy the trained model:

1. **Report**: A detailed report outlining the methodology, models, and results of the time series forecasting task.
2. **Jupyter Notebooks**: 
   - **1. EDA (Exploratory Data Analysis)**: Preprocessing, visualization, and understanding of the Appliances Energy dataset.
   - **2. Training**: Training both RNN and LSTM-VAE models, comparing direct (100 steps) and one-step forecasting strategies.
   - **3. Evaluation**: Evaluation of the models using key metrics such as MAE, RMSE, R2, and CRPS.
3. **Python Package**: 
   - Trained models saved as `.pkl` files.
   - Encapsulated code for model training, evaluation, and deployment.
4. **GitHub Repository**: 
   - [Link to GitHub repository](https://github.com/adiazcarral/adc_testdatascience_2) for cloning and accessing the project.

## Features

- Time series forecasting with both deterministic (RNN) and probabilistic (LSTM-VAE) models.
- Evaluation of models on normalized and denormalized data.
- Training strategies for direct (100-step) and one-step predictions.
- Detailed exploratory data analysis with visualizations.
- Model evaluation with key metrics: MAE, RMSE, R2, CRPS.

## Installation

To install and set up the project, use the following steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/adiazcarral/adc_testdatascience_2.git
   cd adc_testdatascience_2

	2.	Set up the environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


	3.	Install the dependencies:

pip install -r requirements.txt


	4.	For Docker setup, build the image:

docker build -t adc-testdatascience-2 .


	5.	Run the application (in the virtual environment or Docker container):

python app.py



Usage

Once the package is installed, you can use it to train, evaluate, and test the models:
	1.	Training:
Use the provided Jupyter notebooks (Training notebook) to train the RNN and LSTM-VAE models.
	2.	Evaluation:
After training, evaluate the models using the test_lstm_vae_forecast() and similar functions provided in the notebooks.
	3.	Model Inference:
For inference, load the model from the .pkl file and use the provided methods to forecast appliance energy consumption.
	4.	Run the complete pipeline:
You can run the full pipeline (training, evaluation, and inference) using the provided Python scripts or Docker container.

Credits

This package was created with
Cookiecutter and the
audreyr/cookiecutter-pypackage
project template.

License

This project is licensed under the MIT License - see the LICENSE file for details.

### How to Serve and Build with MkDocs:

1. **Install MkDocs**: If you haven't already, install MkDocs with:

   ```bash
   pip install mkdocs

	2.	Build the Documentation:
Navigate to the root of the repository and run:

mkdocs build

This will generate the documentation in the site directory.

	3.	Serve the Documentation Locally:
To view the documentation locally, run:

mkdocs serve

This will start a local server, usually at http://127.0.0.1:8000, where you can preview the documentation.

	4.	Deploying Online:
You can deploy the documentation to GitHub Pages with:

mkdocs gh-deploy


## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)
project template.
