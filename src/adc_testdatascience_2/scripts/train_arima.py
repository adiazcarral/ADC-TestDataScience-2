import pandas as pd
from src.adc_testdatascience_2.models.arima import ARIMAForecaster
import pickle
import os

# Load preprocessed data (already scaled)
data_path = "src/adc_testdatascience_2/data/processed_energy.csv"
scaler_path = "src/adc_testdatascience_2/data/processed_energy_scaler.pkl"

df = pd.read_csv(data_path)
appliances_series = df["Appliances"].values

# Train/test split indices
input_window = 1000
forecast_horizon = 100
train_size = int(len(appliances_series) * 0.8)

# Prepare training data
train_series = appliances_series[:train_size]

# Initialize and fit model
arima_model = ARIMAForecaster(p=10, d=1, q=0)
arima_model.fit(train_series)

# Save model
model_path = "src/adc_testdatascience_2/models/arima_forecaster.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(arima_model, f)

print("✅ ARIMA model trained and saved.")
