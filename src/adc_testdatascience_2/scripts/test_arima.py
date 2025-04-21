import pandas as pd
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Load ARIMA model
model_path = "src/adc_testdatascience_2/models/arima_forecaster.pkl"
with open(model_path, 'rb') as f:
    arima_model = pickle.load(f)

# Load full data
data_path = "src/adc_testdatascience_2/data/processed_energy.csv"
df = pd.read_csv(data_path)
appliances_series = df["Appliances"].values

# Parameters
input_window = 1000
forecast_horizon = 100

# Extract test portion
start_index = int(len(appliances_series) * 0.8)
context = appliances_series[start_index - input_window:start_index]
true_future = appliances_series[start_index:start_index + forecast_horizon]

# Forecast using ARIMA
arima_model.fit(appliances_series[:start_index])  # Refit just in case
forecast = arima_model.forecast(steps=forecast_horizon).numpy()

# Plot
plt.figure(figsize=(12, 5))
plt.plot(np.arange(input_window), context, label="History (Appliances)", linewidth=2)
plt.plot(np.arange(input_window, input_window + forecast_horizon), true_future, label="True Future", color="green", linewidth=2)
plt.plot(np.arange(input_window, input_window + forecast_horizon), forecast, label="ARIMA Forecast", linestyle='--', color="orange")
plt.axvline(x=input_window, color='gray', linestyle=':', label="Forecast Start")

plt.title("ARIMA Forecast with Historical Context")
plt.xlabel("Time steps")
plt.ylabel("Appliances (normalized)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
plot_path = "src/adc_testdatascience_2/scripts/plots/arima_forecast_with_context.png"
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path)
plt.show()

print("✅ ARIMA forecast complete.")
