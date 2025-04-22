import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.adc_testdatascience_2.models.rnn import SimpleRNN
from src.adc_testdatascience_2.utils.data_utils import get_dataloaders
import pickle
import os

# --- CONFIG ---
csv_path = "src/adc_testdatascience_2/data/processed_energy.csv"
scaler_path = "src/adc_testdatascience_2/data/processed_energy_scaler.pkl"
model_path = "src/adc_testdatascience_2/models/rnn_1step.pth"
plot_path = "src/adc_testdatascience_2/scripts/plots/rnn_onestep_recursive_forecast.png"

input_window = 100
forecast_steps = 100
input_dim = 26
hidden_dim = 64
num_layers = 2

# --- Load and preprocess data ---
df = pd.read_csv(csv_path)
data = df.values
appliances_idx = df.columns.get_loc("Appliances")

# Initialize input with the last available input_window before forecast starts
start_idx = 0
X_window = data[start_idx:start_idx + input_window].copy()  # shape (100, 26)
y_true = data[start_idx + input_window:start_idx + input_window + forecast_steps, appliances_idx]

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleRNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Forecast 1 step ahead recursively for 100 steps
predictions = []

with torch.no_grad():
    for step in range(forecast_steps):
        x_input_tensor = torch.tensor(X_window, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 100, 26)
        pred = model(x_input_tensor).cpu().numpy().flatten()[0]
        predictions.append(pred)

        # Build new input window: remove oldest, add next true multivariate vector
        next_input = data[start_idx + input_window + step]
        X_window = np.vstack([X_window[1:], next_input])  # move window forward

# --- Plot ---
plt.figure(figsize=(12, 5))
plt.plot(y_true, label="True", linewidth=2)
plt.plot(predictions, label="Forecast", linestyle='--')
plt.title("RNN 1-Step Recursive Forecast (100 Steps)")
plt.xlabel("Future Time Steps")
plt.ylabel("Appliances (normalized)")
plt.legend()
plt.grid(True)
plt.tight_layout()
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path)
plt.show()

print("✅ 1-step recursive forecast complete and plotted.")
