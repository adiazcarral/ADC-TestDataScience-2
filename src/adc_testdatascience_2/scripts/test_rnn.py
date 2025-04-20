import torch
import matplotlib.pyplot as plt
import numpy as np
from src.adc_testdatascience_2.models.rnn import SimpleRNN
from src.adc_testdatascience_2.utils.data_utils import get_dataloaders
import pickle
import os

# Paths
model_path = "src/adc_testdatascience_2/models/rnn_direct.pth"
scaler_path = "src/adc_testdatascience_2/data/processed_energy_scaler.pkl"
plot_path = "src/adc_testdatascience_2/scripts/plots/rnn_direct_forecast.png"
csv_path = "src/adc_testdatascience_2/data/processed_energy.csv"

# Constants
input_window = 1000
forecast_horizon = 100
input_dim = 25
hidden_dim = 64
num_layers = 2

# Load data
_, _, test_loader = get_dataloaders(
    csv_path=csv_path,
    input_window=input_window,
    output_window=forecast_horizon
)

# Take the first batch and first sample
X_test, y_test = next(iter(test_loader))
X_test = X_test[0].unsqueeze(0)  # shape: (1, 500, 25)
y_test = y_test[0].numpy()       # shape: (100,)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleRNN(input_dim=input_dim, hidden_dim=hidden_dim,
                  output_dim=forecast_horizon, num_layers=num_layers).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load the scaler for inverse transformation
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Forecast directly
with torch.no_grad():
    X_test = X_test.to(device)
    prediction = model(X_test).cpu().numpy().flatten()

# # Inverse transformation of the 'Appliances' values only
# # Create an array of shape (100, 26) for inverse transformation
# # Set all input features to 0 and apply to only the 'Appliances' column
# y_test_with_zeros = np.zeros((y_test.shape[0], 26))
# prediction_with_zeros = np.zeros((prediction.shape[0], 26))

# # Set the last column to the true and predicted 'Appliances' values
# y_test_with_zeros[:, -1] = y_test
# prediction_with_zeros[:, -1] = prediction

# # Inverse transform both arrays
# y_test = scaler.inverse_transform(y_test_with_zeros)[:, -1]
# prediction = scaler.inverse_transform(prediction_with_zeros)[:, -1]

# Plot results
plt.figure(figsize=(12, 5))
plt.plot(y_test, label="True", linewidth=2)
plt.plot(prediction, label="Forecast", linestyle='--')
plt.title("Direct RNN Forecast (100 steps) - Inverse Transformed")
plt.xlabel("Future time steps")
plt.ylabel("Appliances")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path)
plt.show()

print("✅ Direct forecast complete with inverse transformation.")
