import torch
import matplotlib.pyplot as plt
import numpy as np
from src.adc_testdatascience_2.models.rnn import SimpleRNN
from src.adc_testdatascience_2.utils.data_utils import get_dataloaders
import pickle

model_path = "src/adc_testdatascience_2/models/rnn_autoreg.pth"
input_window = 500
forecast_horizon = 100

# Load test data
_, _, test_loader = get_dataloaders(
        csv_path="src/adc_testdatascience_2/data/processed_energy.csv",
        input_window=500, output_window=100
    )
X_test, y_test = next(iter(test_loader))  # take the first batch
X_test = X_test[0].unsqueeze(0)  # shape: (1, 500, 27)
y_test = y_test[0]  # shape: (100,)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the scaler
scaler_path = "src/adc_testdatascience_2/data/processed_energy_scaler.pkl"
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Load model
input_size = 25
hidden_size = 32
num_layers = 1
model = SimpleRNN(input_size, hidden_size, num_layers).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Autoregressive forecasting
with torch.no_grad():
    history = X_test.clone().to(device)  # current input window
    predictions = []

    for step in range(forecast_horizon):
        output = model(history)  # shape: (1, 1) - prediction for the next step (just 'Appliances')
        predictions.append(output.item())

        # Create the new input window
        next_input = torch.zeros(1, 1, history.shape[2]).to(device)
        
        # Keep all features except "Appliances" in the next input window
        next_input[0, 0, :-1] = history[0, -1, :-1]  # keep all features except 'Appliances'
        
        # Replace the last 'Appliances' value with the predicted value
        next_input[0, 0, -1] = output.item()  # predicted value for 'Appliances'

        # Slide the window by removing the first time step and adding the predicted value
        history = torch.cat([history[:, 1:, :], next_input], dim=1)

# Convert to arrays
predictions = np.array(predictions)
y_true = y_test.numpy()

# Plot
plt.figure(figsize=(12, 5))
plt.plot(y_true, label="True", linewidth=2)
plt.plot(predictions, label="Forecast", linestyle='--')
plt.title("Autoregressive RNN Forecast (100 steps)")
plt.xlabel("Future time steps")
plt.ylabel("Appliances (normalized)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("src/adc_testdatascience_2/scripts/plots/rnn_autoregressive_forecast.png")
plt.show()

print("✅ Autoregressive forecast complete.")
