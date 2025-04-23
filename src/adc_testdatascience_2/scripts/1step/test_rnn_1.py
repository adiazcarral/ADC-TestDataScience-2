import torch
import numpy as np
import matplotlib.pyplot as plt
from src.adc_testdatascience_2.utils.data_utils import get_dataloaders
from src.adc_testdatascience_2.models.rnn import SimpleRNN
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os


# Config
raw_path = "src/adc_testdatascience_2/data/energydata_complete.csv"
csv_path = "src/adc_testdatascience_2/data/processed_energy.csv"
model_path = "src/adc_testdatascience_2/models/rnn_1step.pth"
plot_path = "src/adc_testdatascience_2/scripts/plots/rnn_1step_eval.png"
input_window = 100
output_window = 1
input_dim = 26
hidden_dim = 64
num_layers = 2
forecast_steps = 100  # predict 100 test targets

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
train_loader, val_loader, test_loader = get_dataloaders(
    csv_path=csv_path,
    input_window=input_window,
    output_window=output_window
)

# Load model
model = SimpleRNN(input_dim, hidden_dim, output_dim=1, num_layers=num_layers).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ➤ Collect test windows and targets directly
X_test_windows = []
y_true = []

for X_batch, y_batch in test_loader:
    for x, y in zip(X_batch, y_batch):
        X_test_windows.append(x)
        y_true.append(y[0].item())  # Appliances
        if len(X_test_windows) >= forecast_steps:
            break
    if len(X_test_windows) >= forecast_steps:
        break

# ➤ Forecast using RNN model
predictions = []
with torch.no_grad():
    for X_input in X_test_windows:
        X_tensor = X_input.unsqueeze(0).to(device)
        y_pred = model(X_tensor).cpu().numpy().flatten()[0]
        predictions.append(y_pred)

# ➤ Sanity check
y_true = np.array(y_true)
assert len(predictions) == len(y_true) == forecast_steps, "Mismatch in prediction/target length"

# Step 1: Load the original (unscaled) CSV
df = pd.read_csv(raw_path)
df['Appliances'] = df['Appliances'].rolling(6*6, min_periods=1).mean()

# Step 2: Fit a scaler to only the Appliances column
appliances_scaler = MinMaxScaler()
appliances_scaler.fit_transform(df[["Appliances"]])  # Assuming the scaling was done this way

# Step 3: Inverse transform
y_true_unscaled = appliances_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
y_pred_unscaled = appliances_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# ➤ Plot true vs predicted
plt.figure(figsize=(12, 5))
plt.plot(y_true_unscaled, label="True (Test)", linewidth=2, color='blue')
plt.plot(y_pred_unscaled, label="Forecast (1-step RNN)", linestyle='--', color='orange')
plt.title("1-step RNN Forecast vs True (First 100 Test Points)")
plt.xlabel("Time step")
plt.ylabel("Appliances")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ➤ Save plot
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plt.savefig(plot_path)
plt.show()

print("✅ Correct 1-step RNN forecast complete and plot saved.")
