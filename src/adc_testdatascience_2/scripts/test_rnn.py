import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import pandas as pd

from src.adc_testdatascience_2.models.rnn import SimpleRNN
from src.adc_testdatascience_2.utils.data_utils import get_dataloaders


def run_rnn_forecast():
    # Paths
    model_path = "src/adc_testdatascience_2/models/rnn_direct.pth"
    scaler_path = "src/adc_testdatascience_2/data/processed_energy_scaler.pkl"
    plot_path = "src/adc_testdatascience_2/scripts/plots/rnn_direct_forecast.png"
    plot_path_clean = "src/adc_testdatascience_2/scripts/plots/rnn_direct_clean_forecast.png"
    csv_path = "src/adc_testdatascience_2/data/processed_energy.csv"

    # Constants
    input_window = 1000
    forecast_horizon = 100
    input_dim = 26
    hidden_dim = 128
    num_layers = 2

    # Load data
    _, _, test_loader = get_dataloaders(
        csv_path=csv_path,
        input_window=input_window,
        output_window=forecast_horizon
    )

    # Take the first batch and sample
    X_test, y_test = next(iter(test_loader))
    X_test = X_test[0].unsqueeze(0)  # shape: (1, 1000, 26)
    y_test = y_test[0].numpy()       # shape: (100,)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleRNN(input_dim=input_dim, hidden_dim=hidden_dim,
                      output_dim=forecast_horizon, num_layers=num_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load the scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Forecast
    with torch.no_grad():
        X_test = X_test.to(device)
        prediction = model(X_test).cpu().numpy().flatten()

    # Plot forecast result
    plt.figure(figsize=(12, 5))
    plt.plot(y_test, label="True", linewidth=2)
    plt.plot(prediction, label="Forecast", linestyle='--')
    plt.title("Direct RNN Forecast (100 steps)")
    plt.xlabel("Future time steps")
    plt.ylabel("Appliances")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.show()

    print("✅ Direct forecast complete.")

    # Load raw normalized data
    df = pd.read_csv(csv_path)
    df['Appliances'] = df['Appliances'].rolling(6 * 6, min_periods=1).mean()
    df['Appliances'] = np.log1p(df['Appliances'])
    appliances_norm = df['Appliances'].values

    # Build input for clean forecast
    input_seq = appliances_norm[:input_window]
    X_input = np.zeros((1, input_window, 26))
    X_input[0, :, -1] = input_seq  # Use only the 'Appliances' feature

    # Forecast again
    X_input_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
    with torch.no_grad():
        prediction = model(X_input_tensor).cpu().numpy().flatten()

    # Ground truth for comparison
    y_true_future = appliances_norm[input_window:input_window + forecast_horizon]
    x_history = np.arange(input_window)
    x_forecast = np.arange(input_window, input_window + forecast_horizon)

    # Plot clean forecast
    plt.figure(figsize=(12, 5))
    plt.plot(x_history, input_seq, label="History (Appliances)", linewidth=2, color="tab:blue")
    plt.plot(x_forecast, y_true_future, label="True Future", linewidth=2, color="green")
    plt.plot(x_forecast, prediction, label="Forecast", linestyle='--', color="orange")
    plt.axvline(x=input_window, color='gray', linestyle=':', label="Forecast Start")
    plt.title("Direct RNN Forecast with History (Normalized)")
    plt.xlabel("Time steps")
    plt.ylabel("Appliances (normalized)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path_clean), exist_ok=True)
    plt.savefig(plot_path_clean)
    plt.show()

    print("✅ Clean forecast plot based on true time index complete.")


if __name__ == "__main__":
    run_rnn_forecast()
