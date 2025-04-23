# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from src.adc_testdatascience_2.models.lstm_vae import LSTMVAE
# from src.adc_testdatascience_2.utils.data_utils import get_dataloaders
# import os 
# import pandas as pd


# def test_lstm_vae_direct_forecast(model, test_loader, device, forecast_steps=100,
#                                    target_mean=0.0, target_std=1.0, target_column="Appliances"):
#     model.eval()
#     predictions = []
#     y_true = []

#     # Collect test windows and targets directly
#     X_test_windows = []
#     for X_batch, y_batch in test_loader:
#         for x, y in zip(X_batch, y_batch):
#             X_test_windows.append(x)
#             y_true.append(y[0].item())
#             if len(X_test_windows) >= forecast_steps:
#                 break
#         if len(X_test_windows) >= forecast_steps:
#             break

#     # Forecast with uncertainty
#     with torch.no_grad():
#         for X_input in X_test_windows:
#             X_tensor = X_input.unsqueeze(0).to(device)
#             samples = model.sample(X_tensor, num_samples=100).squeeze(-1).squeeze(0)  # [samples, output_window]

#             mean_pred = samples.mean(dim=0).cpu().numpy()[0]  # scalar
#             predictions.append(mean_pred)

#     # Inverse transform if needed
#     y_true = np.array(y_true) * target_std + target_mean
#     predictions = np.array(predictions) * target_std + target_mean

#     # Plot
#     plt.figure(figsize=(12, 5))
#     plt.plot(y_true, label="True (Test)", linewidth=2, color='blue')
#     plt.plot(predictions, label="Forecast (LSTM-VAE)", linestyle='--', color='orange')
#     plt.title("LSTM-VAE Forecast vs True (First 100 Test Points)")
#     plt.xlabel("Time step")
#     plt.ylabel(target_column)
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()

#     plot_path = "src/adc_testdatascience_2/scripts/plots/lstmvae_direct_forecast_compact.png"
#     os.makedirs(os.path.dirname(plot_path), exist_ok=True)
#     plt.savefig(plot_path)
#     plt.show()

#     print("\u2705 LSTM-VAE direct forecast complete and plot saved.")


# if __name__ == "__main__":
#     # Load model
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = LSTMVAE(num_input=26, num_hidden=64, num_layers=2, dropout=0.3, output_window=1, output_dim=1)
#     model.load_state_dict(torch.load("src/adc_testdatascience_2/models/lstmvae_1step.pth", map_location=device))
#     model.to(device)

#     # Load test data
#     _, _, test_loader = get_dataloaders(
#         csv_path="src/adc_testdatascience_2/data/processed_energy.csv",
#         input_window=100, output_window=1
#     )

#     # Run test
#     test_lstm_vae_direct_forecast(
#         model=model,
#         test_loader=test_loader,
#         device=device,
#         forecast_steps=100,
#         target_mean=0.0,
#         target_std=1.0,
#         target_column="Appliances"
#     )

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.adc_testdatascience_2.models.lstm_vae import LSTMVAE
from src.adc_testdatascience_2.utils.data_utils import get_dataloaders
import os
import joblib


def test_lstm_vae_direct_forecast(model, test_loader, device, scaler, forecast_steps=100,
                                   target_column="Appliances"):
    model.eval()
    predictions = []
    lower_bounds = []
    upper_bounds = []
    y_true = []

    # Get the index of the target column from the scaler
    target_index = list(scaler.feature_names_in_).index(target_column)

    # Collect test windows and targets
    X_test_windows = []
    for X_batch, y_batch in test_loader:
        for x, y in zip(X_batch, y_batch):
            X_test_windows.append(x)
            y_true.append(y[0].item())
            if len(X_test_windows) >= forecast_steps:
                break
        if len(X_test_windows) >= forecast_steps:
            break

    # Forecast with uncertainty
    with torch.no_grad():
        for X_input in X_test_windows:
            X_tensor = X_input.unsqueeze(0).to(device)
            samples = model.sample(X_tensor, num_samples=100).squeeze(-1).squeeze(0)  # [samples]

            mean_pred = samples.mean(dim=0).cpu().numpy()[0]
            lower = torch.quantile(samples, 0.05, dim=0).cpu().numpy()[0]
            upper = torch.quantile(samples, 0.95, dim=0).cpu().numpy()[0]

            predictions.append(mean_pred)
            lower_bounds.append(lower)
            upper_bounds.append(upper)

    # Inverse transform using scaler
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(predictions).reshape(-1, 1)
    y_lower = np.array(lower_bounds).reshape(-1, 1)
    y_upper = np.array(upper_bounds).reshape(-1, 1)

    dummy = np.zeros((forecast_steps, scaler.data_min_.shape[0]))  # MinMaxScaler-compatible
    dummy[:, target_index] = y_true[:, 0]
    y_true_inv = scaler.inverse_transform(dummy)[:, target_index]

    dummy[:, target_index] = y_pred[:, 0]
    y_pred_inv = scaler.inverse_transform(dummy)[:, target_index]

    dummy[:, target_index] = y_lower[:, 0]
    y_lower_inv = scaler.inverse_transform(dummy)[:, target_index]

    dummy[:, target_index] = y_upper[:, 0]
    y_upper_inv = scaler.inverse_transform(dummy)[:, target_index]

    # Plot forecast with uncertainty
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_inv, label="True (Test)", linewidth=2, color='blue')
    plt.plot(y_pred_inv, label="Forecast (LSTM-VAE)", linestyle='--', color='orange')
    plt.fill_between(np.arange(forecast_steps), y_lower_inv, y_upper_inv,
                     alpha=0.3, label="90% Confidence Interval", color='orange')
    plt.title("LSTM-VAE Forecast with Uncertainty (First 100 Test Points)")
    plt.xlabel("Time step")
    plt.ylabel(target_column)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = "src/adc_testdatascience_2/scripts/plots/lstmvae_direct_forecast_uncertainty.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    plt.show()

    print("✅ LSTM-VAE forecast with uncertainty complete and plot saved.")


if __name__ == "__main__":
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMVAE(num_input=26, num_hidden=64, num_layers=2, dropout=0.3, output_window=1, output_dim=1)
    model.load_state_dict(torch.load("src/adc_testdatascience_2/models/lstmvae_1step.pth", map_location=device))
    model.to(device)

    # Load test data
    _, _, test_loader = get_dataloaders(
        csv_path="src/adc_testdatascience_2/data/processed_energy.csv",
        input_window=100, output_window=1
    )

    # Load saved scaler
    scaler = joblib.load("src/adc_testdatascience_2/data/processed_energy_scaler.pkl")

    # Run test
    test_lstm_vae_direct_forecast(
        model=model,
        test_loader=test_loader,
        device=device,
        scaler=scaler,
        forecast_steps=100,
        target_column="Appliances"
    )
