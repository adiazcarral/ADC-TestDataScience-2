import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.adc_testdatascience_2.models.lstm_vae import LSTMVAE
from src.adc_testdatascience_2.utils.data_utils import get_dataloaders
import joblib


def inverse_transform_column(scaler, data, target_index):
    dummy = np.zeros((len(data), scaler.data_min_.shape[0]))
    dummy[:, target_index] = data[:, 0]
    return scaler.inverse_transform(dummy)[:, target_index]


def forecast_with_uncertainty(model, test_loader, device, forecast_steps, num_samples=100):
    model.eval()
    predictions, lower_bounds, upper_bounds, y_true = [], [], [], []

    X_test_windows = []
    for X_batch, y_batch in test_loader:
        for x, y in zip(X_batch, y_batch):
            X_test_windows.append(x)
            y_true.append(y[0].item())
            if len(X_test_windows) >= forecast_steps:
                break
        if len(X_test_windows) >= forecast_steps:
            break

    with torch.no_grad():
        for X_input in X_test_windows:
            X_tensor = X_input.unsqueeze(0).to(device)
            samples = model.sample(X_tensor, num_samples=num_samples).squeeze(-1).squeeze(0)

            predictions.append(samples.mean(dim=0).cpu().numpy()[0])
            lower_bounds.append(torch.quantile(samples, 0.05, dim=0).cpu().numpy()[0])
            upper_bounds.append(torch.quantile(samples, 0.95, dim=0).cpu().numpy()[0])

    return (
        np.array(predictions).reshape(-1, 1),
        np.array(lower_bounds).reshape(-1, 1),
        np.array(upper_bounds).reshape(-1, 1),
        np.array(y_true).reshape(-1, 1)
    )


def plot_forecast(y_true_inv, y_pred_inv, y_lower_inv, y_upper_inv, target_column, output_path):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true_inv, label="True (Test)", linewidth=2, color='blue')
    plt.plot(y_pred_inv, label="Forecast (LSTM-VAE)", linestyle='--', color='orange')
    plt.fill_between(np.arange(len(y_true_inv)), y_lower_inv, y_upper_inv,
                     alpha=0.3, label="90% Confidence Interval", color='orange')

    plt.title("LSTM-VAE Forecast with Uncertainty (First 100 Test Points)")
    plt.xlabel("Time step")
    plt.ylabel(target_column)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()


def test_lstm_vae_direct_forecast(model, test_loader, device, scaler, forecast_steps=100, target_column="Appliances"):
    target_index = list(scaler.feature_names_in_).index(target_column)

    y_pred, y_lower, y_upper, y_true = forecast_with_uncertainty(
        model, test_loader, device, forecast_steps
    )

    y_true_inv = inverse_transform_column(scaler, y_true, target_index)
    y_pred_inv = inverse_transform_column(scaler, y_pred, target_index)
    y_lower_inv = inverse_transform_column(scaler, y_lower, target_index)
    y_upper_inv = inverse_transform_column(scaler, y_upper, target_index)

    plot_path = os.path.join("src", "adc_testdatascience_2", "scripts", "plots", "lstmvae_direct_forecast_uncertainty.png")
    plot_forecast(y_true_inv, y_pred_inv, y_lower_inv, y_upper_inv, target_column, plot_path)

    print("✅ LSTM-VAE forecast with uncertainty complete and plot saved.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMVAE(
        num_input=26, num_hidden=128, num_layers=2,
        dropout=0.3, output_window=1, output_dim=1
    )
    model_path = os.path.join("src", "adc_testdatascience_2", "models", "lstmvae_1step.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    data_path = os.path.join("src", "adc_testdatascience_2", "data", "processed_energy.csv")
    _, _, test_loader = get_dataloaders(
        csv_path=data_path,
        input_window=100,
        output_window=1
    )

    scaler_path = os.path.join("src", "adc_testdatascience_2", "data", "processed_energy_scaler.pkl")
    scaler = joblib.load(scaler_path)

    test_lstm_vae_direct_forecast(
        model=model,
        test_loader=test_loader,
        device=device,
        scaler=scaler,
        forecast_steps=100,
        target_column="Appliances"
    )
