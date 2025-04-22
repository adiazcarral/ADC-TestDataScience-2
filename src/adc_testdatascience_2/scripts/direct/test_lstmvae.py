import torch
import numpy as np
import matplotlib.pyplot as plt
from src.adc_testdatascience_2.models.lstm_vae import LSTMVAE
from src.adc_testdatascience_2.utils.data_utils import get_dataloaders
import os 
import pandas as pd

def test_lstm_vae_with_uncertainty(model, test_loader, device, df_raw, input_window, n_samples=100,
                                   target_mean=0.0, target_std=1.0, target_column="Appliances", n_batches=1):
    model.eval()
    all_y_true, all_y_hat, all_y_lower, all_y_upper = [], [], [], []

    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(test_loader):
            if i >= n_batches:
                break

            x_batch = x_batch.to(device)
            batch_size, pred_len = y_batch.shape[0], y_batch.shape[1]

            # Sample outputs from the VAE model
            samples = model.sample(x_batch, n_samples)  # shape: [batch, n_samples, output_window, output_dim]
            samples = samples.squeeze(-1)               # shape: [batch, n_samples, output_window]
            samples = samples.permute(0, 2, 1)          # shape: [batch, output_window, n_samples]

            # Compute statistics
            mean_pred = samples.mean(dim=-1)            # [batch, output_window]
            lower_5 = torch.quantile(samples, 0.05, dim=-1)
            upper_95 = torch.quantile(samples, 0.95, dim=-1)

            y_true = y_batch.squeeze(-1).cpu()          # [batch, output_window]

            # Store only the first sample for visualization
            all_y_true.append(y_true[0])
            all_y_hat.append(mean_pred[0])
            all_y_lower.append(lower_5[0])
            all_y_upper.append(upper_95[0])

            # ==== Plot 1: Standard prediction vs true + CI ====
            time_axis = np.arange(pred_len)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(time_axis, y_true[0], label="Observed", color="black")
            ax.plot(time_axis, mean_pred[0], label="Predicted Mean", color="tab:blue")
            ax.fill_between(time_axis, lower_5[0], upper_95[0], alpha=0.3, label="90% CI", color="tab:blue")
            ax.set_xlabel("Time Step [hours]")
            ax.set_ylabel(target_column)
            ax.set_title(f"Probabilistic Forecast with 90% CI - {target_column}")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.show()

            # ==== Plot 2: Context + forecast ====
            # Start index of the forecast in the raw data
            start_idx = input_window + i  # assumes test_loader is not shuffled
            df_raw[target_column] = df_raw[target_column].rolling(6*6, min_periods=1).mean()
            df_raw[target_column] = np.log1p(df_raw[target_column])
            context = df_raw[target_column].values[start_idx - input_window:start_idx]
            forecast_range = np.arange(input_window, input_window + pred_len)
            full_time = np.arange(input_window + pred_len)

            full_truth = np.concatenate([context, y_true[0]])
            full_pred = np.concatenate([context, mean_pred[0]])

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(full_time, full_truth, label="Ground Truth", color="black")
            ax.plot(full_time, full_pred, label="Predicted Mean", color="tab:blue")
            ax.fill_between(forecast_range, lower_5[0], upper_95[0], alpha=0.3, label="90% CI", color="tab:blue")
            ax.axvline(input_window, color="red", linestyle="--", label="Forecast Start")
            ax.set_xlabel("Time Step [hours]")
            ax.set_ylabel(target_column)
            ax.set_title(f"Forecast with Context Window - {target_column}")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            plt.savefig("src/adc_testdatascience_2/scripts/plots/lstmvae_direct_forecast.png")
            plt.show()


if __name__ == "__main__":
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMVAE(num_input=26, num_hidden=64, num_layers=2, dropout=0.3, output_window=100, output_dim=1)
    model.load_state_dict(torch.load("src/adc_testdatascience_2/models/lstmvae_direct.pth", map_location=device))
    model.to(device)

    # Load test data
    _, _, test_loader = get_dataloaders(
        csv_path="src/adc_testdatascience_2/data/processed_energy.csv",
        input_window=1000, output_window=100
    )

    # Replace with your actual target stats if applicable
    target_mean = 0.0
    target_std = 1.0

    # Run test
    df_raw = pd.read_csv("src/adc_testdatascience_2/data/processed_energy.csv")

    test_lstm_vae_with_uncertainty(
        model=model,
        test_loader=test_loader,
        device=device,
        df_raw=df_raw,
        input_window=1000,
        target_mean=0.0,
        target_std=1.0,
        target_column="Appliances",
        n_batches=1
    )