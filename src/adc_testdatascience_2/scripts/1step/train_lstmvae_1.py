import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.adc_testdatascience_2.models.lstm_vae import LSTMVAE, loss
from src.adc_testdatascience_2.utils.data_utils import get_dataloaders


def train_direct_lstmvae(model, train_loader, val_loader, device, epochs=10):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        train_recon, train_kl = 0.0, 0.0

        for x_batch, y_batch in train_loader:  # ✅ use y_batch
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            x_hat, mu, log_var = model(x_batch)
            recon_loss, kl_loss = loss(x_hat, y_batch, mu, log_var)  # ✅ compare to target
            total_loss = recon_loss + 1e-2 * kl_loss
            total_loss.backward()
            optimizer.step()

            train_recon += recon_loss.item() * x_batch.size(0)
            train_kl += kl_loss.item() * x_batch.size(0)

        val_recon, val_kl, val_mae, val_rmse = evaluate(model, val_loader, device)

        print(f"\n📈 Epoch {epoch + 1}/{epochs}")
        print(f"    🏋️ Train Recon: {train_recon / len(train_loader.dataset):.4f} | KL: {train_kl / len(train_loader.dataset):.4f}")
        print(f"    🧪 Val Recon: {val_recon:.4f} | KL: {val_kl:.4f} | MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f}")

    torch.save(model.state_dict(), "src/adc_testdatascience_2/models/lstmvae_1step.pth")
    print("✅ Model saved")


def evaluate(model, loader, device):
    model.eval()
    recon_total, kl_total = 0.0, 0.0
    preds, targets = [], []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_hat, mu, log_var = model(x_batch)
            recon_loss, kl_loss = loss(x_hat, y_batch, mu, log_var)

            recon_total += recon_loss.item() * x_batch.size(0)
            kl_total += kl_loss.item() * x_batch.size(0)

            preds.append(x_hat.cpu())
            targets.append(y_batch.cpu())

    preds = torch.cat(preds, dim=0).squeeze(-1).numpy()    # [N, 100]
    targets = torch.cat(targets, dim=0).squeeze(-1).numpy()  # [N, 100]

    mae = mean_absolute_error(targets, preds)
    rmse = mean_squared_error(targets, preds, squared=False)

    avg_recon = recon_total / len(loader.dataset)
    avg_kl = kl_total / len(loader.dataset)
    return avg_recon, avg_kl, mae, rmse


if __name__ == "__main__":
    train_loader, val_loader, _ = get_dataloaders(
        csv_path="src/adc_testdatascience_2/data/processed_energy.csv",
        input_window=100, output_window=1
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMVAE(
        num_input=26,      # 26 input features
        num_hidden=64,
        num_layers=2,
        dropout=0.3,
        output_window=1,  # 100 time steps forecasted
        output_dim=1        # predicting 1 variable: Appliances
    )
    train_direct_lstmvae(model, train_loader, val_loader, device, epochs=10)