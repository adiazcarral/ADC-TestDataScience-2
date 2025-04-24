import os
import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.adc_testdatascience_2.models.lstm_vae import LSTMVAE, loss
from src.adc_testdatascience_2.utils.data_utils import get_dataloaders


def train_direct_lstmvae(model, train_loader, val_loader, device, epochs=10, save_path=None):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        train_recon, train_kl = 0.0, 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            x_hat, mu, log_var = model(x_batch)
            recon_loss, kl_loss = loss(x_hat, y_batch, mu, log_var)
            total_loss = recon_loss + 1e-2 * kl_loss
            total_loss.backward()
            optimizer.step()

            train_recon += recon_loss.item() * x_batch.size(0)
            train_kl += kl_loss.item() * x_batch.size(0)

        val_recon, val_kl, val_mae, val_rmse = evaluate(model, val_loader, device)

        print(f"\n📈 Epoch {epoch + 1}/{epochs}")
        print(f"    🏋️ Train Recon: {train_recon / len(train_loader.dataset):.4f} | KL: {train_kl / len(train_loader.dataset):.4f}")
        print(f"    🧪 Val Recon: {val_recon:.4f} | KL: {val_kl:.4f} | MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"✅ Model saved to {save_path}")


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

    preds = torch.cat(preds, dim=0).squeeze(-1).numpy()
    targets = torch.cat(targets, dim=0).squeeze(-1).numpy()

    mae = mean_absolute_error(targets, preds)
    rmse = mean_squared_error(targets, preds, squared=False)

    return recon_total / len(loader.dataset), kl_total / len(loader.dataset), mae, rmse


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))  # src/.../scripts/1step/
    csv_path = os.path.join(base_dir, "..", "..", "data", "processed_energy.csv")
    model_path = os.path.join(base_dir, "..", "..", "models", "lstmvae_1step.pth")

    print(f"📄 Using dataset: {csv_path}")
    print(f"💾 Model will be saved to: {model_path}")

    train_loader, val_loader, _ = get_dataloaders(
        csv_path=csv_path,
        input_window=100,
        output_window=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMVAE(
        num_input=26,
        num_hidden=128,
        num_layers=2,
        dropout=0.3,
        output_window=1,
        output_dim=1
    )

    train_direct_lstmvae(model, train_loader, val_loader, device, epochs=10, save_path=model_path)
