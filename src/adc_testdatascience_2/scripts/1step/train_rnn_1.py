import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.adc_testdatascience_2.models.rnn import SimpleRNN
from src.adc_testdatascience_2.utils.data_utils import get_dataloaders


def train_direct_rnn(model, train_loader, val_loader, device, model_save_path, epochs=10):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-03, weight_decay=1e-04)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)

        val_loss, val_mae, val_rmse = evaluate(model, val_loader, device)

        print(f"\n📈 Epoch {epoch+1}/{epochs}")
        print(f"    🏋️ Train Loss: {train_loss/len(train_loader.dataset):.4f}")
        print(f"    🧪 Val Loss: {val_loss:.4f} | MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f}")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model saved to {model_save_path}")


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    criterion = nn.MSELoss()
    loss_total = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss_total += loss.item() * x_batch.size(0)
            preds.append(pred.cpu().numpy())
            targets.append(y_batch.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mae = mean_absolute_error(targets, preds)
    rmse = mean_squared_error(targets, preds, squared=False)
    return loss_total / len(loader.dataset), mae, rmse


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))  # path to /scripts/1step/
    csv_path = os.path.join(base_dir, "..", "..", "data", "processed_energy.csv")  # go up two levels
    model_path = os.path.join(base_dir, "..", "..", "models", "rnn_1step.pth")

    print(f"📄 Looking for CSV at: {csv_path}")
    print(f"💾 Will save model to: {model_path}")

    train_loader, val_loader, _ = get_dataloaders(
        csv_path=csv_path,
        input_window=100,
        output_window=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleRNN(input_dim=26, hidden_dim=128, num_layers=2, output_dim=1)
    train_direct_rnn(model, train_loader, val_loader, device, model_save_path=model_path, epochs=10)
