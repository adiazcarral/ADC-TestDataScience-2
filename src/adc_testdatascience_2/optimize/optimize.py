import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.adc_testdatascience_2.models.lstm_vae import LSTMVAE  # Assuming you have this model defined
from src.adc_testdatascience_2.utils.data_utils import get_dataloaders  # Ensure this is adapted for time series
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define hyperparameter grids per model
hyperparams = {
    "LSTMVAE": [
        {"lr": 1e-3, "batch_size": 64, "num_layers": 2, "num_hidden": 128},
        {"lr": 1e-4, "batch_size": 128, "num_layers": 3, "num_hidden": 256},
    ],
}

models = {
    "LSTMVAE": LSTMVAE,
}

def train_and_validate(model, train_loader, val_loader, lr, epochs=5):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Using MSELoss for regression task

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2, model


def main(model_name):
    assert model_name in models, f"Unsupported model: {model_name}"
    ModelClass = models[model_name]
    grid = hyperparams[model_name]

    best_rmse = float("inf")
    best_model = None
    best_config = None

    for config in grid:
        print(f"🔍 Testing config: {config}")
        train_loader, val_loader, _ = get_dataloaders(batch_size=config["batch_size"], subset_fraction=0.1)
        model = ModelClass(num_layers=config["num_layers"], num_hidden=config["num_hidden"])
        mae, rmse, r2, trained_model = train_and_validate(model, train_loader, val_loader, config["lr"])
        print(f"→ MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

        if rmse < best_rmse:  # You can also use MAE or another metric
            best_rmse = rmse
            best_model = trained_model
            best_config = config

    # Save best model
    torch.save(best_model.state_dict(), f"{model_name}_best.pth")
    print(f"✅ Best model saved: {model_name}_best.pth")
    print(f"🏆 Best config: {best_config} | RMSE: {best_rmse:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name: LSTMVAE")
    args = parser.parse_args()
    main(args.model)
