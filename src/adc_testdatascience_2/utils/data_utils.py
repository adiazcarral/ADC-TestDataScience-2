# data_utils.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def create_sequences(data, input_window=500, output_window=100):
    X, y = [], []
    for i in range(len(data) - input_window - output_window):
        X.append(data[i:i + input_window])
        y.append(data[i + input_window:i + input_window + output_window, 0])  # target: Appliances
    return np.array(X), np.array(y)


def get_dataloaders(csv_path, input_window=500, output_window=100, batch_size=64, test_size=0.2, val_size=0.1):
    print(f"📂 Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    data = df.values.astype(np.float32)

    print("📐 Creating sequences for forecasting...")
    X, y = create_sequences(data, input_window=input_window, output_window=output_window)

    print("🔀 Splitting into train/val/test...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, shuffle=False)
    relative_val_size = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=relative_val_size, shuffle=False)

    print("📦 Creating TensorDatasets and DataLoaders...")
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=batch_size)

    print("✅ DataLoaders ready!")
    return train_loader, val_loader, test_loader

