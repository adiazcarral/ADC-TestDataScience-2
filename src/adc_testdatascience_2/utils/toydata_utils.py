import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def create_sequences(dataframe, input_window=500, output_window=100, step=6):
    obs_teta_idx = dataframe.columns.get_loc("obs_q")
    
    X_data = dataframe.values
    y_data = dataframe["obs_q"].values

    X, y = [], []
    for i in range(0, len(dataframe) - input_window - output_window, step):
        X.append(X_data[i:i + input_window])  # [window, num_features]
        y.append(y_data[i + input_window:i + input_window + output_window])  # [output_window]

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)[..., np.newaxis]  # Add feature dim for y: (samples, window, 1)

    print(f"✅ X shape: {X.shape} (samples, input_window, input_features)")
    print(f"✅ y shape: {y.shape} (samples, output_window, 1)")

    return X, y


def get_dataloaders(csv_path, input_window=500, output_window=100):
    df = pd.read_csv(csv_path)

    # Ensure all values are float32
    df = df.astype(np.float32)

    # Create sequences
    X, y = create_sequences(df, input_window=input_window, output_window=output_window)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Split into train/val/test
    total_samples = len(X_tensor)
    train_end = int(0.7 * total_samples)
    val_end = int(0.85 * total_samples)

    train_dataset = TensorDataset(X_tensor[:train_end], y_tensor[:train_end])
    val_dataset = TensorDataset(X_tensor[train_end:val_end], y_tensor[train_end:val_end])
    test_dataset = TensorDataset(X_tensor[val_end:], y_tensor[val_end:])

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
