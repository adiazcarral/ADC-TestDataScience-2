# data_utils.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def create_sequences(dataframe, input_window=500, output_window=100, step=12):
    appliances_idx = dataframe.columns.get_loc("Appliances")
    
    # Drop Appliances column for X, keep it for y
    # X_data = dataframe.drop(columns=["Appliances"]).values
    X_data = dataframe
    y_data = dataframe["Appliances"].values

    X, y = [], []
    for i in range(0, len(dataframe) - input_window - output_window, step):  # Increment by 'step' (6)
        X.append(X_data[i:i + input_window])  # [window, num_features-1]
        y.append(y_data[i + input_window:i + input_window + output_window])  # [output_window]

    X = np.array(X)
    y = np.array(y)

    print(f"✅ X shape: {X.shape} (samples, input_window, input_features)")
    print(f"✅ y shape: {y.shape} (samples, output_window)")

    return X, y


def safe_rolling_sum(df, column="Appliances", window=7*24*6):  # 1008
    values = df[column].values.astype(np.float64)
    cum = np.cumsum(np.insert(values, 0, 0))  # Pad with zero for correct diff
    result = cum[window:] - cum[:-window]
    padded_result = np.concatenate([np.full(window-1, result[0]), result])
    df['Appliances_cumulative'] = padded_result
    return df


def get_dataloaders(csv_path, input_window=500, output_window=100):
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # OPTIONAL: You can print this to confirm column names
    # print(df.columns)
    batch_size = 64
    # Ensure all values are float32 except the index
    df = df.astype(np.float32)

    df['Appliances'] = df['Appliances'].rolling(6*6, min_periods=1).mean()
    df['Appliances'] = np.log1p(df['Appliances'])


    # Create sequences using the DataFrame (so we can access column names)
    X, y = create_sequences(df, input_window=input_window, output_window=output_window)

    # Convert to PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # Split (e.g., 70% train, 15% val, 15% test)
    total_samples = len(X)
    train_end = int(0.7 * total_samples)
    val_end = int(0.85 * total_samples)

    train_dataset = torch.utils.data.TensorDataset(X[:train_end], y[:train_end])
    val_dataset = torch.utils.data.TensorDataset(X[train_end:val_end], y[train_end:val_end])
    test_dataset = torch.utils.data.TensorDataset(X[val_end:], y[val_end:])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
