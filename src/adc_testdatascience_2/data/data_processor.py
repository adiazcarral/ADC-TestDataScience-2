import os
import pandas as pd
import numpy as np
from typing import Tuple
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import unittest
import seaborn as sns
from typing import Tuple


class ApplianceEnergyProcessor:
    """
    Class to process the Appliance Energy Prediction dataset for multivariate time series forecasting.
    It includes loading, seasonality checking, windowing, and train/val/test split.
    """

    def __init__(
        self,
        dataset_path: str = "./energydata_complete.csv",
        target_column: str = "Appliances",
        window_size: int = 500,
        horizon: int = 100,
        test_size: float = 0.15,
        val_size: float = 0.15,
    ):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.window_size = window_size
        self.horizon = horizon
        self.test_size = test_size
        self.val_size = val_size

    def load_data(self):
        print("📥 Loading Appliance Energy dataset...")
        df = pd.read_csv(self.dataset_path, parse_dates=["date"], index_col="date")
        df = df.dropna()
        self.features = df.columns.tolist()
        self.df = df
        print(f"✅ Data loaded with shape: {df.shape}")

    def check_seasonality(self):
        print("🔍 Checking for seasonality in the target variable...")
        result = seasonal_decompose(self.df[self.target_column], model="additive", period=1440)
        result.plot()
        plt.suptitle("Seasonality Decomposition (daily, 1440 steps)")
        plt.tight_layout()
        plt.show()
        print("⚠️ Seasonality visual inspection done. Decide based on the seasonal component above.")

    def scale_and_window(self):
        print("📐 Scaling data and generating windows...")
        scaler = MinMaxScaler()
        scaled_array = scaler.fit_transform(self.df)
        self.scaled_df = pd.DataFrame(scaled_array, columns=self.df.columns, index=self.df.index)

        X, y_direct, y_stepwise = [], [], []

        total_len = len(self.scaled_df)
        for i in range(self.window_size, total_len - self.horizon):
            window = self.scaled_df.iloc[i - self.window_size:i].values
            future_window = self.scaled_df.iloc[i:i + self.horizon][self.target_column].values

            X.append(window)
            y_direct.append(future_window)  # direct forecast
            y_stepwise.append(future_window[0])  # for t+1 autoregressive loop

        self.X = np.array(X)
        self.y_direct = np.array(y_direct)
        self.y_stepwise = np.array(y_stepwise)
        print(f"✅ Generated {len(self.X)} windows")

    def train_val_test_split(self):
        print("🔀 Splitting into train, validation and test sets...")
        X_train, X_temp, y_direct_train, y_direct_temp, y_stepwise_train, y_stepwise_temp = train_test_split(
            self.X, self.y_direct, self.y_stepwise, test_size=self.test_size + self.val_size, shuffle=False
        )

        relative_val_size = self.val_size / (self.test_size + self.val_size)
        X_val, X_test, y_direct_val, y_direct_test, y_stepwise_val, y_stepwise_test = train_test_split(
            X_temp, y_direct_temp, test_size=relative_val_size, shuffle=False
        )

        y_stepwise_val, y_stepwise_test = train_test_split(y_stepwise_temp, test_size=relative_val_size, shuffle=False)

        self.data = {
            "train": (X_train, y_direct_train, y_stepwise_train),
            "val": (X_val, y_direct_val, y_stepwise_val),
            "test": (X_test, y_direct_test, y_stepwise_test),
        }
        print("✅ Splits created:")
        for key in self.data:
            print(f"  {key}: {self.data[key][0].shape[0]} samples")

    def run_pipeline(self):
        self.load_data()
        self.check_seasonality()
        self.scale_and_window()
        self.train_val_test_split()


# Optional tests using unittest
class TestApplianceDataProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = ApplianceEnergyProcessor(dataset_path="./energydata_complete.csv")
        self.processor.load_dataset()

    def test_dataset_load(self):
        self.assertTrue(len(self.processor.df) > 0)

    def test_normalization_range(self):
        self.processor.normalize_data()
        self.assertTrue((self.processor.normalized_data.values >= 0).all())
        self.assertTrue((self.processor.normalized_data.values <= 1).all())


if __name__ == "__main__":
    # processor = ApplianceEnergyProcessor(data_path="./data/energydata_complete.csv")
    processor = ApplianceEnergyProcessor(dataset_path="./energydata_complete.csv")
    processor.run_pipeline()
    # To run tests: uncomment the line below
    # unittest.main()
