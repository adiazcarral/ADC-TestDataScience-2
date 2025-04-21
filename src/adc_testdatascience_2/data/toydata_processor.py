import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

class ToyDatasetProcessor:
    def __init__(self, dataset_path, save_path="src/adc_testdatascience_2/data/processed_toydata.csv"):
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.df = None
        self.cleaned_df = None
        self.time_index = None  # generated hourly vector
        self.sequence_length = 1000
        self.forecast_horizon = 100

    def load_dataset(self):
        print(f"📂 Loading dataset from: {self.dataset_path}")
        self.df = pd.read_csv(self.dataset_path)
        print(f"✅ Loaded dataset with shape: {self.df.shape}")

    def clean_data(self):
        print("🧹 Cleaning data: removing NaNs in 'obs_teta'...")
        df = self.df.copy()
        df = df.dropna(subset=['obs_teta'])  # Remove rows where target is NaN
        self.cleaned_df = df.reset_index(drop=True)
        print(f"✅ Cleaned dataset shape: {self.cleaned_df.shape}")

    def generate_time_index(self):
        print("🕒 Generating artificial hourly time index...")
        n = len(self.cleaned_df)
        self.time_index = pd.date_range(start='2022-01-01 00:00:00', periods=n, freq='H')
        print(f"✅ Time index generated with {n} hourly entries.")

    def save_clean_data(self):
        print(f"💾 Saving cleaned dataset to: {self.save_path}")
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.cleaned_df.to_csv(self.save_path, index=False)
        print("✅ Data saved.")

    def run_all(self):
        self.load_dataset()
        self.clean_data()
        self.generate_time_index()
        self.save_clean_data()


if __name__ == "__main__":
    processor = ToyDatasetProcessor(
        dataset_path="src/adc_testdatascience_2/data/toydata.csv"
    )
    processor.run_all()