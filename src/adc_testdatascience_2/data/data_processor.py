import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import pickle
import matplotlib.pyplot as plt
import numpy as np


class ApplianceEnergyProcessor:
    def __init__(self, dataset_path=None, save_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Define default paths if not provided
        self.dataset_path = dataset_path or os.path.join(base_dir, "..", "data", "energydata_complete.csv")
        self.save_path = save_path or os.path.join(base_dir, "..", "data", "processed_energy.csv")

        self.df = None
        self.cleaned_df = None
        self.sequence_length = 100
        self.forecast_horizon = 1

    def load_dataset(self):
        print(f"📂 Loading dataset from: {self.dataset_path}")
        self.df = pd.read_csv(self.dataset_path, parse_dates=['date'], index_col='date')
        print(f"✅ Loaded dataset with shape: {self.df.shape}")

    def clean_data(self):
        print("🧹 Cleaning data: removing NaNs and zero values...")
        df = self.df.copy()
        df = df.dropna()
        df = df[df['Appliances'] > 0]
        df = df.drop(columns=['rv1', 'rv2'], errors='ignore')
        self.cleaned_df = df
        print(f"✅ Cleaned dataset shape: {self.cleaned_df.shape}")

    def analyze_seasonality(self):
        print("📈 Analyzing seasonality...")
        result = seasonal_decompose(self.cleaned_df['Appliances'], model='additive', period=1440)
        result.plot()
        plt.show()

        adf_test = adfuller(self.cleaned_df['Appliances'])
        adf_statistic, adf_p_value = adf_test[0], adf_test[1]

        print(f"ADF Statistic: {adf_statistic}")
        print(f"p-value for ADF Test: {adf_p_value}")
        print("❗ The series is " + ("stationary" if adf_p_value < 0.05 else "non-stationary"))

        seasonality_strength = np.var(result.seasonal) / np.var(result.observed)
        if seasonality_strength > 0.8:
            print("🌞 Seasonality strength: Very high")
        elif seasonality_strength > 0.5:
            print("🌤️ Seasonality strength: Moderate")
        elif seasonality_strength > 0.2:
            print("🌥️ Seasonality strength: Mild")
        else:
            print("🌑 Seasonality strength: Low or negligible")

        plot_acf(self.cleaned_df['Appliances'], lags=50)
        plt.show()

    def normalize_and_export(self):
        print("📊 Normalizing and exporting data...")
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(self.cleaned_df)
        self.cleaned_df = pd.DataFrame(df_scaled, columns=self.cleaned_df.columns, index=self.cleaned_df.index)

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.cleaned_df.to_csv(self.save_path)

        scaler_path = self.save_path.replace(".csv", "_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        print(f"✅ Data and scaler saved at:\n- {self.save_path}\n- {scaler_path}")

    def run_pipeline(self):
        self.load_dataset()
        self.clean_data()
        self.analyze_seasonality()
        self.normalize_and_export()


if __name__ == "__main__":
    processor = ApplianceEnergyProcessor()
    processor.run_pipeline()
