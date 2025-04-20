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
    def __init__(self, dataset_path, save_path="src/adc_testdatascience_2/data/processed_energy.csv"):
        self.dataset_path = dataset_path
        self.save_path = save_path
        self.df = None
        self.cleaned_df = None
        self.sequence_length = 500
        self.forecast_horizon = 100

    def load_dataset(self):
        print(f"📂 Loading dataset from: {self.dataset_path}")
        self.df = pd.read_csv(self.dataset_path, parse_dates=['date'], index_col='date')
        print(f"✅ Loaded dataset with shape: {self.df.shape}")

    def clean_data(self):
        print("🧹 Cleaning data: removing NaNs and zero values...")
        df = self.df.copy()
        df = df.dropna()
        df = df[df['Appliances'] > 0]  # Filter based on target, but keep all columns
        # Drop 'rv1' and 'rv2' if they exist
        df = df.drop(columns=['rv1', 'rv2'], errors='ignore')
        self.cleaned_df = df  # Keep all features
        print(f"✅ Cleaned dataset shape: {self.cleaned_df.shape}")

    def analyze_seasonality(self):
        print("📈 Analyzing seasonality...")
        
        # Decompose the time series into trend, seasonal, and residual components
        result = seasonal_decompose(self.cleaned_df['Appliances'], model='additive', period=1440)  # 1440 = daily if minute-level
        
        # Plot the decomposed components
        result.plot()
        plt.show()
        
        # Perform Augmented Dickey-Fuller test for stationarity
        adf_test = adfuller(self.cleaned_df['Appliances'])
        adf_statistic = adf_test[0]
        adf_p_value = adf_test[1]
        
        print(f"ADF Statistic: {adf_statistic}")
        print(f"p-value for ADF Test: {adf_p_value}")
        
        # Evaluate seasonality based on p-value and trend
        if adf_p_value < 0.05:
            print("❗ The series is stationary (no strong trend).")
        else:
            print("❗ The series is non-stationary (has a trend).")

        seasonality_strength = np.var(result.seasonal) / np.var(result.observed)

        if seasonality_strength > 0.8:
            print("🌞 Seasonality strength: Very high")
        elif seasonality_strength > 0.5:
            print("🌤️ Seasonality strength: Moderate")
        elif seasonality_strength > 0.2:
            print("🌥️ Seasonality strength: Mild")
        else:
            print("🌑 Seasonality strength: Low or negligible")
        
        # Plot ACF (Autocorrelation Function) to understand temporal dependencies
        plot_acf(self.cleaned_df['Appliances'], lags=50)
        plt.show()


    def normalize_and_export(self):
            print("📊 Normalizing and exporting data...")

            # Assuming that self.cleaned_df contains all the features and 'Appliances' column
            scaler = MinMaxScaler()
            
            # Scale all columns in self.cleaned_df, not just the 'Appliances' column
            df_scaled = scaler.fit_transform(self.cleaned_df)
            
            # Update cleaned_df with the scaled data and the original column names
            self.cleaned_df = pd.DataFrame(df_scaled, columns=self.cleaned_df.columns, index=self.cleaned_df.index)

            # Ensure the save directory exists
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

            # Save the processed dataframe to the specified path
            self.cleaned_df.to_csv(self.save_path)

            # Save the scaler to a file
            scaler_path = self.save_path.replace(".csv", "_scaler.pkl")  # Save scaler with a similar name
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

    def run_pipeline(self):
        self.load_dataset()
        self.clean_data()
        self.analyze_seasonality()
        self.normalize_and_export()

if __name__ == "__main__":
    processor = ApplianceEnergyProcessor(
        dataset_path="src/adc_testdatascience_2/data/energydata_complete.csv"
    )
    processor.run_pipeline()