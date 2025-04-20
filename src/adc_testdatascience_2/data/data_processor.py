import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose


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
        df = df[df['Appliances'] > 0]  # Remove rows where target is zero
        self.cleaned_df = df[['Appliances']]  # Keep only the target column for forecasting
        print(f"✅ Cleaned dataset shape: {self.cleaned_df.shape}")

    def analyze_seasonality(self):
        print("📈 Analyzing seasonality...")
        result = seasonal_decompose(self.cleaned_df['Appliances'], model='additive', period=1440)  # 1440 = daily if minute-level
        result.plot()

    def normalize_and_export(self):
        print("📊 Normalizing and exporting data...")
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(self.cleaned_df)
        self.cleaned_df = pd.DataFrame(df_scaled, columns=["Appliances"], index=self.cleaned_df.index)

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.cleaned_df.to_csv(self.save_path)
        print(f"✅ Processed data saved to {self.save_path}")

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