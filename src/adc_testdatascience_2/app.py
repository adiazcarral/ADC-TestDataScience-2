from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import joblib
import numpy as np

# Define input format
class InputData(BaseModel):
    window: list[list[float]]  # 2D list: [timesteps, features]

# Load model and scaler
model = torch.load("models/lstm_vae_best.pth", map_location=torch.device("cpu"))
model.eval()

scaler = joblib.load("processed_energy_scaler.pkl")

app = FastAPI(title="Appliances Energy Forecasting API")

@app.post("/predict")
def predict(data: InputData):
    try:
        x = np.array(data.window).astype(np.float32)
        x_scaled = scaler.transform(x)
        x_tensor = torch.tensor(x_scaled).unsqueeze(0)  # shape: [1, timesteps, features]

        with torch.no_grad():
            mean, _, _, _ = model(x_tensor)
            forecast = mean.squeeze(0).numpy()  # shape: [forecast_steps, 1]

        return {"forecast": forecast.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
