import os
import torch
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from src.adc_testdatascience_1.models.logistic import LogisticRegression

app = FastAPI()

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model from pickle file
model_path = os.path.join("src", "adc_testdatascience_1", "models", "logistic_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

model.eval()  # Set model to evaluation mode
model.to(device)

# Pydantic model for input validation
class PredictionInput(BaseModel):
    inputs: List[List[float]]  # Expecting 2D list of floats

# Prediction endpoint
@app.post("/predict")
async def predict(data: PredictionInput):
    try:
        inputs = torch.tensor(data.inputs, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get predicted class
        return {"prediction": predicted.tolist()}
    except Exception as e:
        return {"error": str(e)}
