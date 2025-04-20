import torch
import torch.nn as nn
from statsmodels.tsa.arima.model import ARIMA


class ARIMAForecaster(nn.Module):
    def __init__(self, p=5, d=1, q=0):
        super(ARIMAForecaster, self).__init__()
        self.p = p
        self.d = d
        self.q = q
        self.model = None

    def fit(self, train_data):
        self.model = ARIMA(train_data, order=(self.p, self.d, self.q))
        self.model_fit = self.model.fit()

    def forecast(self, steps=100):
        if self.model_fit is None:
            raise ValueError("Model is not fitted yet!")
        forecast = self.model_fit.forecast(steps=steps)
        return torch.tensor(forecast, dtype=torch.float32)

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

    def predict(self, data):
        self.model_fit = self.model.fit()
        forecast = self.model_fit.forecast(steps=len(data))
        return torch.tensor(forecast, dtype=torch.float32)

