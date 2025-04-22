import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from src.adc_testdatascience_2.models.lstm_encoder_decoder import LSTMEncoderDecoder
from src.adc_testdatascience_2.utils.data_utils import get_dataloaders

# Paths
model_path = "src/adc_testdatascience_2/models/lstm_encoder_decoder.pth"
scaler_path = "src/adc_testdatascience_2/data/processed_energy_scaler.pkl"
csv_path = "src/adc_testdatascience_2/data/processed_energy.csv"
plot_path = "src/adc_testdatascience_2/scripts/plots/lstm_encoder_decoder_forecast.png"

# Constants
input_window = 1000
forecast_horizon = 100
input_dim = 26
output_dim = 1
hidden_dim = 64
num_layers = 2

def test():
    _, _, test_loader = get_dataloaders(csv_path, input_window, forecast_horizon)

    X_test, y_test = next(iter(test_loader))
    X_test = X_test[0].unsqueeze(0)  # (1, input_window, 26)
    y_test = y_test[0].numpy()       # (100,)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMEncoderDecoder(input_dim, hidden_dim, output_dim, forecast_horizon, num_layers).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    with torch.no_grad():
        X_test = X_test.to(device)
        prediction = model(X_test).cpu().numpy().flatten()

    # # Inverse transform
    # y_test_zeros = np.zeros((y_test.shape[0], 26))
    # pred_zeros = np.zeros((prediction.shape[0], 26))
    # y_test_zeros[:, -1] = y_test
    # pred_zeros[:, -1] = prediction
    # y_test_inv = scaler.inverse_transform(y_test_zeros)[:, -1]
    # pred_inv = scaler.inverse_transform(pred_zeros)[:, -1]

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(y_test, label="True")
    plt.plot(prediction, label="Forecast", linestyle='--')
    plt.title("LSTM Encoder-Decoder Forecast (100 steps)")
    plt.xlabel("Future time steps")
    plt.ylabel("Appliances")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()

    print("✅ LSTM Encoder-Decoder forecast complete.")

if __name__ == "__main__":
    test()
