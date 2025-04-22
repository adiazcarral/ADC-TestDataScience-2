import torch
import torch.nn as nn
import torch.optim as optim
from src.adc_testdatascience_2.models.lstm_encoder_decoder import LSTMEncoderDecoder
from src.adc_testdatascience_2.utils.data_utils import get_dataloaders

# Paths and constants
csv_path = "src/adc_testdatascience_2/data/processed_energy.csv"
model_save_path = "src/adc_testdatascience_2/models/lstm_encoder_decoder.pth"
input_window = 1000
forecast_horizon = 100
input_dim = 26
output_dim = 1
hidden_dim = 64
num_layers = 2
num_epochs = 10
lr = 1e-3
batch_size = 32

def train():
    _, train_loader, val_loader = get_dataloaders(csv_path, input_window, forecast_horizon)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMEncoderDecoder(input_dim, hidden_dim, output_dim, forecast_horizon, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model saved to {model_save_path}")

if __name__ == "__main__":
    train()
