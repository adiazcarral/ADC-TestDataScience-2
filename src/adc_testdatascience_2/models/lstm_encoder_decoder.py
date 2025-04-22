import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell


class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, forecast_len, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.forecast_len = forecast_len

    def forward(self, hidden, cell):
        batch_size = hidden.size(1)
        decoder_input = torch.zeros(batch_size, 1, 1).to(hidden.device)
        outputs = []

        for _ in range(self.forecast_len):
            output, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))
            prediction = self.output_layer(output)
            outputs.append(prediction)
            decoder_input = prediction  # use predicted output as next input

        return torch.cat(outputs, dim=1)  # (batch_size, forecast_len, output_dim)


class LSTMEncoderDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, forecast_len, num_layers=1, dropout=0.2):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(output_dim, hidden_dim, forecast_len, num_layers, dropout)

    def forward(self, x):
        hidden, cell = self.encoder(x)
        output_seq = self.decoder(hidden, cell)
        return output_seq.squeeze(-1)  # shape: (batch_size, forecast_len)
