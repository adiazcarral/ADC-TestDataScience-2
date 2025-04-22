import torch
import torch.nn as nn
import numpy as np

class SimpleRNN(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=64, output_dim=100, num_layers=2, dropout=0.3):
        super(SimpleRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)  # output 100 time steps

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)               # out: (batch, seq_len, hidden)
        out = self.fc(out[:, -1, :])           # Use last hidden state for forecast
        return out                             # output: (batch, 100)

    def loss(self, y_true, y_pred):
        return nn.MSELoss()(y_pred, y_true)


