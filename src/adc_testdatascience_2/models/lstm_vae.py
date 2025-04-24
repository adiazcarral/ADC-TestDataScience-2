import torch
import torch.nn as nn


class LSTMVAE(nn.Module):
    def __init__(self, num_input, num_hidden, num_layers, dropout, output_window, output_dim):
        super(LSTMVAE, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = num_hidden
        self.output_window = output_window
        self.output_dim = output_dim

        # Encoder
        self.encoder = nn.LSTM(num_input, num_hidden, num_layers, batch_first=True, dropout=dropout)

        # Latent space
        self.scale = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.Tanh())
        self.mu = nn.Linear(num_hidden, num_hidden)
        self.log_var = nn.Linear(num_hidden, num_hidden)

        # Decoder: Linear layer maps from latent to full output window of target variable
        self.decoder = nn.Linear(num_hidden, output_window)

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        _, (h_n, _) = self.encoder(x)  # shape: (num_layers, batch, hidden_size)
        return self.scale(h_n[-1])     # shape: (batch, hidden_size)

    def decode(self, encoded, z):
        out = encoded * (1 + z)  # shape: (batch, hidden_size)
        # out = encoded #* (1 + z)  # shape: (batch, hidden_size)
        decoded = self.decoder(out)  # shape: (batch, output_window * output_dim)
        return decoded.view(-1, self.output_window)#, self.output_dim)  # reshape to (batch, 100, 1)

    def forward(self, x):
        encoded = self.encode(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparametrize(mu, log_var)
        decoded = self.decode(encoded, z)
        return decoded, mu, log_var

    def sample(self, x, num_samples):
        encoded = self.encode(x).unsqueeze(1).repeat(1, num_samples, 1)
        z = torch.randn((x.size(0), num_samples, self.hidden_size)).to(x.device)
        decoded = self.decoder(encoded * (1 + z))
        return decoded.view(x.size(0), num_samples, self.output_window, self.output_dim)


def loss(x_hat, x_target, mu, log_var):
    recon_loss = nn.functional.mse_loss(x_hat.squeeze(-1), x_target.squeeze(-1), reduction='mean')
    weight = (x_target.squeeze(-1) != 0).float() + 1
    weighted_recon_loss = torch.mean(weight * recon_loss)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x_hat.size(0)
    return weighted_recon_loss, kl_loss


