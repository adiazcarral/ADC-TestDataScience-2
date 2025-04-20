import torch
import torch.nn as nn


class LSTMVAE(nn.Module):
    def __init__(self, num_input, num_hidden, num_layers, dropout):
        super(LSTMVAE, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = num_hidden

        # Encoder: Stack multiple LSTM layers with dropout
        self.encoder = nn.LSTM(num_input, num_hidden, num_layers, batch_first=True, dropout=dropout)
        
        # Decoder: Single linear layer
        self.decoder = nn.Linear(num_hidden, num_input)
        
        # Additional layers and parameters for VAE
        self.scale = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.Tanh())
        self.mu = nn.Linear(num_hidden, num_hidden)
        self.log_var = nn.Linear(num_hidden, num_hidden)

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        _, (h_n, _) = self.encoder(x)
        return self.scale(h_n[-1])

    def decode(self, encoded, z):
        return self.decoder(encoded * (1 + z))

    def forward(self, x):
        encoded = self.encode(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparametrize(mu, log_var)
        decoded = self.decode(encoded, z)
        return decoded, mu, log_var

    def sample(self, x, num_samples):
        encoded = self.encode(x).unsqueeze(1).repeat(1, num_samples, 1)
        z = torch.randn((x.size(0), num_samples, self.hidden_size))
        return self.decode(encoded, z)

def loss(x_hat, x, mu, log_var): 
    recon_loss = nn.functional.mse_loss(x_hat.squeeze(), x.squeeze(), reduction='none') # No reduction 
    weight = (x.squeeze() != 0).float() + 1 # More weight on non-zero values 
    weighted_recon_loss = torch.mean(weight * recon_loss)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) 
    return weighted_recon_loss, kl_loss
