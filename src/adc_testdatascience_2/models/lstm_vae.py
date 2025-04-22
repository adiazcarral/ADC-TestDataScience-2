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
        self.decoder = nn.Linear(num_hidden, output_window * output_dim)

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
        return decoded.view(-1, self.output_window, self.output_dim)  # reshape to (batch, 100, 1)

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


# class LSTMVAE(nn.Module):
#     def __init__(self, 
#                 num_input=26,      # Number of input features
#                 num_hidden=64,     # Hidden dimension in LSTM
#                 output_dim=1,      # Output feature size (e.g., 1 target variable)
#                 output_window=100, # Number of forecast steps
#                 num_layers=2,      # Number of LSTM layers
#                 dropout=0.3):      # Dropout rate
#         super(LSTMVAE, self).__init__()
#         self.hidden_dim = num_hidden
#         self.output_dim = output_dim
#         self.output_window = output_window
#         self.num_layers = num_layers

#         # Encoder
#         self.encoder = nn.LSTM(num_input, num_hidden, num_layers, batch_first=True, dropout=dropout)
#         self.fc_mu = nn.Linear(num_hidden, num_hidden)
#         self.fc_logvar = nn.Linear(num_hidden, num_hidden)

#         # Decoder
#         self.decoder = nn.LSTM(num_hidden + output_dim, num_hidden, num_layers, batch_first=True, dropout=dropout)
#         self.fc_out = nn.Linear(num_hidden, output_dim)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def encode(self, x):
#         _, (h_n, _) = self.encoder(x)
#         h_last = h_n[-1]  # (batch, hidden_dim)
#         mu = self.fc_mu(h_last)
#         logvar = self.fc_logvar(h_last)
#         z = self.reparameterize(mu, logvar)
#         return z, mu, logvar

#     def decode(self, z):
#         batch_size = z.size(0)

#         # Initial hidden and cell states from latent code
#         h_0 = z.unsqueeze(0).repeat(self.num_layers, 1, 1)  # [num_layers, batch, hidden_dim]
#         c_0 = torch.zeros_like(h_0)

#         # Start with zeros
#         decoder_input = torch.zeros(batch_size, 1, self.output_dim, device=z.device)

#         outputs = []
#         hidden = (h_0, c_0)

#         for _ in range(self.output_window):
#             # Inject latent code at each time step
#             lstm_input = torch.cat([decoder_input, z.unsqueeze(1)], dim=-1)  # [batch, 1, output_dim + hidden_dim]
#             out, hidden = self.decoder(lstm_input, hidden)
#             pred = self.fc_out(out)  # [batch, 1, output_dim]
#             outputs.append(pred)
#             decoder_input = pred  # Autoregressive feedback

#         return torch.cat(outputs, dim=1)  # [batch, output_window, output_dim]

#     def forward(self, x):
#         z, mu, logvar = self.encode(x)
#         recon = self.decode(z)
#         return recon, mu, logvar

#     def sample(self, x, n_samples):
#         z, _, _ = self.encode(x)  # [batch, hidden_dim]
#         z = z.unsqueeze(1).repeat(1, n_samples, 1)  # [batch, n_samples, hidden_dim]
#         z_samples = z + torch.randn_like(z)        # Add noise for stochasticity

#         # Flatten batch and sample dims for decoding
#         flat_z = z_samples.view(-1, self.hidden_dim)  # [batch * n_samples, hidden_dim]
#         output = self.decode(flat_z)  # [batch * n_samples, output_window, output_dim]

#         # Reshape back to [batch, n_samples, output_window, output_dim]
#         return output.view(x.size(0), n_samples, self.output_window, self.output_dim)

