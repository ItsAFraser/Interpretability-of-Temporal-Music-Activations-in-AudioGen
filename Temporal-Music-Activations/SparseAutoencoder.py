import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """Simple fully connected sparse autoencoder."""

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.activation = nn.ReLU()

        # Xavier init for stable training starts.
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor):
        latent_pre = self.encoder(x)
        latent = self.activation(latent_pre)
        recon = self.decoder(latent)
        return recon, latent
