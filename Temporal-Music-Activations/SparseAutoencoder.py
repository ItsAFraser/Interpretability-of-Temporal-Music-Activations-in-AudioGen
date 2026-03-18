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

    def normalize_decoder(self) -> None:
        """Project decoder columns to unit L2 norm (in-place, no grad).

        Call this after every optimizer step to prevent the model from trivially
        reducing reconstruction loss by shrinking encoder weights and inflating
        decoder columns, which defeats sparsity regularisation.
        """
        with torch.no_grad():
            # decoder.weight shape: [input_dim, latent_dim]
            # Each column j corresponds to the j-th latent feature's dictionary vector.
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.decoder.weight.div_(norms)

    def forward(self, x: torch.Tensor):
        latent_pre = self.encoder(x)
        latent = self.activation(latent_pre)
        recon = self.decoder(latent)
        return recon, latent
