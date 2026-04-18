"""PyTorch autoencoder for learning dense embeddings from tabular data."""

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Encoder-decoder autoencoder for tabular data.

    The encoder output (latent vector) serves as the embedding.
    """

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror of encoder)
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
