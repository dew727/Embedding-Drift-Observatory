"""PyTorch autoencoder for learning dense embeddings from tabular data."""

import torch
import torch.nn as nn


def _build_block(in_dim: int, out_dim: int, dropout: float) -> list[nn.Module]:
    """Linear → BatchNorm → ReLU → Dropout block."""
    layers = [nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers


class Autoencoder(nn.Module):
    """Encoder-decoder autoencoder for tabular data.

    The encoder output (latent vector) is the embedding used downstream.
    BatchNorm stabilizes training on heterogeneous tabular features;
    Dropout regularizes against overfitting on small datasets.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: Number of input features.
            latent_dim: Size of the bottleneck embedding.
            hidden_dims: Layer sizes for the encoder; decoder mirrors these in reverse.
            dropout: Dropout rate applied after each hidden activation (0 = disabled).
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Encoder: input → hidden_dims → latent
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.extend(_build_block(prev, h, dropout))
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder: latent → reversed hidden_dims → input (no final activation — raw reconstruction)
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend(_build_block(prev, h, dropout))
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, latent_embedding)."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent embedding only. Used by EmbeddingPipeline.transform()."""
        return self.encoder(x)


def train_autoencoder(
    model: Autoencoder,
    X: "np.ndarray",  # noqa: F821
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 5,
    val_fraction: float = 0.1,
    device: str = "cpu",
) -> Autoencoder:
    """Train an Autoencoder on baseline data with early stopping.

    Args:
        model: Autoencoder instance to train in-place.
        X: Baseline feature matrix (n_samples, n_features), already normalized.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        patience: Early-stop after this many epochs with no val-loss improvement.
        val_fraction: Fraction of X held out for early-stopping validation.
        device: 'cpu' or 'cuda'.

    Returns:
        Trained model in eval() mode.
    """
    import numpy as np

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Hold out a small validation split for early stopping
    n = len(X)
    n_val = max(1, int(n * val_fraction))
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    X_train = torch.tensor(X[idx[n_val:]], dtype=torch.float32, device=device)
    X_val = torch.tensor(X[idx[:n_val]], dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_train), device=device)
        for i in range(0, len(X_train), batch_size):
            batch = X_train[perm[i : i + batch_size]]
            x_hat, _ = model(batch)
            loss = criterion(x_hat, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val)[0], X_val).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model
