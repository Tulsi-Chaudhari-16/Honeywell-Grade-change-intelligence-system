"""
GCIS ML Pipeline — 1D-CNN Autoencoder Transition Embedding Model

Produces a fixed-length vector representation of a grade transition's
multivariate time-series window for approximate nearest-neighbour similarity
search in Qdrant.

Architecture:
  Encoder: 3x Conv1d layers (with BatchNorm + ReLU) -> AdaptiveAvgPool -> Linear -> EMBEDDING_DIM
  Decoder: Linear -> reshape -> 3x ConvTranspose1d -> reconstructed input
  Loss:     Reconstruction loss (MSE)

Why a purpose-built model instead of a generic sentence embedder:
  The input is normalised numeric time-series (n_features × n_timesteps).
  A sentence/text embedder has no domain alignment to this input type.
  A trained autoencoder learns the actual covariance structure of grade transitions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import EMBEDDING_DIM, INPUT_WINDOW_STEPS


# ─── Architecture ─────────────────────────────────────────────────────────────


class TransitionEncoder(nn.Module):
    """Encoder half: (batch, n_features, seq_len) -> (batch, EMBEDDING_DIM)"""

    def __init__(self, n_features: int, embedding_dim: int = EMBEDDING_DIM) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),   # -> (batch, 128, 4)
        )
        self.proj = nn.Linear(128 * 4, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        out = out.flatten(1)
        return self.proj(out)


class TransitionDecoder(nn.Module):
    """Decoder half: (batch, EMBEDDING_DIM) -> (batch, n_features, seq_len)"""

    def __init__(
        self, n_features: int, seq_len: int, embedding_dim: int = EMBEDDING_DIM
    ) -> None:
        super().__init__()
        self._n_features = n_features
        self._seq_len = seq_len
        self.proj = nn.Linear(embedding_dim, 128 * 4)
        self.layers = nn.Sequential(
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, n_features, kernel_size=3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.proj(z).view(-1, 128, 4)
        out = self.layers(out)
        # Interpolate to exact original seq_len
        return nn.functional.interpolate(out, size=self._seq_len, mode="linear", align_corners=False)


class TransitionAutoencoder(nn.Module):
    def __init__(
        self, n_features: int, seq_len: int, embedding_dim: int = EMBEDDING_DIM
    ) -> None:
        super().__init__()
        self.encoder = TransitionEncoder(n_features, embedding_dim)
        self.decoder = TransitionDecoder(n_features, seq_len, embedding_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


# ─── Training ─────────────────────────────────────────────────────────────────


def train(
    X_train: np.ndarray,
    X_val: np.ndarray,
    output_dir: Path,
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> dict[str, Any]:
    """
    Train the autoencoder.

    Args:
        X_train: (n_transitions, n_features, seq_len)
        X_val:   (n_transitions, n_features, seq_len)
        output_dir: where to save weights + config

    Returns:
        metrics dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Embedding] Training on {device}")

    n_features = X_train.shape[1]
    seq_len = X_train.shape[2]

    def _loader(X: np.ndarray, shuffle: bool) -> DataLoader:
        t = torch.tensor(X, dtype=torch.float32)
        return DataLoader(TensorDataset(t), batch_size=batch_size, shuffle=shuffle)

    train_loader = _loader(X_train, shuffle=True)
    val_loader = _loader(X_val, shuffle=False)

    model = TransitionAutoencoder(n_features, seq_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    patience_count = 0
    patience = 12

    for epoch in range(1, epochs + 1):
        model.train()
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            _, x_hat = model(xb)
            loss = loss_fn(x_hat, xb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                _, x_hat = model(xb)
                val_losses.append(loss_fn(x_hat, xb).item())

        val_loss = float(np.mean(val_losses))
        if epoch % 20 == 0:
            print(f"  [Embedding] Epoch {epoch}/{epochs} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), output_dir / "encoder_best.pt")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  [Embedding] Early stopping at epoch {epoch}")
                break

    arch_config = {
        "n_features": n_features,
        "seq_len": seq_len,
        "embedding_dim": EMBEDDING_DIM,
    }
    (output_dir / "arch_config.json").write_text(json.dumps(arch_config))

    metrics = {
        "best_val_reconstruction_loss": round(best_val_loss, 6),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_features": n_features,
        "seq_len": seq_len,
        "embedding_dim": EMBEDDING_DIM,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[Embedding] Best val loss: {best_val_loss:.6f}")
    return metrics


# ─── Inference ────────────────────────────────────────────────────────────────


class TransitionEmbedder:
    """
    Loaded encoder for live embedding at inference time.
    Called by Historical Retrieval Agent via infer/embed.py.
    """

    def __init__(self, encoder: TransitionEncoder, device: torch.device) -> None:
        self._encoder = encoder
        self._device = device

    @classmethod
    def load(cls, model_dir: Path) -> "TransitionEmbedder":
        config = json.loads((model_dir / "arch_config.json").read_text())
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = TransitionEncoder(config["n_features"], config["embedding_dim"])
        encoder.load_state_dict(
            torch.load(model_dir / "encoder_best.pt", map_location=device),
            strict=False,   # encoder is a submodule, weights may be prefixed
        )
        encoder.to(device).eval()
        return cls(encoder, device)

    def embed(self, sequence: np.ndarray) -> np.ndarray:
        """
        Args:
            sequence: (n_features, seq_len) array — normalised feature window

        Returns:
            embedding: (EMBEDDING_DIM,) float32 numpy array
        """
        x = torch.tensor(sequence[np.newaxis, :, :], dtype=torch.float32).to(self._device)
        with torch.no_grad():
            z = self._encoder(x)
        return z.cpu().numpy()[0]
