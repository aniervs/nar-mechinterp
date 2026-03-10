"""
Sparse Autoencoder (SAE) for GNN/NAR activations.

Implements the standard SAE architecture from Anthropic's "Scaling Monosemanticity"
adapted for graph neural network processor activations:
    encoder: h -> ReLU(W_enc @ (h - b_dec) + b_enc)
    decoder: f -> W_dec @ f + b_dec
    loss: MSE(h, h_hat) + lambda * L1(f)

Uses untied weights (W_enc != W_dec^T) as recommended for better feature recovery.
Each (node, message-passing step) is treated as an independent sample, analogous
to how LLM SAEs treat each (token, layer) independently.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEOutput:
    """Output of a forward pass through the SAE."""
    reconstructed: torch.Tensor       # Reconstructed activations
    features: torch.Tensor            # Sparse feature activations (after ReLU)
    loss: torch.Tensor                # Total loss (reconstruction + sparsity)
    reconstruction_loss: torch.Tensor # MSE reconstruction loss
    sparsity_loss: torch.Tensor       # L1 sparsity penalty
    l0: torch.Tensor                  # Mean L0 norm (avg number of active features)


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder with untied weights for extracting monosemantic features.

    Args:
        input_dim: Dimension of the input activations (hidden_dim of the NAR processor).
        dict_size: Number of dictionary features. Typical expansion factors: 4x, 8x, 16x.
        sparsity_coeff: L1 penalty coefficient (lambda).
    """

    def __init__(
        self,
        input_dim: int,
        dict_size: int,
        sparsity_coeff: float = 1e-3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.sparsity_coeff = sparsity_coeff

        # Encoder: W_enc @ (h - b_dec) + b_enc
        self.W_enc = nn.Parameter(torch.empty(dict_size, input_dim))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))

        # Decoder: W_dec @ f + b_dec
        self.W_dec = nn.Parameter(torch.empty(input_dim, dict_size))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Kaiming uniform, then normalize decoder columns."""
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        # Normalize decoder columns to unit norm (standard SAE practice)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to sparse feature space."""
        # Subtract decoder bias (centering), then project and apply ReLU
        return F.relu(F.linear(x - self.b_dec, self.W_enc, self.b_enc))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activation space."""
        return F.linear(features, self.W_dec, self.b_dec)

    def forward(self, x: torch.Tensor) -> SAEOutput:
        """
        Forward pass: encode -> decode, compute losses.

        Args:
            x: Input activations, shape (..., input_dim). Can be any batch shape.

        Returns:
            SAEOutput with reconstructed activations, features, and losses.
        """
        features = self.encode(x)
        reconstructed = self.decode(features)

        # Reconstruction loss: MSE
        reconstruction_loss = F.mse_loss(reconstructed, x)

        # Sparsity loss: L1 on feature activations
        sparsity_loss = features.abs().mean()

        # Total loss
        loss = reconstruction_loss + self.sparsity_coeff * sparsity_loss

        # L0: average number of non-zero features per sample
        l0 = (features > 0).float().sum(dim=-1).mean()

        return SAEOutput(
            reconstructed=reconstructed,
            features=features,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=sparsity_loss,
            l0=l0,
        )

    @torch.no_grad()
    def normalize_decoder(self):
        """Normalize decoder weight columns to unit norm (call after each optimizer step)."""
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    @torch.no_grad()
    def get_feature_norms(self) -> torch.Tensor:
        """Get the norm of each decoder feature vector (column of W_dec)."""
        return self.W_dec.norm(dim=0)

    @torch.no_grad()
    def get_top_features(
        self,
        x: torch.Tensor,
        k: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the top-k most active features for given activations.

        Args:
            x: Input activations, shape (..., input_dim).
            k: Number of top features to return.

        Returns:
            (values, indices): Top-k feature activation values and their indices.
        """
        features = self.encode(x)
        # Flatten all batch dims, take top-k per sample
        flat = features.reshape(-1, self.dict_size)
        values, indices = flat.topk(k, dim=-1)
        return values, indices

    @torch.no_grad()
    def get_dead_features(
        self,
        activation_buffer: torch.Tensor,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        """
        Identify dead features that never activate above threshold.

        Args:
            activation_buffer: Collected activations, shape (num_samples, input_dim).
            threshold: Activation threshold below which a feature is considered dead.

        Returns:
            Boolean tensor of shape (dict_size,), True for dead features.
        """
        features = self.encode(activation_buffer)
        max_activation = features.max(dim=0).values
        return max_activation <= threshold

    def get_config(self) -> dict:
        """Return a dict of the SAE's configuration."""
        return {
            "input_dim": self.input_dim,
            "dict_size": self.dict_size,
            "sparsity_coeff": self.sparsity_coeff,
        }

    @classmethod
    def from_config(cls, config: dict) -> "SparseAutoencoder":
        """Create an SAE from a config dict."""
        return cls(**config)


class SAETrainer:
    """
    Training loop for the Sparse Autoencoder.

    Handles optimizer setup, decoder normalization, dead feature resampling,
    and logging.
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        lr: float = 3e-4,
        resample_dead_every: int = 25000,
        dead_feature_threshold: float = 1e-8,
        log_every: int = 100,
    ):
        self.sae = sae
        self.optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
        self.resample_dead_every = resample_dead_every
        self.dead_feature_threshold = dead_feature_threshold
        self.log_every = log_every
        self.step = 0
        self.log_history: list[dict] = []
        self._activation_buffer: list[torch.Tensor] = []
        self._buffer_max_size = 10000

    def train_step(self, activations: torch.Tensor) -> SAEOutput:
        """
        Perform a single training step.

        Args:
            activations: Batch of activation vectors, shape (batch_size, input_dim).

        Returns:
            SAEOutput from the forward pass.
        """
        self.sae.train()
        output = self.sae(activations)

        self.optimizer.zero_grad()
        output.loss.backward()
        self.optimizer.step()

        # Normalize decoder columns after each step
        self.sae.normalize_decoder()

        # Buffer activations for dead feature resampling
        self._activation_buffer.append(activations.detach())
        if len(self._activation_buffer) > self._buffer_max_size // max(1, activations.shape[0]):
            self._activation_buffer.pop(0)

        self.step += 1

        # Log periodically
        if self.step % self.log_every == 0:
            log_entry = {
                "step": self.step,
                "loss": output.loss.item(),
                "reconstruction_loss": output.reconstruction_loss.item(),
                "sparsity_loss": output.sparsity_loss.item(),
                "l0": output.l0.item(),
            }
            self.log_history.append(log_entry)

        # Resample dead features periodically
        if (
            self.resample_dead_every > 0
            and self.step % self.resample_dead_every == 0
            and len(self._activation_buffer) > 0
        ):
            self._resample_dead_features()

        return output

    @torch.no_grad()
    def _resample_dead_features(self):
        """Resample dead features using high-loss activation vectors."""
        buffer = torch.cat(self._activation_buffer, dim=0)
        dead_mask = self.sae.get_dead_features(buffer, self.dead_feature_threshold)
        num_dead = dead_mask.sum().item()

        if num_dead == 0:
            return

        # Find activations with highest reconstruction error
        output = self.sae(buffer)
        per_sample_loss = (output.reconstructed - buffer).pow(2).sum(dim=-1)
        _, top_indices = per_sample_loss.topk(min(num_dead, len(buffer)))

        # Resample dead encoder/decoder directions from high-loss activations
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        for i, dead_idx in enumerate(dead_indices):
            if i >= len(top_indices):
                break
            source = buffer[top_indices[i]]
            # Set decoder column to the (normalized) high-loss activation
            self.sae.W_dec.data[:, dead_idx] = F.normalize(source, dim=0)
            # Set encoder row to match
            self.sae.W_enc.data[dead_idx] = F.normalize(source, dim=0)
            # Reset biases
            self.sae.b_enc.data[dead_idx] = 0.0

    def get_training_stats(self) -> dict:
        """Get summary statistics from training."""
        if not self.log_history:
            return {}
        recent = self.log_history[-10:]
        return {
            "steps": self.step,
            "recent_loss": sum(e["loss"] for e in recent) / len(recent),
            "recent_recon_loss": sum(e["reconstruction_loss"] for e in recent) / len(recent),
            "recent_sparsity_loss": sum(e["sparsity_loss"] for e in recent) / len(recent),
            "recent_l0": sum(e["l0"] for e in recent) / len(recent),
        }
