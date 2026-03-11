"""
Sparse Autoencoder (SAE) variants for GNN/NAR activations.

Implements three SAE architectures for graph neural network processor activations:

1. **SparseAutoencoder** (Standard, baseline):
   encoder: h -> ReLU(W_enc @ (h - b_dec) + b_enc)
   decoder: f -> W_dec @ f + b_dec
   loss: MSE(h, h_hat) + lambda * L1(f)

2. **BatchTopKSAE** (Primary, recommended):
   Same encoder/decoder, but sparsity enforced by keeping top-K activations
   across the entire batch (BatchTopK). Eliminates activation shrinkage and
   allows adaptive per-sample sparsity while controlling average sparsity.
   Reference: Bussmann et al., "BatchTopK Sparse Autoencoders" (arXiv 2412.06410)

3. **Transcoder** (For circuit analysis):
   Maps processor MLP input -> sparse features -> processor MLP output.
   Each latent represents a sparse computational motif.
   Reference: Dunefsky et al., "Transcoders Find Interpretable LLM Feature Circuits" (NeurIPS 2024)

Uses untied weights (W_enc != W_dec^T) as recommended for better feature recovery.
Each (node, message-passing step) is treated as an independent sample.
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


class BatchTopKSAE(nn.Module):
    """
    Batch Top-K Sparse Autoencoder.

    Instead of L1 penalty, enforces sparsity by keeping only the top-K activations
    across the entire batch. This allows variable per-sample sparsity while
    controlling average sparsity (avg K features per sample).

    Key advantages over standard SAE:
    - No activation shrinkage (no L1 penalty competing with reconstruction)
    - Adaptive per-sample sparsity (complex inputs get more features)
    - Direct sparsity control (set K, no lambda tuning)

    Args:
        input_dim: Dimension of the input activations.
        dict_size: Number of dictionary features.
        k: Average number of active features per sample.
    """

    def __init__(
        self,
        input_dim: int,
        dict_size: int,
        k: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.k = k

        # Same encoder/decoder structure as standard SAE
        self.W_enc = nn.Parameter(torch.empty(dict_size, input_dim))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(torch.empty(input_dim, dict_size))
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode activations to pre-activation feature space (before top-k)."""
        return F.linear(x - self.b_dec, self.W_enc, self.b_enc)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features back to activation space."""
        return F.linear(features, self.W_dec, self.b_dec)

    def _batch_topk(self, pre_activations: torch.Tensor) -> torch.Tensor:
        """
        Apply BatchTopK: keep top-(K * batch_size) activations globally,
        then redistribute to per-sample sparse vectors.

        Args:
            pre_activations: Shape (batch_size, dict_size), raw encoder outputs.

        Returns:
            Sparse activations with the same shape, zeros everywhere except top-k.
        """
        batch_size = pre_activations.shape[0]
        total_k = self.k * batch_size

        # ReLU first — only consider positive pre-activations
        relu_acts = F.relu(pre_activations)

        # Flatten, take global top-k
        flat = relu_acts.reshape(-1)
        total_k = min(total_k, flat.numel())

        # Get the k-th largest value as threshold
        threshold = flat.kthvalue(flat.numel() - total_k + 1).values

        # Keep only values above threshold
        sparse_acts = relu_acts * (relu_acts >= threshold).float()
        return sparse_acts

    def forward(self, x: torch.Tensor) -> SAEOutput:
        """
        Forward pass with BatchTopK sparsity.

        Args:
            x: Input activations, shape (batch_size, input_dim) or (*, input_dim).

        Returns:
            SAEOutput with reconstructed activations, sparse features, and losses.
        """
        original_shape = x.shape
        flat_x = x.reshape(-1, self.input_dim)

        pre_acts = self.encode(flat_x)
        features = self._batch_topk(pre_acts)
        reconstructed = self.decode(features)

        # Reshape back
        features = features.reshape(*original_shape[:-1], self.dict_size)
        reconstructed = reconstructed.reshape(original_shape)

        # Loss is purely reconstruction (sparsity is architectural, not penalized)
        reconstruction_loss = F.mse_loss(reconstructed, x)
        sparsity_loss = torch.tensor(0.0, device=x.device)
        loss = reconstruction_loss

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
        """Normalize decoder weight columns to unit norm."""
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    @torch.no_grad()
    def get_feature_norms(self) -> torch.Tensor:
        return self.W_dec.norm(dim=0)

    @torch.no_grad()
    def get_top_features(self, x: torch.Tensor, k: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        pre_acts = F.relu(self.encode(x))
        flat = pre_acts.reshape(-1, self.dict_size)
        values, indices = flat.topk(k, dim=-1)
        return values, indices

    @torch.no_grad()
    def get_dead_features(self, activation_buffer: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        pre_acts = F.relu(self.encode(activation_buffer))
        max_activation = pre_acts.max(dim=0).values
        return max_activation <= threshold

    def get_config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "dict_size": self.dict_size,
            "k": self.k,
        }

    @classmethod
    def from_config(cls, config: dict) -> "BatchTopKSAE":
        return cls(**config)


@dataclass
class TranscoderOutput:
    """Output of a forward pass through the Transcoder."""
    predicted_output: torch.Tensor   # Predicted MLP output
    features: torch.Tensor           # Sparse feature activations
    loss: torch.Tensor               # Total loss
    reconstruction_loss: torch.Tensor  # MSE between predicted and actual output
    sparsity_loss: torch.Tensor      # Sparsity penalty
    l0: torch.Tensor                 # Mean L0 norm


class Transcoder(nn.Module):
    """
    Transcoder for mechanistic circuit analysis of NAR processors.

    Unlike an SAE which autoencodes activations (input -> features -> input),
    a transcoder maps **component input -> features -> component output**.
    Each latent represents a sparse computational motif that transforms
    the input in a specific way.

    Supports an optional skip connection (Skip Transcoder) which adds an
    affine transformation from input to output, allowing the sparse features
    to model only the nonlinear residual.

    Args:
        input_dim: Dimension of the component's input.
        output_dim: Dimension of the component's output.
        dict_size: Number of dictionary features.
        sparsity_coeff: L1 penalty coefficient.
        use_skip: If True, adds a learned skip connection (Skip Transcoder).
        sparsity_mode: "l1" for standard L1 penalty, "batchtopk" for BatchTopK.
        k: Average number of active features per sample (only for batchtopk mode).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dict_size: int,
        sparsity_coeff: float = 1e-3,
        use_skip: bool = False,
        sparsity_mode: str = "l1",
        k: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dict_size = dict_size
        self.sparsity_coeff = sparsity_coeff
        self.use_skip = use_skip
        self.sparsity_mode = sparsity_mode
        self.k = k

        # Encoder: input -> sparse features
        self.W_enc = nn.Parameter(torch.empty(dict_size, input_dim))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))

        # Decoder: sparse features -> output
        self.W_dec = nn.Parameter(torch.empty(output_dim, dict_size))
        self.b_dec = nn.Parameter(torch.zeros(output_dim))

        # Optional skip connection: linear map from input to output
        if use_skip:
            self.W_skip = nn.Parameter(torch.empty(output_dim, input_dim))
            self.b_skip = nn.Parameter(torch.zeros(output_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)
        if self.use_skip:
            nn.init.kaiming_uniform_(self.W_skip)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to sparse feature space."""
        pre_acts = F.linear(x, self.W_enc, self.b_enc)
        if self.sparsity_mode == "batchtopk":
            return self._batch_topk(pre_acts)
        return F.relu(pre_acts)

    def _batch_topk(self, pre_activations: torch.Tensor) -> torch.Tensor:
        """Apply BatchTopK sparsity."""
        original_shape = pre_activations.shape
        flat_pre = pre_activations.reshape(-1, self.dict_size)
        batch_size = flat_pre.shape[0]
        total_k = min(self.k * batch_size, flat_pre.numel())

        relu_acts = F.relu(flat_pre)
        flat = relu_acts.reshape(-1)
        threshold = flat.kthvalue(flat.numel() - total_k + 1).values
        sparse_acts = relu_acts * (relu_acts >= threshold).float()
        return sparse_acts.reshape(original_shape)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode sparse features to predicted output."""
        return F.linear(features, self.W_dec, self.b_dec)

    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> TranscoderOutput:
        """
        Forward pass: encode input -> sparse features -> predicted output.

        Args:
            x: Component input, shape (..., input_dim).
            target: Component output to predict, shape (..., output_dim).
                If None, loss is not computed (inference mode).

        Returns:
            TranscoderOutput with predicted output, features, and losses.
        """
        features = self.encode(x)
        predicted = self.decode(features)

        # Add skip connection if enabled
        if self.use_skip:
            skip_out = F.linear(x, self.W_skip, self.b_skip)
            predicted = predicted + skip_out

        # Compute losses if target is provided
        if target is not None:
            reconstruction_loss = F.mse_loss(predicted, target)
            if self.sparsity_mode == "l1":
                sparsity_loss = features.abs().mean()
                loss = reconstruction_loss + self.sparsity_coeff * sparsity_loss
            else:
                sparsity_loss = torch.tensor(0.0, device=x.device)
                loss = reconstruction_loss
        else:
            reconstruction_loss = torch.tensor(0.0, device=x.device)
            sparsity_loss = torch.tensor(0.0, device=x.device)
            loss = torch.tensor(0.0, device=x.device)

        l0 = (features > 0).float().sum(dim=-1).mean()

        return TranscoderOutput(
            predicted_output=predicted,
            features=features,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=sparsity_loss,
            l0=l0,
        )

    @torch.no_grad()
    def normalize_decoder(self):
        """Normalize decoder weight columns to unit norm."""
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    @torch.no_grad()
    def get_feature_norms(self) -> torch.Tensor:
        return self.W_dec.norm(dim=0)

    @torch.no_grad()
    def get_dead_features(self, input_buffer: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        features = self.encode(input_buffer)
        max_activation = features.max(dim=0).values
        return max_activation <= threshold

    def get_config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "dict_size": self.dict_size,
            "sparsity_coeff": self.sparsity_coeff,
            "use_skip": self.use_skip,
            "sparsity_mode": self.sparsity_mode,
            "k": self.k,
        }

    @classmethod
    def from_config(cls, config: dict) -> "Transcoder":
        return cls(**config)


class SAETrainer:
    """
    Training loop for SAE variants (SparseAutoencoder, BatchTopKSAE, or Transcoder).

    Handles optimizer setup, decoder normalization, dead feature resampling,
    and logging. Works with any of the three SAE variants.
    """

    def __init__(
        self,
        sae: nn.Module,
        lr: float = 3e-4,
        resample_dead_every: int = 25000,
        dead_feature_threshold: float = 1e-8,
        log_every: int = 100,
    ):
        self.sae = sae
        self.is_transcoder = isinstance(sae, Transcoder)
        self.optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
        self.resample_dead_every = resample_dead_every
        self.dead_feature_threshold = dead_feature_threshold
        self.log_every = log_every
        self.step = 0
        self.log_history: list[dict] = []
        self._activation_buffer: list[torch.Tensor] = []
        self._buffer_max_size = 10000

    def train_step(
        self,
        activations: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> SAEOutput | TranscoderOutput:
        """
        Perform a single training step.

        Args:
            activations: Batch of activation vectors, shape (batch_size, input_dim).
            targets: For Transcoder only — component output targets, shape (batch_size, output_dim).

        Returns:
            SAEOutput or TranscoderOutput from the forward pass.
        """
        self.sae.train()
        if self.is_transcoder:
            output = self.sae(activations, target=targets)
        else:
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
        if self.is_transcoder:
            # For transcoder, we can only resample encoder side
            features = self.sae.encode(buffer)
            per_sample_loss = features.sum(dim=-1)  # proxy — no target available
        else:
            output = self.sae(buffer)
            per_sample_loss = (output.reconstructed - buffer).pow(2).sum(dim=-1)

        _, top_indices = per_sample_loss.topk(min(num_dead, len(buffer)))

        # Resample dead encoder/decoder directions from high-loss activations
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        for i, dead_idx in enumerate(dead_indices):
            if i >= len(top_indices):
                break
            source = buffer[top_indices[i]]
            # Set encoder row to normalized high-loss activation
            self.sae.W_enc.data[dead_idx] = F.normalize(source, dim=0)
            # Set decoder column similarly (only first input_dim rows matter for transcoder)
            dec_col_dim = self.sae.W_dec.shape[0]
            if dec_col_dim == source.shape[0]:
                self.sae.W_dec.data[:, dead_idx] = F.normalize(source, dim=0)
            else:
                # Transcoder: output_dim != input_dim, just reinit decoder column
                nn.init.kaiming_uniform_(self.sae.W_dec.data[:, dead_idx:dead_idx+1])
            # Reset encoder bias
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
