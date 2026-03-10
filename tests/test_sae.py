"""Tests for the Sparse Autoencoder module."""

import pytest
import torch

from interp.sae import SparseAutoencoder, SAEOutput, SAETrainer


class TestSparseAutoencoder:
    @pytest.fixture
    def sae(self):
        return SparseAutoencoder(input_dim=32, dict_size=128, sparsity_coeff=1e-3)

    def test_creation(self, sae):
        assert sae.input_dim == 32
        assert sae.dict_size == 128
        assert sae.W_enc.shape == (128, 32)
        assert sae.W_dec.shape == (32, 128)
        assert sae.b_enc.shape == (128,)
        assert sae.b_dec.shape == (32,)

    def test_decoder_columns_normalized(self, sae):
        norms = sae.W_dec.norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_encode(self, sae):
        x = torch.randn(16, 32)
        features = sae.encode(x)
        assert features.shape == (16, 128)
        assert (features >= 0).all()  # ReLU output

    def test_decode(self, sae):
        features = torch.randn(16, 128).relu()
        reconstructed = sae.decode(features)
        assert reconstructed.shape == (16, 32)

    def test_forward(self, sae):
        x = torch.randn(16, 32)
        output = sae(x)
        assert isinstance(output, SAEOutput)
        assert output.reconstructed.shape == (16, 32)
        assert output.features.shape == (16, 128)
        assert output.loss.dim() == 0
        assert output.reconstruction_loss.dim() == 0
        assert output.sparsity_loss.dim() == 0
        assert output.l0.dim() == 0

    def test_forward_batched(self, sae):
        x = torch.randn(4, 8, 32)
        output = sae(x)
        assert output.reconstructed.shape == (4, 8, 32)
        assert output.features.shape == (4, 8, 128)

    def test_loss_components(self, sae):
        x = torch.randn(64, 32)
        output = sae(x)
        # Total loss should be recon + sparsity_coeff * sparsity
        expected = output.reconstruction_loss + sae.sparsity_coeff * output.sparsity_loss
        assert torch.allclose(output.loss, expected)

    def test_normalize_decoder(self, sae):
        # Perturb decoder weights
        sae.W_dec.data *= 2.0
        sae.normalize_decoder()
        norms = sae.W_dec.norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_get_top_features(self, sae):
        x = torch.randn(16, 32)
        values, indices = sae.get_top_features(x, k=5)
        assert values.shape == (16, 5)
        assert indices.shape == (16, 5)
        # Values should be sorted descending
        assert (values[:, :-1] >= values[:, 1:]).all()

    def test_get_dead_features(self, sae):
        x = torch.randn(100, 32)
        dead = sae.get_dead_features(x, threshold=0.0)
        assert dead.shape == (128,)
        assert dead.dtype == torch.bool

    def test_config_roundtrip(self, sae):
        config = sae.get_config()
        sae2 = SparseAutoencoder.from_config(config)
        assert sae2.input_dim == sae.input_dim
        assert sae2.dict_size == sae.dict_size
        assert sae2.sparsity_coeff == sae.sparsity_coeff

    def test_sparsity_increases_with_coeff(self):
        x = torch.randn(64, 32)
        sae_low = SparseAutoencoder(32, 128, sparsity_coeff=1e-5)
        sae_high = SparseAutoencoder(32, 128, sparsity_coeff=1.0)
        # Copy weights so the only difference is the loss coefficient
        sae_high.load_state_dict(sae_low.state_dict())
        out_low = sae_low(x)
        out_high = sae_high(x)
        # Higher sparsity coeff should yield higher total loss
        assert out_high.loss > out_low.loss


class TestSAETrainer:
    @pytest.fixture
    def trainer(self):
        sae = SparseAutoencoder(input_dim=16, dict_size=64, sparsity_coeff=1e-3)
        return SAETrainer(sae, lr=1e-3, log_every=1, resample_dead_every=0)

    def test_train_step(self, trainer):
        x = torch.randn(32, 16)
        output = trainer.train_step(x)
        assert isinstance(output, SAEOutput)
        assert trainer.step == 1

    def test_loss_decreases(self, trainer):
        # Train on a fixed batch and verify loss goes down
        x = torch.randn(64, 16)
        losses = []
        for _ in range(50):
            output = trainer.train_step(x)
            losses.append(output.loss.item())
        # Loss should generally decrease
        assert losses[-1] < losses[0]

    def test_logging(self, trainer):
        x = torch.randn(32, 16)
        for _ in range(5):
            trainer.train_step(x)
        assert len(trainer.log_history) == 5
        assert "loss" in trainer.log_history[0]
        assert "l0" in trainer.log_history[0]

    def test_get_training_stats(self, trainer):
        x = torch.randn(32, 16)
        for _ in range(10):
            trainer.train_step(x)
        stats = trainer.get_training_stats()
        assert "steps" in stats
        assert "recent_loss" in stats
        assert stats["steps"] == 10

    def test_decoder_stays_normalized(self, trainer):
        x = torch.randn(32, 16)
        for _ in range(10):
            trainer.train_step(x)
        norms = trainer.sae.W_dec.norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
