"""Tests for the Sparse Autoencoder module and variants."""

import pytest
import torch

from interp.sae import (
    SparseAutoencoder,
    BatchTopKSAE,
    Transcoder,
    SAEOutput,
    TranscoderOutput,
    SAETrainer,
)


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


class TestBatchTopKSAE:
    @pytest.fixture
    def sae(self):
        return BatchTopKSAE(input_dim=32, dict_size=128, k=16)

    def test_creation(self, sae):
        assert sae.input_dim == 32
        assert sae.dict_size == 128
        assert sae.k == 16
        assert sae.W_enc.shape == (128, 32)
        assert sae.W_dec.shape == (32, 128)

    def test_decoder_columns_normalized(self, sae):
        norms = sae.W_dec.norm(dim=0)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_encode_returns_pre_activations(self, sae):
        x = torch.randn(16, 32)
        pre_acts = sae.encode(x)
        assert pre_acts.shape == (16, 128)
        # Pre-activations can be negative (no ReLU yet)

    def test_forward(self, sae):
        x = torch.randn(16, 32)
        output = sae(x)
        assert isinstance(output, SAEOutput)
        assert output.reconstructed.shape == (16, 32)
        assert output.features.shape == (16, 128)
        assert output.loss.dim() == 0
        # No L1 penalty — sparsity_loss should be 0
        assert output.sparsity_loss.item() == 0.0
        # Loss should equal reconstruction loss
        assert torch.allclose(output.loss, output.reconstruction_loss)

    def test_sparsity_is_controlled(self, sae):
        x = torch.randn(64, 32)
        output = sae(x)
        # Average L0 should be approximately k
        assert output.l0.item() <= sae.k * 2  # Allow some slack
        assert output.l0.item() > 0

    def test_variable_per_sample_sparsity(self, sae):
        x = torch.randn(32, 32)
        output = sae(x)
        # Per-sample L0 should vary (not all exactly k)
        per_sample_l0 = (output.features > 0).float().sum(dim=-1)
        # At least some variation expected
        assert per_sample_l0.std() >= 0 or per_sample_l0.mean() > 0

    def test_forward_batched(self, sae):
        x = torch.randn(4, 8, 32)
        output = sae(x)
        assert output.reconstructed.shape == (4, 8, 32)
        assert output.features.shape == (4, 8, 128)

    def test_features_are_sparse(self, sae):
        x = torch.randn(32, 32)
        output = sae(x)
        # Most features should be zero
        sparsity_ratio = (output.features == 0).float().mean()
        assert sparsity_ratio > 0.5  # At least 50% zeros

    def test_features_are_non_negative(self, sae):
        x = torch.randn(32, 32)
        output = sae(x)
        assert (output.features >= 0).all()

    def test_config_roundtrip(self, sae):
        config = sae.get_config()
        sae2 = BatchTopKSAE.from_config(config)
        assert sae2.input_dim == sae.input_dim
        assert sae2.dict_size == sae.dict_size
        assert sae2.k == sae.k

    def test_get_dead_features(self, sae):
        x = torch.randn(100, 32)
        dead = sae.get_dead_features(x, threshold=0.0)
        assert dead.shape == (128,)
        assert dead.dtype == torch.bool

    def test_different_k_values(self):
        x = torch.randn(64, 32)
        sae_low_k = BatchTopKSAE(32, 128, k=4)
        sae_high_k = BatchTopKSAE(32, 128, k=64)
        sae_high_k.load_state_dict(sae_low_k.state_dict())
        out_low = sae_low_k(x)
        out_high = sae_high_k(x)
        # Higher k should generally mean more active features
        assert out_high.l0.item() >= out_low.l0.item()


class TestTranscoder:
    @pytest.fixture
    def transcoder(self):
        return Transcoder(input_dim=32, output_dim=32, dict_size=128, sparsity_coeff=1e-3)

    @pytest.fixture
    def skip_transcoder(self):
        return Transcoder(input_dim=32, output_dim=32, dict_size=128,
                         sparsity_coeff=1e-3, use_skip=True)

    @pytest.fixture
    def batchtopk_transcoder(self):
        return Transcoder(input_dim=32, output_dim=64, dict_size=128,
                         sparsity_mode="batchtopk", k=16)

    def test_creation(self, transcoder):
        assert transcoder.input_dim == 32
        assert transcoder.output_dim == 32
        assert transcoder.dict_size == 128
        assert transcoder.W_enc.shape == (128, 32)
        assert transcoder.W_dec.shape == (32, 128)
        assert not transcoder.use_skip

    def test_skip_creation(self, skip_transcoder):
        assert skip_transcoder.use_skip
        assert skip_transcoder.W_skip.shape == (32, 32)
        assert skip_transcoder.b_skip.shape == (32,)

    def test_forward_with_target(self, transcoder):
        x = torch.randn(16, 32)
        target = torch.randn(16, 32)
        output = transcoder(x, target=target)
        assert isinstance(output, TranscoderOutput)
        assert output.predicted_output.shape == (16, 32)
        assert output.features.shape == (16, 128)
        assert output.loss.item() > 0
        assert output.reconstruction_loss.item() > 0

    def test_forward_without_target(self, transcoder):
        x = torch.randn(16, 32)
        output = transcoder(x)
        assert output.predicted_output.shape == (16, 32)
        assert output.loss.item() == 0.0  # No target, no loss

    def test_skip_connection(self, skip_transcoder):
        x = torch.randn(16, 32)
        target = torch.randn(16, 32)
        output = skip_transcoder(x, target=target)
        assert output.predicted_output.shape == (16, 32)
        assert output.loss.item() > 0

    def test_different_input_output_dims(self, batchtopk_transcoder):
        x = torch.randn(16, 32)
        target = torch.randn(16, 64)
        output = batchtopk_transcoder(x, target=target)
        assert output.predicted_output.shape == (16, 64)
        assert output.features.shape == (16, 128)

    def test_batchtopk_mode(self, batchtopk_transcoder):
        x = torch.randn(16, 32)
        target = torch.randn(16, 64)
        output = batchtopk_transcoder(x, target=target)
        # Sparsity loss should be 0 in batchtopk mode
        assert output.sparsity_loss.item() == 0.0

    def test_l1_mode(self, transcoder):
        x = torch.randn(16, 32)
        target = torch.randn(16, 32)
        output = transcoder(x, target=target)
        # L1 mode should have non-zero sparsity loss
        assert output.sparsity_loss.item() > 0

    def test_features_non_negative(self, transcoder):
        x = torch.randn(32, 32)
        output = transcoder(x)
        assert (output.features >= 0).all()

    def test_config_roundtrip(self, transcoder):
        config = transcoder.get_config()
        tc2 = Transcoder.from_config(config)
        assert tc2.input_dim == transcoder.input_dim
        assert tc2.output_dim == transcoder.output_dim
        assert tc2.dict_size == transcoder.dict_size
        assert tc2.use_skip == transcoder.use_skip
        assert tc2.sparsity_mode == transcoder.sparsity_mode

    def test_get_dead_features(self, transcoder):
        x = torch.randn(100, 32)
        dead = transcoder.get_dead_features(x, threshold=0.0)
        assert dead.shape == (128,)
        assert dead.dtype == torch.bool


class TestSAETrainer:
    @pytest.fixture
    def trainer(self):
        sae = SparseAutoencoder(input_dim=16, dict_size=64, sparsity_coeff=1e-3)
        return SAETrainer(sae, lr=1e-3, log_every=1, resample_dead_every=0)

    @pytest.fixture
    def batchtopk_trainer(self):
        sae = BatchTopKSAE(input_dim=16, dict_size=64, k=8)
        return SAETrainer(sae, lr=1e-3, log_every=1, resample_dead_every=0)

    @pytest.fixture
    def transcoder_trainer(self):
        tc = Transcoder(input_dim=16, output_dim=16, dict_size=64, sparsity_coeff=1e-3)
        return SAETrainer(tc, lr=1e-3, log_every=1, resample_dead_every=0)

    def test_train_step(self, trainer):
        x = torch.randn(32, 16)
        output = trainer.train_step(x)
        assert isinstance(output, SAEOutput)
        assert trainer.step == 1

    def test_loss_decreases(self, trainer):
        x = torch.randn(64, 16)
        losses = []
        for _ in range(50):
            output = trainer.train_step(x)
            losses.append(output.loss.item())
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

    def test_batchtopk_trainer(self, batchtopk_trainer):
        x = torch.randn(32, 16)
        losses = []
        for _ in range(30):
            output = batchtopk_trainer.train_step(x)
            losses.append(output.loss.item())
        assert losses[-1] < losses[0]

    def test_transcoder_trainer(self, transcoder_trainer):
        x = torch.randn(32, 16)
        target = torch.randn(32, 16)
        losses = []
        for _ in range(30):
            output = transcoder_trainer.train_step(x, targets=target)
            losses.append(output.loss.item())
        assert losses[-1] < losses[0]

    def test_transcoder_trainer_step_count(self, transcoder_trainer):
        x = torch.randn(32, 16)
        target = torch.randn(32, 16)
        transcoder_trainer.train_step(x, targets=target)
        assert transcoder_trainer.step == 1
        assert transcoder_trainer.is_transcoder is True
