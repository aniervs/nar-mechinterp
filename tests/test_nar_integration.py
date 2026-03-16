"""Integration tests for NAR model training on real CLRS data.

These tests verify that the full pipeline — data loading, batching,
model forward/backward, and loss computation — works end-to-end.
"""

import pytest
import torch
import torch.nn.functional as F

from models.nar_model import NARModel, NAROutput
from data.clrs_dataset import (
    get_clrs_dataloader,
    get_algorithm_spec,
    batch_to_model_inputs,
    spec_to_model_types,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bfs_loader(tmp_path_factory):
    data_dir = str(tmp_path_factory.mktemp("clrs_integ"))
    return get_clrs_dataloader(
        "bfs", batch_size=4, num_samples=16, num_nodes=8,
        edge_probability=0.3, data_dir=data_dir,
    )


@pytest.fixture(scope="module")
def bfs_spec():
    return get_algorithm_spec("bfs")


@pytest.fixture(scope="module")
def bfs_types(bfs_spec):
    return spec_to_model_types(bfs_spec)


@pytest.fixture
def small_model():
    return NARModel(
        hidden_dim=32, num_layers=2, num_heads=4,
        processor_type="mpnn", decode_hints=True, encode_hints=True,
    )


# ---------------------------------------------------------------------------
# Shape / forward tests with real data
# ---------------------------------------------------------------------------

class TestBatchToModelInputs:
    """Verify batch_to_model_inputs produces the shapes the model expects."""

    def test_inputs_have_adjacency_and_features(self, bfs_loader, bfs_spec):
        batch = next(iter(bfs_loader))
        inputs, outputs, hints = batch_to_model_inputs(batch, bfs_spec)
        assert "adjacency" in inputs
        assert "node_features" in inputs
        assert inputs["adjacency"].dim() == 3  # (B, N, N)
        assert inputs["node_features"].dim() == 3  # (B, N, F)

    def test_outputs_pointer_is_index_tensor(self, bfs_loader, bfs_spec):
        batch = next(iter(bfs_loader))
        inputs, outputs, hints = batch_to_model_inputs(batch, bfs_spec)
        assert "pi" in outputs
        pi = outputs["pi"]
        assert pi.dim() == 2  # (B, N) — integer indices
        assert pi.dtype == torch.long

    def test_hint_shapes(self, bfs_loader, bfs_spec):
        batch = next(iter(bfs_loader))
        inputs, outputs, hints = batch_to_model_inputs(batch, bfs_spec)
        B = batch.num_graphs
        N = inputs["adjacency"].shape[1]
        # reach_h: node mask hint — (B, N, T)
        assert "reach_h" in hints
        assert hints["reach_h"].dim() == 3
        assert hints["reach_h"].shape[0] == B
        assert hints["reach_h"].shape[1] == N
        # pi_h: node pointer hint — (B, N, T)
        assert "pi_h" in hints
        assert hints["pi_h"].dim() == 3
        assert hints["pi_h"].shape[0] == B
        assert hints["pi_h"].shape[1] == N

    def test_pointer_hint_values_in_range(self, bfs_loader, bfs_spec):
        batch = next(iter(bfs_loader))
        inputs, outputs, hints = batch_to_model_inputs(batch, bfs_spec)
        N = inputs["adjacency"].shape[1]
        pi_h = hints["pi_h"]
        assert pi_h.min() >= 0
        assert pi_h.max() < N


# ---------------------------------------------------------------------------
# Forward pass tests with real data
# ---------------------------------------------------------------------------

class TestForwardWithRealData:
    """Model forward pass on real batches — no training, just shape/loss checks."""

    def test_forward_produces_output_loss(self, small_model, bfs_loader, bfs_spec, bfs_types):
        output_types, hint_types = bfs_types
        batch = next(iter(bfs_loader))
        inputs, outputs, hints = batch_to_model_inputs(batch, bfs_spec)
        num_steps = batch.lengths.max().item()

        result = small_model(
            inputs=inputs, outputs=outputs, hints=hints,
            output_types=output_types, hint_types=hint_types,
            num_steps=num_steps,
        )
        assert isinstance(result, NAROutput)
        assert result.output_loss.item() > 0
        assert result.total_loss.item() > 0
        assert result.total_loss.requires_grad

    def test_forward_with_hints_produces_hint_loss(self, small_model, bfs_loader, bfs_spec, bfs_types):
        output_types, hint_types = bfs_types
        batch = next(iter(bfs_loader))
        inputs, outputs, hints = batch_to_model_inputs(batch, bfs_spec)
        num_steps = batch.lengths.max().item()

        result = small_model(
            inputs=inputs, outputs=outputs, hints=hints,
            output_types=output_types, hint_types=hint_types,
            num_steps=num_steps,
        )
        assert result.hint_loss is not None
        assert result.hint_loss.item() > 0

    def test_pointer_prediction_shape(self, small_model, bfs_loader, bfs_spec, bfs_types):
        output_types, hint_types = bfs_types
        batch = next(iter(bfs_loader))
        inputs, outputs, hints = batch_to_model_inputs(batch, bfs_spec)
        num_steps = batch.lengths.max().item()

        result = small_model(
            inputs=inputs, outputs=outputs, hints=hints,
            output_types=output_types, hint_types=hint_types,
            num_steps=num_steps,
        )
        # BFS output is 'pi' (predecessor pointer)
        assert "pi" in result.predictions
        pred = result.predictions["pi"]
        assert pred.dim() == 3  # (B, N, N) logits

    def test_backward_pass_succeeds(self, small_model, bfs_loader, bfs_spec, bfs_types):
        output_types, hint_types = bfs_types
        batch = next(iter(bfs_loader))
        inputs, outputs, hints = batch_to_model_inputs(batch, bfs_spec)
        num_steps = batch.lengths.max().item()

        result = small_model(
            inputs=inputs, outputs=outputs, hints=hints,
            output_types=output_types, hint_types=hint_types,
            num_steps=num_steps,
        )
        result.total_loss.backward()
        # At least some gradients should be non-zero
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in small_model.parameters()
        )
        assert has_grad


# ---------------------------------------------------------------------------
# Training integration test
# ---------------------------------------------------------------------------

class TestTrainingLoop:
    """End-to-end training: loss should decrease over a few steps."""

    def test_loss_decreases(self, bfs_loader, bfs_spec, bfs_types):
        output_types, hint_types = bfs_types
        model = NARModel(
            hidden_dim=32, num_layers=2, num_heads=4,
            processor_type="mpnn", decode_hints=True, encode_hints=True,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        losses = []
        for epoch in range(10):
            epoch_loss = 0.0
            n_batches = 0
            for batch in bfs_loader:
                inputs, outputs, hints = batch_to_model_inputs(batch, bfs_spec)
                num_steps = batch.lengths.max().item()
                optimizer.zero_grad()
                result = model(
                    inputs=inputs, outputs=outputs, hints=hints,
                    output_types=output_types, hint_types=hint_types,
                    num_steps=num_steps,
                )
                result.total_loss.backward()
                optimizer.step()
                epoch_loss += result.total_loss.item()
                n_batches += 1
            losses.append(epoch_loss / n_batches)

        # Loss at end should be lower than at start
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: start={losses[0]:.4f}, end={losses[-1]:.4f}\n"
            f"All losses: {[f'{l:.4f}' for l in losses]}"
        )

    def test_accuracy_above_random(self, bfs_spec, bfs_types):
        """After brief training, pointer accuracy should beat random (1/N)."""
        output_types, hint_types = bfs_types
        model = NARModel(
            hidden_dim=32, num_layers=2, num_heads=4,
            processor_type="mpnn", decode_hints=True, encode_hints=True,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Generate a small fixed dataset (overfit on it)
        from data.clrs_dataset import get_clrs_dataloader
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = get_clrs_dataloader(
                "bfs", batch_size=8, num_samples=8, num_nodes=8,
                edge_probability=0.3, data_dir=tmpdir,
            )
            batch = next(iter(loader))

        inputs, outputs, hints = batch_to_model_inputs(batch, bfs_spec)
        num_steps = batch.lengths.max().item()
        target = outputs["pi"]  # (B, N) indices

        model.train()
        for _ in range(50):
            optimizer.zero_grad()
            result = model(
                inputs=inputs, outputs=outputs, hints=hints,
                output_types=output_types, hint_types=hint_types,
                num_steps=num_steps,
            )
            result.total_loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            result = model(
                inputs=inputs, output_types=output_types,
                hint_types=hint_types, num_steps=num_steps,
            )
        pred = result.predictions["pi"].argmax(dim=-1)  # (B, N)
        accuracy = (pred == target).float().mean().item()
        random_chance = 1.0 / 8  # 8 nodes
        assert accuracy > random_chance, (
            f"Accuracy {accuracy:.3f} not above random chance {random_chance:.3f}"
        )
