"""Tests for the ActivationCollector and make_activation_dataloader."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from interp.activation_collector import ActivationCollector, make_activation_dataloader
from models.nar_model import NARModel


@pytest.fixture
def model():
    """Small NAR model for testing."""
    return NARModel(
        hidden_dim=32,
        num_layers=1,
        num_heads=4,
        processor_type="mpnn",
    )


@pytest.fixture
def simple_dataloader():
    """Dataloader yielding dict batches with adjacency and node_features."""
    num_samples = 8
    num_nodes = 6
    adj = (torch.rand(num_samples, num_nodes, num_nodes) > 0.5).float()
    node_features = torch.randn(num_samples, num_nodes, 8)
    # Split into batches of 4
    dataset = TensorDataset(adj, node_features)
    dl = DataLoader(dataset, batch_size=4)

    class DictWrapper:
        """Wraps TensorDataset batches into dicts."""
        def __init__(self, loader):
            self._loader = loader

        def __iter__(self):
            for adj_b, nf_b in self._loader:
                yield {"adjacency": adj_b, "node_features": nf_b}

        def __len__(self):
            return len(self._loader)

    return DictWrapper(dl)


class TestActivationCollector:
    def test_collect_returns_2d_tensor(self, model, simple_dataloader):
        collector = ActivationCollector(model)
        activations = collector.collect(simple_dataloader, num_steps=1)
        assert activations.dim() == 2
        assert activations.shape[1] == model.hidden_dim

    def test_collect_num_batches_limit(self, model, simple_dataloader):
        collector = ActivationCollector(model)
        acts_all = collector.collect(simple_dataloader, num_steps=1)
        acts_one = collector.collect(simple_dataloader, num_batches=1, num_steps=1)
        assert acts_one.shape[0] < acts_all.shape[0]

    def test_collect_multiple_steps(self, model, simple_dataloader):
        collector = ActivationCollector(model)
        acts_1 = collector.collect(simple_dataloader, num_steps=1)
        acts_2 = collector.collect(simple_dataloader, num_steps=2)
        # 2 steps should produce ~2x the activations
        assert acts_2.shape[0] == acts_1.shape[0] * 2

    def test_collect_empty_raises(self, model):
        """Empty dataloader should raise RuntimeError."""
        empty_loader = iter([])
        collector = ActivationCollector(model)
        with pytest.raises(RuntimeError, match="No activations collected"):
            collector.collect(empty_loader, num_steps=1)

    def test_collect_streaming(self, model, simple_dataloader):
        collector = ActivationCollector(model)
        chunks = list(collector.collect_streaming(simple_dataloader, num_steps=1))
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.dim() == 2
            assert chunk.shape[1] == model.hidden_dim

    def test_collect_streaming_matches_collect(self, model, simple_dataloader):
        collector = ActivationCollector(model)
        acts = collector.collect(simple_dataloader, num_steps=1)
        chunks = list(collector.collect_streaming(simple_dataloader, num_steps=1))
        streamed = torch.cat(chunks, dim=0)
        assert acts.shape == streamed.shape
        assert torch.allclose(acts, streamed)

    def test_activations_on_cpu(self, model, simple_dataloader):
        collector = ActivationCollector(model)
        acts = collector.collect(simple_dataloader, num_steps=1)
        assert acts.device == torch.device("cpu")


class TestMakeActivationDataloader:
    def test_returns_dataloader(self):
        acts = torch.randn(100, 32)
        dl = make_activation_dataloader(acts, batch_size=16)
        assert isinstance(dl, DataLoader)

    def test_batch_shape(self):
        acts = torch.randn(100, 32)
        dl = make_activation_dataloader(acts, batch_size=16)
        (batch,) = next(iter(dl))
        assert batch.shape == (16, 32)

    def test_all_data_yielded(self):
        acts = torch.randn(100, 32)
        dl = make_activation_dataloader(acts, batch_size=16, shuffle=False)
        all_batches = torch.cat([b for (b,) in dl], dim=0)
        assert all_batches.shape == (100, 32)
        assert torch.allclose(all_batches, acts)

    def test_shuffle_flag(self):
        acts = torch.randn(100, 32)
        dl_no_shuffle = make_activation_dataloader(acts, batch_size=100, shuffle=False)
        (batch,) = next(iter(dl_no_shuffle))
        assert torch.allclose(batch, acts)
