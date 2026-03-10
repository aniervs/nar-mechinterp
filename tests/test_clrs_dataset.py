"""Tests for the CLRS dataset loader."""

import pytest
import torch

from data.clrs_dataset import (
    get_clrs_dataset,
    get_clrs_dataloader,
    get_algorithm_spec,
    pyg_to_dense,
    CLRSBatch,
    AVAILABLE_ALGORITHMS,
)


@pytest.fixture(scope="module")
def bfs_dataset(tmp_path_factory):
    """Create a small BFS dataset (shared across tests in this module)."""
    data_dir = str(tmp_path_factory.mktemp("clrs"))
    return get_clrs_dataset(
        "bfs", split="train", num_samples=10, num_nodes=8,
        edge_probability=0.3, data_dir=data_dir,
    )


class TestGetClrsDataset:
    def test_load_bfs(self, bfs_dataset):
        assert len(bfs_dataset) == 10

    def test_item_has_edge_index(self, bfs_dataset):
        item = bfs_dataset[0]
        assert hasattr(item, 'edge_index')
        assert item.edge_index.dim() == 2
        assert item.edge_index.shape[0] == 2

    def test_item_has_algorithm_fields(self, bfs_dataset):
        item = bfs_dataset[0]
        assert hasattr(item, 'inputs')
        assert hasattr(item, 'outputs')
        assert hasattr(item, 'hints')
        assert 's' in item.inputs  # source node
        assert 'pi' in item.outputs  # predecessor pointer

    def test_invalid_algorithm(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            get_clrs_dataset("nonexistent_algo", data_dir=str(tmp_path))


class TestGetClrsDataloader:
    def test_batching(self, bfs_dataset, tmp_path_factory):
        data_dir = str(tmp_path_factory.mktemp("clrs_loader"))
        loader = get_clrs_dataloader(
            "bfs", batch_size=4, num_samples=10, num_nodes=8,
            edge_probability=0.3, data_dir=data_dir,
        )
        batch = next(iter(loader))
        assert isinstance(batch, CLRSBatch)
        assert batch.num_graphs == 4
        assert batch.edge_index.dim() == 2
        assert batch.batch.shape[0] == 4 * 8  # 4 graphs * 8 nodes

    def test_batch_has_lengths(self, tmp_path_factory):
        data_dir = str(tmp_path_factory.mktemp("clrs_len"))
        loader = get_clrs_dataloader(
            "bfs", batch_size=3, num_samples=6, num_nodes=8,
            data_dir=data_dir,
        )
        batch = next(iter(loader))
        assert batch.lengths.shape == (3,)
        assert (batch.lengths > 0).all()

    def test_batch_node_data_padded(self, tmp_path_factory):
        data_dir = str(tmp_path_factory.mktemp("clrs_pad"))
        loader = get_clrs_dataloader(
            "bfs", batch_size=4, num_samples=8, num_nodes=8,
            data_dir=data_dir,
        )
        batch = next(iter(loader))
        # Hint fields should be padded to max steps
        assert 'reach_h' in batch.node_data
        reach_h = batch.node_data['reach_h']
        assert reach_h.dim() == 2  # (total_nodes, max_steps)
        assert reach_h.shape[0] == 4 * 8


class TestGetAlgorithmSpec:
    def test_from_hardcoded(self):
        spec = get_algorithm_spec("bfs")
        assert "pos" in spec.input_fields or "s" in spec.input_fields
        assert len(spec.output_fields) > 0

    def test_from_dataset(self, bfs_dataset):
        spec = get_algorithm_spec("bfs", dataset=bfs_dataset)
        assert "s" in spec.input_fields
        assert "pi" in spec.output_fields
        assert "reach_h" in spec.hint_fields

    def test_unknown_algorithm(self):
        spec = get_algorithm_spec("unknown_algo")
        assert spec.input_fields == []
        assert spec.output_fields == []


class TestPygToDense:
    def test_basic_conversion(self, bfs_dataset):
        item = bfs_dataset[0]
        dense = pyg_to_dense(item)
        assert 'adjacency' in dense
        assert 'node_features' in dense
        assert dense['adjacency'].shape == (8, 8)
        assert dense['node_features'].shape[0] == 8

    def test_adjacency_is_binary(self, bfs_dataset):
        item = bfs_dataset[0]
        dense = pyg_to_dense(item)
        adj = dense['adjacency']
        assert ((adj == 0) | (adj == 1)).all()


class TestCLRSBatch:
    def test_to_device(self, tmp_path_factory):
        data_dir = str(tmp_path_factory.mktemp("clrs_dev"))
        loader = get_clrs_dataloader(
            "bfs", batch_size=2, num_samples=4, num_nodes=8,
            data_dir=data_dir,
        )
        batch = next(iter(loader))
        # Just test .to() doesn't error (CPU to CPU)
        batch2 = batch.to(torch.device("cpu"))
        assert batch2.edge_index.device.type == "cpu"
