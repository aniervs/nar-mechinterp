"""Tests for the concept label extractor."""

import pytest
import torch

from data.clrs_dataset import (
    get_clrs_dataset,
    get_clrs_dataloader,
    get_algorithm_spec,
)
from interp.concept_labels import (
    extract_concept_labels,
    collect_concept_labels,
    ConceptLabels,
)


@pytest.fixture(scope="module")
def bfs_loader(tmp_path_factory):
    data_dir = str(tmp_path_factory.mktemp("clrs_concept"))
    loader = get_clrs_dataloader(
        "bfs", batch_size=4, num_samples=8, num_nodes=6,
        edge_probability=0.3, data_dir=data_dir, shuffle=False,
    )
    return loader


@pytest.fixture(scope="module")
def bfs_spec(tmp_path_factory):
    data_dir = str(tmp_path_factory.mktemp("clrs_spec"))
    ds = get_clrs_dataset(
        "bfs", split="train", num_samples=4, num_nodes=6,
        edge_probability=0.3, data_dir=data_dir,
    )
    return get_algorithm_spec("bfs", dataset=ds)


class TestExtractConceptLabels:
    def test_bfs_basic(self, bfs_loader, bfs_spec):
        batch = next(iter(bfs_loader))
        result = extract_concept_labels(batch, bfs_spec, "bfs")
        assert isinstance(result, ConceptLabels)
        assert result.algorithm == "bfs"
        assert result.num_samples > 0

    def test_bfs_has_expected_concepts(self, bfs_loader, bfs_spec):
        batch = next(iter(bfs_loader))
        result = extract_concept_labels(batch, bfs_spec, "bfs")
        assert "is_visited" in result.labels
        assert "is_frontier" in result.labels
        assert "is_source" in result.labels
        assert "is_unvisited" in result.labels

    def test_labels_are_binary(self, bfs_loader, bfs_spec):
        batch = next(iter(bfs_loader))
        result = extract_concept_labels(batch, bfs_spec, "bfs")
        for name in ["is_visited", "is_frontier", "is_source", "is_unvisited"]:
            vals = result.labels[name]
            assert ((vals == 0) | (vals == 1)).all(), f"{name} is not binary"

    def test_visited_and_unvisited_complement(self, bfs_loader, bfs_spec):
        batch = next(iter(bfs_loader))
        result = extract_concept_labels(batch, bfs_spec, "bfs")
        visited = result.labels["is_visited"]
        unvisited = result.labels["is_unvisited"]
        assert torch.allclose(visited + unvisited, torch.ones_like(visited))

    def test_frontier_is_subset_of_visited(self, bfs_loader, bfs_spec):
        batch = next(iter(bfs_loader))
        result = extract_concept_labels(batch, bfs_spec, "bfs")
        frontier = result.labels["is_frontier"]
        visited = result.labels["is_visited"]
        # Frontier nodes should be visited
        assert (frontier * (1 - visited) == 0).all()

    def test_descriptions_provided(self, bfs_loader, bfs_spec):
        batch = next(iter(bfs_loader))
        result = extract_concept_labels(batch, bfs_spec, "bfs")
        assert len(result.concept_descriptions) > 0
        for name in result.labels:
            assert name in result.concept_descriptions

    def test_all_labels_same_length(self, bfs_loader, bfs_spec):
        batch = next(iter(bfs_loader))
        result = extract_concept_labels(batch, bfs_spec, "bfs")
        lengths = [v.shape[0] for v in result.labels.values()]
        assert len(set(lengths)) == 1, f"Inconsistent lengths: {lengths}"


class TestCollectConceptLabels:
    def test_collect_across_batches(self, bfs_loader, bfs_spec):
        result = collect_concept_labels(bfs_loader, bfs_spec, "bfs")
        assert isinstance(result, ConceptLabels)
        assert result.num_samples > 0
        assert "is_visited" in result.labels
        # Should have more samples than a single batch
        single = extract_concept_labels(next(iter(bfs_loader)), bfs_spec, "bfs")
        assert result.num_samples >= single.num_samples

    def test_collect_with_limit(self, bfs_loader, bfs_spec):
        result = collect_concept_labels(bfs_loader, bfs_spec, "bfs", num_batches=1)
        assert result.num_samples > 0

    def test_generic_algorithm(self, tmp_path_factory):
        """Test that unknown algorithms fall back to generic extraction."""
        data_dir = str(tmp_path_factory.mktemp("clrs_generic"))
        ds = get_clrs_dataset(
            "dijkstra", split="train", num_samples=4, num_nodes=6,
            edge_probability=0.3, data_dir=data_dir,
        )
        spec = get_algorithm_spec("dijkstra", dataset=ds)
        loader = get_clrs_dataloader(
            "dijkstra", batch_size=4, num_samples=4, num_nodes=6,
            edge_probability=0.3, data_dir=data_dir,
        )
        result = collect_concept_labels(loader, spec, "dijkstra")
        assert "is_settled" in result.labels or "is_source" in result.labels
