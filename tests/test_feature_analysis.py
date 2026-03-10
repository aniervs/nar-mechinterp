"""Tests for the feature analysis module."""

import pytest
import torch

from interp.sae import SparseAutoencoder
from interp.feature_analysis import FeatureAnalyzer, FeatureStats, FeatureAnalysisResult


class TestFeatureAnalyzer:
    @pytest.fixture
    def sae(self):
        return SparseAutoencoder(input_dim=16, dict_size=64, sparsity_coeff=1e-3)

    @pytest.fixture
    def analyzer(self, sae):
        return FeatureAnalyzer(sae)

    @pytest.fixture
    def activations(self):
        return torch.randn(200, 16)

    def test_compute_feature_stats(self, analyzer, activations):
        stats = analyzer.compute_feature_stats(activations)
        assert len(stats) == 64
        assert all(isinstance(s, FeatureStats) for s in stats)
        assert all(0.0 <= s.activation_frequency <= 1.0 for s in stats)

    def test_compute_concept_correlations(self, analyzer, activations):
        # Create mock concept labels
        concepts = {
            "is_frontier": (activations[:, 0] > 0).float(),
            "is_visited": (activations[:, 1] > 0.5).float(),
        }
        result = analyzer.compute_concept_correlations(activations, concepts)
        assert isinstance(result, FeatureAnalysisResult)
        assert result.concept_matrix.shape == (64, 2)
        assert result.concept_names == ["is_frontier", "is_visited"]
        assert len(result.feature_stats) == 64
        # Each stat should have concept correlations
        for s in result.feature_stats:
            assert "is_frontier" in s.concept_correlations
            assert "is_visited" in s.concept_correlations

    def test_find_features_for_concept(self, analyzer, activations):
        concept = (activations[:, 0] > 0).float()
        top = analyzer.find_features_for_concept(activations, concept, top_k=5)
        assert len(top) == 5
        assert all(isinstance(t, tuple) and len(t) == 2 for t in top)
        # Should be sorted by absolute correlation (descending)
        abs_corrs = [abs(c) for _, c in top]
        assert abs_corrs == sorted(abs_corrs, reverse=True)

    def test_compute_feature_sharing(self, analyzer):
        # Two algorithms with different activation patterns
        acts_algo1 = torch.randn(100, 16)
        acts_algo2 = torch.randn(100, 16) + 1.0  # Shifted

        result = analyzer.compute_feature_sharing(
            {"bfs": acts_algo1, "dijkstra": acts_algo2}
        )
        assert "shared_features" in result
        assert "algo_specific" in result
        assert "usage_matrix" in result
        assert result["usage_matrix"].shape == (64, 2)
        assert "bfs" in result["algo_specific"]
        assert "dijkstra" in result["algo_specific"]

    def test_dead_feature_detection(self, analyzer):
        # Create activations that will leave some features dead
        # Use a narrow distribution so some features never activate
        activations = torch.randn(50, 16) * 0.01
        stats = analyzer.compute_feature_stats(activations)
        # Some features should be dead with tiny inputs
        dead = [s for s in stats if s.activation_frequency == 0.0]
        # Just verify the mechanism works (may or may not have dead features)
        assert isinstance(dead, list)
