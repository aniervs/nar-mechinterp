"""Tests for interpretability metrics."""

import pytest
import torch

from interp.metrics import (
    kl_divergence,
    edge_importance_score,
    compute_minimality,
    compute_specificity,
)
from interp.circuit import Circuit


class TestKLDivergence:
    def test_identical_tensors(self):
        t = torch.randn(4, 10)
        kl = kl_divergence(t, t)
        assert kl.item() == pytest.approx(0.0, abs=1e-5)

    def test_different_tensors(self):
        p = torch.randn(4, 10)
        q = torch.randn(4, 10)
        kl = kl_divergence(p, q)
        assert kl.item() >= 0.0

    def test_returns_scalar(self):
        kl = kl_divergence(torch.randn(2, 5), torch.randn(2, 5))
        assert kl.dim() == 0


class TestEdgeImportanceScore:
    def test_identical_outputs(self):
        t = torch.randn(4, 10)
        score = edge_importance_score(t, t)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_different_outputs(self):
        clean = torch.zeros(4, 10)
        patched = torch.ones(4, 10)
        score = edge_importance_score(clean, patched)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_returns_float(self):
        score = edge_importance_score(torch.randn(2, 5), torch.randn(2, 5))
        assert isinstance(score, float)


class TestComputeMinimality:
    def test_basic(self):
        c = Circuit()
        c.add_node("a", "input", layer=0)
        c.add_node("b", "mlp", layer=1)
        c.add_node("c", "output", layer=2)
        c.add_edge("a", "b")
        c.add_edge("b", "c")
        metrics = compute_minimality(c)
        assert metrics["num_nodes"] == 3
        assert metrics["num_edges"] == 2
        assert metrics["depth"] == 3
        assert 0.0 <= metrics["density"] <= 1.0

    def test_single_node(self):
        c = Circuit()
        c.add_node("a", "input")
        metrics = compute_minimality(c)
        assert metrics["num_nodes"] == 1
        assert metrics["num_edges"] == 0


class TestComputeSpecificity:
    def test_no_others(self):
        c = Circuit()
        c.add_node("a", "input")
        metrics = compute_specificity(c, [])
        assert metrics["uniqueness"] == 1.0

    def test_identical_others(self):
        c = Circuit()
        c.add_node("a", "input")
        c.add_node("b", "output")
        c.add_edge("a", "b")
        metrics = compute_specificity(c, [c])
        assert metrics["avg_similarity"] == pytest.approx(1.0)
        assert metrics["uniqueness"] == pytest.approx(0.0)
