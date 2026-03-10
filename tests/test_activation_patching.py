"""Tests for activation patching."""

import pytest
import torch
import torch.nn as nn

from interp.activation_patching import (
    ActivationPatchingExperiment,
    PatchingResult,
    PathPatching,
    compute_direct_effect,
    create_corrupted_input,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 16)
        self.layer2 = nn.Linear(16, 8)

    def forward(self, **kwargs):
        x = kwargs.get("inputs", torch.randn(2, 8))
        if isinstance(x, dict):
            x = next(iter(x.values()))
        h = torch.relu(self.layer1(x))
        return self.layer2(h)


class TestPatchingResult:
    def test_creation(self):
        r = PatchingResult(
            component_name="layer1",
            clean_metric=1.0,
            patched_metric=0.5,
            effect=0.5,
        )
        assert r.effect == 0.5
        assert r.metadata is None


class TestCreateCorruptedInput:
    def test_shuffle(self):
        clean = {"x": torch.randn(4, 8), "label": "foo"}
        corrupted = create_corrupted_input(clean, "shuffle")
        assert corrupted["x"].shape == clean["x"].shape
        assert corrupted["label"] == "foo"  # Non-tensors passed through

    def test_noise(self):
        clean = {"x": torch.randn(4, 8)}
        corrupted = create_corrupted_input(clean, "noise", noise_scale=0.1)
        assert corrupted["x"].shape == clean["x"].shape
        # Should be different from original
        assert not torch.allclose(corrupted["x"], clean["x"])

    def test_zero(self):
        clean = {"x": torch.ones(4, 8)}
        corrupted = create_corrupted_input(clean, "zero")
        assert torch.all(corrupted["x"] == 0)

    def test_unknown_type_clones(self):
        clean = {"x": torch.randn(4, 8)}
        corrupted = create_corrupted_input(clean, "unknown_type")
        assert torch.allclose(corrupted["x"], clean["x"])


class TestActivationPatchingExperiment:
    @pytest.fixture
    def experiment(self):
        model = SimpleModel()
        metric_fn = lambda clean, patched: (clean - patched).abs().mean().item()
        return ActivationPatchingExperiment(model, metric_fn)

    def test_init(self, experiment):
        assert experiment.ablation_type == "mean"

    def test_patch_single_component(self, experiment):
        clean = {"inputs": torch.randn(2, 8)}
        corrupted = create_corrupted_input(clean, "shuffle")
        result = experiment.patch_single_component(clean, corrupted, "layer1")
        assert isinstance(result, PatchingResult)
        assert result.component_name == "layer1"

    def test_patch_nonexistent_component(self, experiment):
        clean = {"inputs": torch.randn(2, 8)}
        corrupted = create_corrupted_input(clean, "shuffle")
        result = experiment.patch_single_component(clean, corrupted, "nonexistent.module.xyz")
        assert result.effect == 0.0
        assert result.metadata == {"error": "Cannot find module"}

    def test_run_all_components(self, experiment):
        clean = {"inputs": torch.randn(2, 8)}
        corrupted = create_corrupted_input(clean, "shuffle")
        results = experiment.run_all_components(
            clean, corrupted, ["layer1", "layer2"], progress=False
        )
        assert "layer1" in results
        assert "layer2" in results

    def test_get_importance_ranking(self, experiment):
        clean = {"inputs": torch.randn(2, 8)}
        corrupted = create_corrupted_input(clean, "shuffle")
        results = experiment.run_all_components(
            clean, corrupted, ["layer1", "layer2"], progress=False
        )
        ranking = experiment.get_importance_ranking(results)
        assert len(ranking) == 2
        # Should be sorted by absolute effect, descending
        assert abs(ranking[0][1]) >= abs(ranking[1][1])

    def test_collect_baseline(self, experiment):
        data = [{"inputs": torch.randn(4, 8)} for _ in range(3)]
        class DictLoader:
            def __iter__(self):
                return iter(data)
        experiment.register_hooks(["layer1"])
        experiment.collect_baseline_activations(DictLoader(), ["layer1"], num_samples=8)
        assert "layer1" in experiment._baseline_cache
        experiment.remove_hooks()


class TestPathPatching:
    def test_patch_path(self):
        model = SimpleModel()
        metric_fn = lambda clean, patched: (clean - patched).abs().mean().item()
        pp = PathPatching(model, metric_fn)
        clean = {"inputs": torch.randn(2, 8)}
        corrupted = create_corrupted_input(clean, "shuffle")
        results = pp.patch_path(clean, corrupted, ["layer1", "layer2"])
        assert len(results) == 2
        assert all(isinstance(r, PatchingResult) for r in results)


class TestComputeDirectEffect:
    def test_basic(self):
        model = SimpleModel()
        metric_fn = lambda clean, patched: (clean - patched).abs().mean().item()
        clean = {"inputs": torch.randn(2, 8)}
        corrupted = create_corrupted_input(clean, "shuffle")
        result = compute_direct_effect(
            model, clean, corrupted, "layer1", metric_fn
        )
        assert isinstance(result, PatchingResult)
        assert result.component_name == "layer1"
