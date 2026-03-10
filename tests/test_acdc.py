"""Tests for ACDC circuit discovery."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from interp.acdc import ACDC, ACDCConfig, ACDCResult, ACDCForNAR, run_acdc_experiment, run_acdc_multi_algorithm
from interp.circuit import Circuit


class SimpleModel(nn.Module):
    """Minimal model for testing ACDC."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 16)
        self.linear2 = nn.Linear(16, 8)

    def forward(self, **kwargs):
        x = kwargs.get("inputs", torch.randn(2, 8))
        if isinstance(x, dict):
            x = next(iter(x.values()))
        if x.dim() > 2:
            x = x.view(x.shape[0], -1)[:, :8]
        h = torch.relu(self.linear1(x))
        return self.linear2(h)


def make_dataloader(batch_size=4, num_batches=3):
    """Create a simple dict-based dataloader."""
    data = [{"inputs": torch.randn(batch_size, 8)} for _ in range(num_batches)]
    class DictLoader:
        def __init__(self, data):
            self._data = data
        def __iter__(self):
            return iter(self._data)
        def __len__(self):
            return len(self._data)
    return DictLoader(data)


class TestACDCConfig:
    def test_defaults(self):
        config = ACDCConfig()
        assert config.threshold == 0.01
        assert config.metric == "kl_divergence"
        assert config.ablation_type == "mean"
        assert config.verbose is True

    def test_custom(self):
        config = ACDCConfig(threshold=0.05, metric="mse", verbose=False)
        assert config.threshold == 0.05
        assert config.metric == "mse"


class TestACDCResult:
    def test_creation(self):
        c = Circuit(name="test")
        result = ACDCResult(
            circuit=c,
            pruned_edges=[("a", "b", 0.01)],
            iterations=5,
            final_fidelity=0.95,
            history=[{"iteration": 1}],
        )
        assert result.final_fidelity == 0.95
        assert len(result.pruned_edges) == 1
        assert len(result.history) == 1


class TestACDC:
    @pytest.fixture
    def acdc_setup(self):
        model = SimpleModel()
        config = ACDCConfig(
            threshold=0.5,
            max_iterations=2,
            num_samples=4,
            verbose=False,
        )
        return model, config

    def test_init(self, acdc_setup):
        model, config = acdc_setup
        acdc = ACDC(model, config)
        assert acdc.model is model
        assert acdc.config is config

    def test_build_component_graph(self, acdc_setup):
        model, config = acdc_setup
        acdc = ACDC(model, config)
        circuit, names = acdc._build_component_graph()
        assert circuit.num_nodes >= 3  # input + leaves + output
        assert circuit.num_edges > 0
        assert len(names) > 0

    def test_get_metric_fn(self, acdc_setup):
        model, config = acdc_setup
        acdc = ACDC(model, config)
        fn = acdc._get_metric_fn("mse")
        t1 = torch.randn(4, 10)
        t2 = torch.randn(4, 10)
        result = fn(t1, t2)
        assert isinstance(result, float)

    def test_save_result(self, acdc_setup, tmp_path):
        model, config = acdc_setup
        acdc = ACDC(model, config)
        c = Circuit(name="test")
        c.add_node("a", "input")
        result = ACDCResult(
            circuit=c,
            pruned_edges=[],
            iterations=0,
            final_fidelity=1.0,
        )
        save_path = str(tmp_path / "result.json")
        acdc.save_result(result, save_path)
        assert (tmp_path / "result.json").exists()
        assert (tmp_path / "result.meta.json").exists()


class TestRunACDCMultiAlgorithm:
    def test_basic(self):
        """Test that run_acdc_multi_algorithm handles missing dataloaders gracefully."""
        model = SimpleModel()
        config = ACDCConfig(verbose=False, max_iterations=1)
        # No dataloaders match — should skip all
        results = run_acdc_multi_algorithm(
            model=model,
            dataloaders={},
            algorithms=["bfs", "dfs"],
            config=config,
        )
        assert results == {}
