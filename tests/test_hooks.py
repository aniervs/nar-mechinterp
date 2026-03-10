"""Tests for PyTorch hook utilities."""

import pytest
import torch
import torch.nn as nn

from utils.hooks import (
    ActivationStore,
    HookManager,
    ActivationPatcher,
    create_activation_hooks,
)


class TwoLayerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 16)
        self.layer2 = nn.Linear(16, 4)

    def forward(self, inputs=None, **kwargs):
        if inputs is None:
            inputs = torch.randn(2, 8)
        if isinstance(inputs, dict):
            inputs = next(iter(inputs.values()))
        h = torch.relu(self.layer1(inputs))
        return self.layer2(h)


# --- ActivationStore ---

class TestActivationStore:
    def test_clear(self):
        store = ActivationStore()
        store.activations["layer1"].append(torch.randn(2, 4))
        store.gradients["layer1"].append(torch.randn(2, 4))
        store.clear()
        assert len(store.activations) == 0
        assert len(store.gradients) == 0

    def test_get_activation(self):
        store = ActivationStore()
        t = torch.randn(2, 4)
        store.activations["layer1"].append(t)
        result = store.get_activation("layer1", 0)
        assert torch.equal(result, t)

    def test_get_activation_missing(self):
        store = ActivationStore()
        assert store.get_activation("nonexistent") is None

    def test_get_mean_activation(self):
        store = ActivationStore()
        store.activations["l1"].append(torch.ones(2, 4))
        store.activations["l1"].append(torch.ones(2, 4) * 3)
        mean = store.get_mean_activation("l1")
        assert torch.allclose(mean, torch.ones(2, 4) * 2)


# --- HookManager ---

class TestHookManager:
    @pytest.fixture
    def manager(self):
        model = TwoLayerModel()
        return HookManager(model)

    def test_register_forward_hook(self, manager):
        hook_id = manager.register_forward_hook("layer1")
        assert hook_id in manager.hooks
        manager.remove_all_hooks()

    def test_activation_collection(self, manager):
        manager.register_forward_hook("layer1")
        manager.model(inputs=torch.randn(2, 8))
        assert "layer1" in manager.store.activations
        assert len(manager.store.activations["layer1"]) == 1
        assert manager.store.activations["layer1"][0].shape == (2, 16)
        manager.remove_all_hooks()

    def test_register_all_layers(self, manager):
        hook_ids = manager.register_all_layers(layer_types=(nn.Linear,))
        assert len(hook_ids) == 2  # layer1, layer2
        manager.remove_all_hooks()

    def test_set_and_clear_patches(self, manager):
        patch = torch.zeros(2, 16)
        manager.set_patch("layer1", patch)
        assert "layer1" in manager._patch_values
        manager.clear_patches()
        assert len(manager._patch_values) == 0

    def test_patching_context_manager(self, manager):
        manager.register_forward_hook("layer1")
        patch = torch.zeros(2, 16)

        # Without patch
        out1 = manager.model(inputs=torch.randn(2, 8))

        # With patch (layer1 output forced to zeros)
        with manager.patching({"layer1": patch}):
            out2 = manager.model(inputs=torch.randn(2, 8))

        # Patches should be cleared after context
        assert "layer1" not in manager._patch_values
        manager.remove_all_hooks()

    def test_disabled_context_manager(self, manager):
        manager.register_forward_hook("layer1")
        manager.clear_store()
        with manager.disabled():
            manager.model(inputs=torch.randn(2, 8))
        # Activations should NOT be collected when disabled
        assert len(manager.store.activations.get("layer1", [])) == 0
        manager.remove_all_hooks()

    def test_remove_hook(self, manager):
        hook_id = manager.register_forward_hook("layer1")
        manager.remove_hook(hook_id)
        assert hook_id not in manager.hooks

    def test_remove_all_hooks(self, manager):
        manager.register_forward_hook("layer1")
        manager.register_forward_hook("layer2")
        manager.remove_all_hooks()
        assert len(manager.hooks) == 0

    def test_invalid_module_name(self, manager):
        with pytest.raises(ValueError):
            manager.register_forward_hook("nonexistent_module")


# --- ActivationPatcher ---

class TestActivationPatcher:
    @pytest.fixture
    def patcher(self):
        model = TwoLayerModel()
        manager = HookManager(model)
        manager.register_forward_hook("layer1")
        manager.register_forward_hook("layer2")
        return ActivationPatcher(manager, ablation_type="zero")

    def test_get_patch_value_zero(self, patcher):
        act = torch.randn(2, 16)
        patch = patcher.get_patch_value("layer1", act)
        assert torch.all(patch == 0)

    def test_get_patch_value_mean(self):
        model = TwoLayerModel()
        manager = HookManager(model)
        manager.register_forward_hook("layer1")
        patcher = ActivationPatcher(manager, ablation_type="mean")
        act = torch.randn(2, 16)
        # Without baseline, should use batch mean
        patch = patcher.get_patch_value("layer1", act)
        assert patch.shape == act.shape
        manager.remove_all_hooks()

    def test_get_patch_value_resample(self):
        model = TwoLayerModel()
        manager = HookManager(model)
        patcher = ActivationPatcher(manager, ablation_type="resample")
        act = torch.randn(4, 16)
        patch = patcher.get_patch_value("layer1", act)
        assert patch.shape == act.shape
        manager.remove_all_hooks()

    def test_invalid_ablation_type(self):
        model = TwoLayerModel()
        manager = HookManager(model)
        patcher = ActivationPatcher(manager, ablation_type="invalid")
        with pytest.raises(ValueError, match="Unknown ablation type"):
            patcher.get_patch_value("layer1", torch.randn(2, 16))
        manager.remove_all_hooks()

    def test_collect_baseline(self, patcher):
        data = [{"inputs": torch.randn(4, 8)} for _ in range(3)]
        class DictLoader:
            def __iter__(self):
                return iter(data)
        baselines = patcher.collect_baseline(DictLoader(), num_samples=8)
        assert len(baselines) > 0
        patcher.hook_manager.remove_all_hooks()


# --- create_activation_hooks ---

class TestCreateActivationHooks:
    def test_basic(self):
        model = TwoLayerModel()
        manager, hook_ids = create_activation_hooks(model, ["layer1", "layer2"])
        assert len(hook_ids) == 2
        manager.remove_all_hooks()

    def test_with_invalid_name(self):
        model = TwoLayerModel()
        manager, hook_ids = create_activation_hooks(model, ["layer1", "nonexistent"])
        assert len(hook_ids) == 1  # Only layer1 should succeed
        manager.remove_all_hooks()
