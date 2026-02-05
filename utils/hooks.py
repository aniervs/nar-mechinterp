"""
PyTorch Hook Utilities for Activation Collection and Patching.

Provides flexible hooks for:
- Recording activations at specific layers
- Patching activations with corrupted/clean values
- Gradient-based importance scoring
"""

import torch
import torch.nn as nn
from typing import Dict, List, Callable, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager


@dataclass
class ActivationStore:
    """Storage for collected activations."""
    activations: Dict[str, List[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))
    gradients: Dict[str, List[torch.Tensor]] = field(default_factory=lambda: defaultdict(list))
    
    def clear(self):
        """Clear all stored activations and gradients."""
        self.activations.clear()
        self.gradients.clear()
    
    def get_activation(self, name: str, idx: int = -1) -> Optional[torch.Tensor]:
        """Get activation by name and index."""
        if name in self.activations and len(self.activations[name]) > abs(idx):
            return self.activations[name][idx]
        return None
    
    def get_mean_activation(self, name: str) -> Optional[torch.Tensor]:
        """Get mean activation across all stored samples."""
        if name in self.activations and len(self.activations[name]) > 0:
            return torch.stack(self.activations[name]).mean(dim=0)
        return None


class HookManager:
    """
    Manages forward and backward hooks for PyTorch modules.
    
    Features:
    - Named hook registration
    - Activation collection
    - Activation patching
    - Gradient collection
    - Context managers for temporary hooks
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: Dict[str, Any] = {}
        self.store = ActivationStore()
        self._patch_values: Dict[str, torch.Tensor] = {}
        self._enabled = True
        
    def _get_module_by_name(self, name: str) -> nn.Module:
        """Get a module by its dotted name path."""
        parts = name.split('.')
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif hasattr(module, 'layers') and part.isdigit():
                module = module.layers[int(part)]
            else:
                raise ValueError(f"Cannot find module part '{part}' in {name}")
        return module
    
    def register_forward_hook(
        self,
        module_name: str,
        hook_fn: Optional[Callable] = None,
        store_activations: bool = True,
    ) -> str:
        """
        Register a forward hook on a named module.
        
        Args:
            module_name: Dotted path to module (e.g., "processor.layers.0.attention")
            hook_fn: Optional custom hook function
            store_activations: Whether to store activations
            
        Returns:
            Hook identifier
        """
        module = self._get_module_by_name(module_name)
        
        def default_hook(mod, inp, out):
            if not self._enabled:
                return out
            
            # Store activation
            if store_activations:
                if isinstance(out, tuple):
                    self.store.activations[module_name].append(out[0].detach().clone())
                else:
                    self.store.activations[module_name].append(out.detach().clone())
            
            # Apply custom hook
            if hook_fn is not None:
                return hook_fn(mod, inp, out)
            
            # Apply patching if configured
            if module_name in self._patch_values:
                patch = self._patch_values[module_name]
                if isinstance(out, tuple):
                    return (patch,) + out[1:]
                return patch
            
            return out
        
        handle = module.register_forward_hook(default_hook)
        hook_id = f"fwd_{module_name}_{len(self.hooks)}"
        self.hooks[hook_id] = handle
        return hook_id
    
    def register_backward_hook(
        self,
        module_name: str,
        hook_fn: Optional[Callable] = None,
        store_gradients: bool = True,
    ) -> str:
        """
        Register a backward hook on a named module.
        
        Args:
            module_name: Dotted path to module
            hook_fn: Optional custom hook function
            store_gradients: Whether to store gradients
            
        Returns:
            Hook identifier
        """
        module = self._get_module_by_name(module_name)
        
        def default_hook(mod, grad_input, grad_output):
            if not self._enabled:
                return
            
            if store_gradients:
                if isinstance(grad_output, tuple):
                    self.store.gradients[module_name].append(grad_output[0].detach().clone())
                else:
                    self.store.gradients[module_name].append(grad_output.detach().clone())
            
            if hook_fn is not None:
                return hook_fn(mod, grad_input, grad_output)
        
        handle = module.register_full_backward_hook(default_hook)
        hook_id = f"bwd_{module_name}_{len(self.hooks)}"
        self.hooks[hook_id] = handle
        return hook_id
    
    def register_all_layers(
        self,
        layer_types: Tuple[type, ...] = (nn.Linear, nn.MultiheadAttention),
        store_activations: bool = True,
        store_gradients: bool = False,
    ) -> List[str]:
        """
        Register hooks on all layers of specified types.
        
        Args:
            layer_types: Tuple of layer types to hook
            store_activations: Whether to store forward activations
            store_gradients: Whether to store backward gradients
            
        Returns:
            List of hook identifiers
        """
        hook_ids = []
        
        for name, module in self.model.named_modules():
            if isinstance(module, layer_types):
                hook_id = self.register_forward_hook(name, store_activations=store_activations)
                hook_ids.append(hook_id)
                
                if store_gradients:
                    hook_id = self.register_backward_hook(name, store_gradients=True)
                    hook_ids.append(hook_id)
        
        return hook_ids
    
    def set_patch(self, module_name: str, value: torch.Tensor):
        """Set a patch value for a module (will replace its output)."""
        self._patch_values[module_name] = value
    
    def clear_patches(self):
        """Clear all patch values."""
        self._patch_values.clear()
    
    def remove_hook(self, hook_id: str):
        """Remove a specific hook."""
        if hook_id in self.hooks:
            self.hooks[hook_id].remove()
            del self.hooks[hook_id]
    
    def remove_all_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks.values():
            handle.remove()
        self.hooks.clear()
    
    def clear_store(self):
        """Clear stored activations and gradients."""
        self.store.clear()
    
    @contextmanager
    def disabled(self):
        """Context manager to temporarily disable hooks."""
        self._enabled = False
        try:
            yield
        finally:
            self._enabled = True
    
    @contextmanager
    def patching(self, patches: Dict[str, torch.Tensor]):
        """Context manager for temporary activation patching."""
        old_patches = self._patch_values.copy()
        self._patch_values.update(patches)
        try:
            yield
        finally:
            self._patch_values = old_patches
    
    def __del__(self):
        """Clean up hooks on deletion."""
        self.remove_all_hooks()


class ActivationPatcher:
    """
    Utility class for activation patching experiments.
    
    Supports different patching strategies:
    - Zero ablation
    - Mean ablation
    - Resample ablation
    - Noise injection
    """
    
    def __init__(
        self,
        hook_manager: HookManager,
        ablation_type: str = "mean",
    ):
        self.hook_manager = hook_manager
        self.ablation_type = ablation_type
        self._baseline_activations: Dict[str, torch.Tensor] = {}
    
    def collect_baseline(
        self,
        dataloader,
        num_samples: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """
        Collect baseline activations for mean ablation.
        
        Args:
            dataloader: DataLoader with training data
            num_samples: Number of samples to use for baseline
            
        Returns:
            Dict of mean activations per module
        """
        self.hook_manager.clear_store()
        
        samples_collected = 0
        for batch in dataloader:
            if samples_collected >= num_samples:
                break
            
            with torch.no_grad():
                # Run forward pass to collect activations
                _ = self.hook_manager.model(batch)
            
            samples_collected += len(batch)
        
        # Compute means
        self._baseline_activations = {}
        for name, activations in self.hook_manager.store.activations.items():
            if len(activations) > 0:
                self._baseline_activations[name] = torch.stack(activations).mean(dim=0)
        
        return self._baseline_activations
    
    def get_patch_value(
        self,
        module_name: str,
        clean_activation: torch.Tensor,
        corrupted_activation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get the patch value based on ablation type.
        
        Args:
            module_name: Name of the module
            clean_activation: Clean activation to potentially patch
            corrupted_activation: Optional corrupted activation for resample
            
        Returns:
            Patch tensor
        """
        if self.ablation_type == "zero":
            return torch.zeros_like(clean_activation)
        
        elif self.ablation_type == "mean":
            if module_name in self._baseline_activations:
                baseline = self._baseline_activations[module_name]
                # Broadcast to match batch size if needed
                if baseline.shape[0] != clean_activation.shape[0]:
                    baseline = baseline.expand(clean_activation.shape[0], *baseline.shape[1:])
                return baseline
            return clean_activation.mean(dim=0, keepdim=True).expand_as(clean_activation)
        
        elif self.ablation_type == "resample":
            if corrupted_activation is not None:
                return corrupted_activation
            # Shuffle within batch
            perm = torch.randperm(clean_activation.shape[0])
            return clean_activation[perm]
        
        elif self.ablation_type == "noise":
            noise = torch.randn_like(clean_activation) * clean_activation.std()
            return clean_activation + noise
        
        else:
            raise ValueError(f"Unknown ablation type: {self.ablation_type}")
    
    def patch_and_run(
        self,
        inputs: Dict[str, torch.Tensor],
        modules_to_patch: List[str],
        corrupted_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """
        Run model with specific modules patched.
        
        Args:
            inputs: Clean inputs
            modules_to_patch: List of module names to patch
            corrupted_inputs: Optional corrupted inputs for resample ablation
            
        Returns:
            output: Model output with patching
            patched_activations: Dict of patch values used
        """
        patched_activations = {}
        
        # First run with clean inputs to get activations
        self.hook_manager.clear_store()
        with torch.no_grad():
            _ = self.hook_manager.model(**inputs)
        
        clean_activations = {
            name: acts[-1] for name, acts in self.hook_manager.store.activations.items()
        }
        
        # Get corrupted activations if using resample
        corrupted_activations = {}
        if corrupted_inputs is not None and self.ablation_type == "resample":
            self.hook_manager.clear_store()
            with torch.no_grad():
                _ = self.hook_manager.model(**corrupted_inputs)
            corrupted_activations = {
                name: acts[-1] for name, acts in self.hook_manager.store.activations.items()
            }
        
        # Compute patch values
        patches = {}
        for module_name in modules_to_patch:
            if module_name in clean_activations:
                patch = self.get_patch_value(
                    module_name,
                    clean_activations[module_name],
                    corrupted_activations.get(module_name),
                )
                patches[module_name] = patch
                patched_activations[module_name] = patch
        
        # Run with patches
        self.hook_manager.clear_store()
        with self.hook_manager.patching(patches):
            output = self.hook_manager.model(**inputs)
        
        return output, patched_activations


def create_activation_hooks(
    model: nn.Module,
    component_names: List[str],
) -> Tuple[HookManager, List[str]]:
    """
    Convenience function to set up activation hooks for interpretability.
    
    Args:
        model: PyTorch model
        component_names: List of component names to hook
        
    Returns:
        hook_manager: Configured HookManager
        hook_ids: List of hook identifiers
    """
    manager = HookManager(model)
    hook_ids = []
    
    for name in component_names:
        try:
            hook_id = manager.register_forward_hook(name, store_activations=True)
            hook_ids.append(hook_id)
        except ValueError as e:
            print(f"Warning: Could not hook {name}: {e}")
    
    return manager, hook_ids
