"""
Activation Patching for Mechanistic Interpretability.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class PatchingResult:
    """Results from activation patching experiment."""
    component_name: str
    clean_metric: float
    patched_metric: float
    effect: float
    metadata: Dict[str, Any] = None


class ActivationPatchingExperiment:
    """
    Runs activation patching experiments to identify important components.
    """
    
    def __init__(
        self,
        model: nn.Module,
        metric_fn: Callable[[torch.Tensor, torch.Tensor], float],
        ablation_type: str = "mean",
        device: torch.device = None,
    ):
        self.model = model
        self.metric_fn = metric_fn
        self.ablation_type = ablation_type
        self.device = device or next(model.parameters()).device
        
        self._hooks = {}
        self._activations = {}
        self._patches = {}
        self._baseline_cache = {}
        
    def _make_hook(self, name: str):
        """Create a forward hook for a component."""
        def hook(module, inp, out):
            # Store activation
            if isinstance(out, tuple):
                self._activations[name] = out[0].detach().clone()
            else:
                self._activations[name] = out.detach().clone()
            
            # Apply patch if set
            if name in self._patches:
                patch = self._patches[name]
                if isinstance(out, tuple):
                    return (patch,) + out[1:]
                return patch
            return out
        return hook
    
    def _get_module(self, name: str) -> nn.Module:
        """Get module by name."""
        parts = name.split('.')
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            elif part.isdigit() and hasattr(module, 'layers'):
                module = module.layers[int(part)]
            else:
                raise ValueError(f"Cannot find {part} in {name}")
        return module
    
    def register_hooks(self, component_names: List[str]):
        """Register forward hooks on components."""
        for name in component_names:
            try:
                module = self._get_module(name)
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks[name] = hook
            except ValueError as e:
                print(f"Warning: {e}")
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self._hooks.values():
            hook.remove()
        self._hooks.clear()
    
    def collect_baseline_activations(
        self,
        dataloader,
        component_names: List[str],
        num_samples: int = 100,
    ):
        """Collect baseline activations for mean ablation."""
        self.register_hooks(component_names)
        
        all_activations = {name: [] for name in component_names}
        samples_collected = 0
        
        for batch in dataloader:
            if samples_collected >= num_samples:
                break
            
            self._activations.clear()
            
            with torch.no_grad():
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                    self.model(**batch)
                else:
                    self.model(batch.to(self.device))
            
            for name in component_names:
                if name in self._activations:
                    all_activations[name].append(self._activations[name])
            
            # Increment by actual batch size, not 1
            if isinstance(batch, dict):
                batch_sz = next(
                    (v.shape[0] for v in batch.values() if torch.is_tensor(v)), 1
                )
            elif torch.is_tensor(batch):
                batch_sz = batch.shape[0]
            else:
                batch_sz = 1
            samples_collected += batch_sz
        
        # Compute means
        for name, acts in all_activations.items():
            if acts:
                self._baseline_cache[name] = torch.stack(acts).mean(dim=0)
        
        self.remove_hooks()
        print(f"Collected baseline for {len(self._baseline_cache)} components")
    
    def patch_single_component(
        self,
        clean_input: Dict[str, torch.Tensor],
        corrupted_input: Dict[str, torch.Tensor],
        component_name: str,
    ) -> PatchingResult:
        """Patch a single component and measure effect."""
        self._activations.clear()
        self._patches.clear()
        
        # Register hook for this component
        try:
            module = self._get_module(component_name)
            hook = module.register_forward_hook(self._make_hook(component_name))
        except ValueError:
            return PatchingResult(component_name, 0.0, 0.0, 0.0, {'error': 'Cannot find module'})
        
        try:
            # Get clean output
            with torch.no_grad():
                clean_output = self.model(**clean_input)
            
            clean_activation = self._activations.get(component_name)
            
            # Get patch value
            if self.ablation_type == "mean" and component_name in self._baseline_cache:
                patch = self._baseline_cache[component_name]
                if clean_activation is not None and patch.shape[0] != clean_activation.shape[0]:
                    patch = patch.expand(clean_activation.shape[0], *patch.shape[1:])
            elif self.ablation_type == "zero" and clean_activation is not None:
                patch = torch.zeros_like(clean_activation)
            else:
                # Resample from corrupted
                self._activations.clear()
                with torch.no_grad():
                    self.model(**corrupted_input)
                patch = self._activations.get(component_name, torch.zeros(1))
            
            # Run with patch
            self._patches[component_name] = patch
            self._activations.clear()
            
            with torch.no_grad():
                patched_output = self.model(**clean_input)
            
            # Compute effect
            clean_tensor = self._get_output_tensor(clean_output)
            patched_tensor = self._get_output_tensor(patched_output)
            effect = self.metric_fn(clean_tensor, patched_tensor)
            
            return PatchingResult(
                component_name=component_name,
                clean_metric=clean_tensor.mean().item(),
                patched_metric=patched_tensor.mean().item(),
                effect=effect,
            )
        finally:
            hook.remove()
            self._patches.clear()
    
    def _get_output_tensor(self, output) -> torch.Tensor:
        """Extract tensor from model output."""
        if hasattr(output, 'predictions'):
            return list(output.predictions.values())[0]
        elif torch.is_tensor(output):
            return output
        return torch.tensor(0.0)
    
    def run_all_components(
        self,
        clean_batch: Dict[str, torch.Tensor],
        corrupted_batch: Dict[str, torch.Tensor],
        component_names: List[str],
        progress: bool = True,
    ) -> Dict[str, PatchingResult]:
        """Run patching on all components."""
        results = {}
        iterator = tqdm(component_names) if progress else component_names
        
        for name in iterator:
            if progress:
                iterator.set_description(f"Patching {name}")
            results[name] = self.patch_single_component(clean_batch, corrupted_batch, name)
        
        return results
    
    def get_importance_ranking(
        self,
        results: Dict[str, PatchingResult],
    ) -> List[Tuple[str, float]]:
        """Get components ranked by importance."""
        ranking = [(name, r.effect) for name, r in results.items()]
        return sorted(ranking, key=lambda x: abs(x[1]), reverse=True)


def create_corrupted_input(
    clean_input: Dict[str, torch.Tensor],
    corruption_type: str = "shuffle",
    noise_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Create corrupted version of input."""
    corrupted = {}
    
    for key, value in clean_input.items():
        if not torch.is_tensor(value):
            corrupted[key] = value
            continue
        
        if corruption_type == "shuffle":
            perm = torch.randperm(value.shape[0])
            corrupted[key] = value[perm]
        elif corruption_type == "noise":
            noise = torch.randn_like(value) * value.std() * noise_scale
            corrupted[key] = value + noise
        elif corruption_type == "zero":
            corrupted[key] = torch.zeros_like(value)
        else:
            corrupted[key] = value.clone()
    
    return corrupted


class PathPatching:
    """
    Path-level activation patching.

    Instead of patching a single component, patches along a specific
    path through the computation graph to isolate the causal effect
    of information flow along that path.
    """

    def __init__(
        self,
        model: nn.Module,
        metric_fn: Callable[[torch.Tensor, torch.Tensor], float],
        ablation_type: str = "mean",
        device: torch.device = None,
    ):
        self.experiment = ActivationPatchingExperiment(
            model=model,
            metric_fn=metric_fn,
            ablation_type=ablation_type,
            device=device,
        )

    def patch_path(
        self,
        clean_input: Dict[str, torch.Tensor],
        corrupted_input: Dict[str, torch.Tensor],
        path: List[str],
    ) -> List[PatchingResult]:
        """
        Patch along a specific path and measure effects at each step.

        Args:
            clean_input: Clean model input
            corrupted_input: Corrupted model input
            path: List of component names forming the path

        Returns:
            List of PatchingResult for each component in the path
        """
        results = []
        for component in path:
            result = self.experiment.patch_single_component(
                clean_input, corrupted_input, component
            )
            results.append(result)
        return results


def compute_direct_effect(
    model: nn.Module,
    clean_input: Dict[str, torch.Tensor],
    corrupted_input: Dict[str, torch.Tensor],
    component_name: str,
    metric_fn: Callable[[torch.Tensor, torch.Tensor], float],
    ablation_type: str = "mean",
    device: torch.device = None,
) -> PatchingResult:
    """
    Compute the direct causal effect of a component.

    Patches only the specified component while keeping all other
    components at their clean values, measuring the direct (not
    indirect) effect of that component on the output.

    Args:
        model: The model to analyze
        clean_input: Clean model input
        corrupted_input: Corrupted model input
        component_name: Name of the component to patch
        metric_fn: Metric function for measuring effect
        ablation_type: Type of ablation
        device: Device to run on

    Returns:
        PatchingResult with the direct effect
    """
    experiment = ActivationPatchingExperiment(
        model=model,
        metric_fn=metric_fn,
        ablation_type=ablation_type,
        device=device,
    )
    return experiment.patch_single_component(
        clean_input, corrupted_input, component_name
    )
