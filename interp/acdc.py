"""
ACDC: Automatic Circuit DisCovery

Implementation of the ACDC algorithm from "Towards Automated Circuit Discovery 
for Mechanistic Interpretability" (Conmy et al., 2023).

ACDC iteratively identifies the minimal circuit sufficient for a task by:
1. Starting with the full computational graph
2. Iteratively testing edges via activation patching
3. Removing edges that don't significantly affect the output
4. Returning the minimal sufficient circuit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable, Set, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import json
from pathlib import Path

from .circuit import Circuit, CircuitNode, CircuitEdge, create_full_circuit_from_model
from .activation_patching import ActivationPatchingExperiment, create_corrupted_input
from .metrics import kl_divergence, edge_importance_score


@dataclass
class ACDCConfig:
    """Configuration for ACDC algorithm."""
    threshold: float = 0.01  # Edge pruning threshold
    metric: str = "kl_divergence"  # Metric for measuring importance
    ablation_type: str = "mean"  # Type of ablation (mean, zero, resample)
    max_iterations: int = 100  # Maximum pruning iterations
    num_samples: int = 100  # Samples for baseline computation
    batch_size: int = 16  # Batch size for patching
    early_stop_threshold: float = 0.1  # Stop if fidelity drops below this
    verbose: bool = True


@dataclass
class ACDCResult:
    """Results from ACDC circuit discovery."""
    circuit: Circuit
    pruned_edges: List[Tuple[str, str, float]]  # (src, tgt, importance)
    iterations: int
    final_fidelity: float
    history: List[Dict] = field(default_factory=list)


class ACDC:
    """
    ACDC: Automatic Circuit DisCovery
    
    Discovers minimal circuits in neural networks via iterative edge pruning.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ACDCConfig = None,
        device: torch.device = None,
    ):
        """
        Initialize ACDC.
        
        Args:
            model: The neural network to analyze
            config: ACDC configuration
            device: Device to run on
        """
        self.model = model
        self.config = config or ACDCConfig()
        self.device = device or next(model.parameters()).device
        
        # Set up metric function
        self.metric_fn = self._get_metric_fn(self.config.metric)
        
        # Initialize patching experiment
        self.patcher = ActivationPatchingExperiment(
            model=model,
            metric_fn=self.metric_fn,
            ablation_type=self.config.ablation_type,
            device=self.device,
        )
        
        # Circuit representation
        self.full_circuit: Optional[Circuit] = None
        self.current_circuit: Optional[Circuit] = None
        
    def _get_metric_fn(self, metric: str) -> Callable:
        """Get metric function by name."""
        if metric == "kl_divergence":
            return lambda clean, patched: kl_divergence(clean, patched).item()
        elif metric == "mse":
            return lambda clean, patched: F.mse_loss(clean, patched).item()
        elif metric == "logit_diff":
            return lambda clean, patched: (clean - patched).abs().mean().item()
        else:
            return lambda clean, patched: (clean - patched).abs().mean().item()
    
    def _build_component_graph(self) -> Tuple[Circuit, List[str]]:
        """
        Build the computational graph of model components.
        
        Returns:
            circuit: Full circuit representation
            component_names: List of hookable component names
        """
        circuit = Circuit(name="full_model")
        component_names = []
        
        # Add input node
        circuit.add_node("input", "input", layer=0, component="input")
        
        # Traverse model to find components
        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0:
                continue  # Skip container modules
            
            # Determine component type and layer
            if 'attention' in name.lower():
                node_type = "attention"
            elif 'mlp' in name.lower() or 'linear' in name.lower():
                node_type = "mlp"
            elif 'message' in name.lower():
                node_type = "message_passing"
            elif 'node' in name.lower():
                node_type = "node_update"
            elif 'edge' in name.lower():
                node_type = "edge_update"
            else:
                node_type = "unknown"
            
            # Extract layer number from name
            layer = 0
            for part in name.split('.'):
                if part.isdigit():
                    layer = int(part) + 1
                    break
            
            circuit.add_node(name, node_type, layer=layer, component=name)
            component_names.append(name)
        
        # Add output node
        circuit.add_node("output", "output", layer=100, component="output")
        
        # Build edges based on layer structure
        nodes_by_layer = {}
        for node in circuit.nodes.values():
            layer = node.layer
            if layer not in nodes_by_layer:
                nodes_by_layer[layer] = []
            nodes_by_layer[layer].append(node.id)
        
        sorted_layers = sorted(nodes_by_layer.keys())
        for i, layer in enumerate(sorted_layers[:-1]):
            next_layer = sorted_layers[i + 1]
            for src in nodes_by_layer[layer]:
                for tgt in nodes_by_layer[next_layer]:
                    circuit.add_edge(src, tgt, weight=1.0)
        
        return circuit, component_names
    
    def _compute_edge_importance(
        self,
        edge: CircuitEdge,
        clean_batch: Dict[str, torch.Tensor],
        corrupted_batch: Dict[str, torch.Tensor],
    ) -> float:
        """
        Compute importance of an edge via activation patching.
        
        The importance is measured by patching the source node and
        measuring the effect on the target node's output.
        """
        result = self.patcher.patch_single_component(
            clean_batch, corrupted_batch, edge.source
        )
        return abs(result.effect)
    
    def run(
        self,
        dataloader,
        algorithm: str = None,
    ) -> ACDCResult:
        """
        Run ACDC circuit discovery.
        
        Args:
            dataloader: DataLoader with training data
            algorithm: Optional algorithm name for labeling
            
        Returns:
            ACDCResult with discovered circuit
        """
        if self.config.verbose:
            print("=" * 60)
            print("ACDC: Automatic Circuit DisCovery")
            print("=" * 60)
        
        # Build full circuit
        self.full_circuit, component_names = self._build_component_graph()
        self.current_circuit = Circuit.from_dict(self.full_circuit.to_dict())
        self.current_circuit.name = f"circuit_{algorithm}" if algorithm else "discovered_circuit"
        self.current_circuit.algorithm = algorithm
        
        if self.config.verbose:
            print(f"Full circuit: {len(self.full_circuit.nodes)} nodes, {len(self.full_circuit.edges)} edges")
            print(f"Components to analyze: {len(component_names)}")
        
        # Collect baseline activations
        if self.config.ablation_type == "mean":
            if self.config.verbose:
                print("\nCollecting baseline activations...")
            self.patcher.collect_baseline_activations(
                dataloader, component_names, self.config.num_samples
            )
        
        # Get clean and corrupted batches
        batch = next(iter(dataloader))
        if hasattr(batch, 'to'):
            clean_batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                          for k, v in vars(batch).items() if not k.startswith('_')}
        elif isinstance(batch, dict):
            clean_batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                          for k, v in batch.items()}
        else:
            clean_batch = {'inputs': batch.to(self.device)}
        
        corrupted_batch = create_corrupted_input(clean_batch, corruption_type="shuffle")
        
        # Compute initial fidelity
        with torch.no_grad():
            initial_output = self.model(**clean_batch)
        
        # Iterative edge pruning
        pruned_edges = []
        history = []
        iteration = 0
        
        if self.config.verbose:
            print("\nStarting iterative edge pruning...")
        
        while iteration < self.config.max_iterations and len(self.current_circuit.edges) > 0:
            iteration += 1
            edges_to_remove = []
            
            # Score all edges
            edge_scores = []
            for edge in tqdm(self.current_circuit.edges, desc=f"Iteration {iteration}", 
                           disable=not self.config.verbose):
                importance = self._compute_edge_importance(edge, clean_batch, corrupted_batch)
                edge_scores.append((edge, importance))
            
            # Find edges below threshold
            for edge, importance in edge_scores:
                if importance < self.config.threshold:
                    edges_to_remove.append((edge, importance))
            
            # Remove low-importance edges
            if not edges_to_remove:
                if self.config.verbose:
                    print(f"Iteration {iteration}: No edges below threshold. Stopping.")
                break
            
            for edge, importance in edges_to_remove:
                self.current_circuit.remove_edge(edge.source, edge.target)
                pruned_edges.append((edge.source, edge.target, importance))
            
            # Record history
            history.append({
                'iteration': iteration,
                'edges_pruned': len(edges_to_remove),
                'edges_remaining': len(self.current_circuit.edges),
                'nodes_remaining': len(self.current_circuit.nodes),
            })
            
            if self.config.verbose:
                print(f"Iteration {iteration}: Pruned {len(edges_to_remove)} edges. "
                      f"Remaining: {len(self.current_circuit.edges)} edges")
        
        # Remove orphan nodes
        connected_nodes = set()
        for edge in self.current_circuit.edges:
            connected_nodes.add(edge.source)
            connected_nodes.add(edge.target)
        
        orphans = [n for n in list(self.current_circuit.nodes.keys()) 
                  if n not in connected_nodes and n not in ('input', 'output')]
        for orphan in orphans:
            self.current_circuit.remove_node(orphan)
        
        # Compute final fidelity (placeholder - would need circuit execution)
        final_fidelity = 1.0 - len(pruned_edges) / max(len(self.full_circuit.edges), 1)
        
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("ACDC Complete!")
            print(f"Final circuit: {len(self.current_circuit.nodes)} nodes, "
                  f"{len(self.current_circuit.edges)} edges")
            print(f"Pruned: {len(pruned_edges)} edges")
            print(f"Compression: {1 - len(self.current_circuit.edges)/max(len(self.full_circuit.edges), 1):.1%}")
            print("=" * 60)
        
        return ACDCResult(
            circuit=self.current_circuit,
            pruned_edges=pruned_edges,
            iterations=iteration,
            final_fidelity=final_fidelity,
            history=history,
        )
    
    def save_result(self, result: ACDCResult, path: str):
        """Save ACDC result to file."""
        result.circuit.save(path)
        
        # Save additional metadata
        meta_path = Path(path).with_suffix('.meta.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'pruned_edges': result.pruned_edges,
                'iterations': result.iterations,
                'final_fidelity': result.final_fidelity,
                'history': result.history,
                'config': vars(self.config),
            }, f, indent=2)


class ACDCForNAR(ACDC):
    """
    ACDC specialized for Neural Algorithmic Reasoning models.
    
    Handles the specific structure of NAR models with:
    - Encoder-Processor-Decoder architecture
    - Multiple message passing layers
    - Hint supervision
    """
    
    def _build_component_graph(self) -> Tuple[Circuit, List[str]]:
        """Build component graph specific to NAR architecture."""
        circuit = Circuit(name="nar_full")
        component_names = []
        
        # Input node
        circuit.add_node("input", "input", layer=0, component="encoder")
        component_names.append("encoder")
        
        # Processor layers
        if hasattr(self.model, 'processor') and hasattr(self.model.processor, 'layers'):
            for i, layer in enumerate(self.model.processor.layers):
                layer_num = i + 1
                
                # Attention component
                attn_name = f"processor.layers.{i}.node_attention"
                circuit.add_node(attn_name, "attention", layer=layer_num, component=attn_name)
                component_names.append(attn_name)
                
                # Edge MLP
                edge_mlp_name = f"processor.layers.{i}.edge_mlp"
                circuit.add_node(edge_mlp_name, "edge_update", layer=layer_num, component=edge_mlp_name)
                component_names.append(edge_mlp_name)
                
                # Node MLP
                node_mlp_name = f"processor.layers.{i}.node_mlp"
                circuit.add_node(node_mlp_name, "node_update", layer=layer_num, component=node_mlp_name)
                component_names.append(node_mlp_name)
                
                # Intra-layer edges
                circuit.add_edge(attn_name, edge_mlp_name, weight=1.0)
                circuit.add_edge(edge_mlp_name, node_mlp_name, weight=1.0)
                
                # Inter-layer edges
                if i == 0:
                    circuit.add_edge("input", attn_name, weight=1.0)
                else:
                    prev_node_mlp = f"processor.layers.{i-1}.node_mlp"
                    circuit.add_edge(prev_node_mlp, attn_name, weight=1.0)
                    # Residual connection
                    circuit.add_edge(prev_node_mlp, node_mlp_name, weight=0.5, edge_type="residual")
        
        # Output node
        circuit.add_node("output", "output", layer=100, component="decoder")
        component_names.append("decoder")
        
        # Connect last processor layer to output
        if hasattr(self.model, 'processor') and hasattr(self.model.processor, 'layers'):
            num_layers = len(self.model.processor.layers)
            if num_layers > 0:
                last_node_mlp = f"processor.layers.{num_layers-1}.node_mlp"
                circuit.add_edge(last_node_mlp, "output", weight=1.0)
        
        return circuit, component_names


def run_acdc_experiment(
    model: nn.Module,
    dataloader,
    algorithm: str,
    config: ACDCConfig = None,
    save_path: str = None,
) -> ACDCResult:
    """
    Convenience function to run ACDC on a model.
    
    Args:
        model: Neural network model
        dataloader: Data for patching experiments
        algorithm: Algorithm name
        config: ACDC configuration
        save_path: Optional path to save results
        
    Returns:
        ACDCResult
    """
    acdc = ACDCForNAR(model, config)
    result = acdc.run(dataloader, algorithm)
    
    if save_path:
        acdc.save_result(result, save_path)
    
    return result
