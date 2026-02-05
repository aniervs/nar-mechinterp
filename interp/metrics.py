"""
Interpretability Metrics for Evaluating Circuits.

Provides metrics for:
- Circuit fidelity (how well it approximates the full model)
- Circuit minimality (size and complexity)
- Circuit specificity (algorithm-specific vs shared)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .circuit import Circuit, compare_circuits


def compute_faithfulness(
    model: nn.Module,
    circuit: Circuit,
    dataloader,
    output_types: Dict[str, str],
    device: torch.device,
    num_batches: int = 10,
) -> Dict[str, float]:
    """
    Compute faithfulness metrics for a circuit.
    
    Faithfulness measures how well the circuit's behavior matches
    the full model's behavior.
    
    Args:
        model: Full model
        circuit: Discovered circuit
        dataloader: Data for evaluation
        output_types: Types of model outputs
        device: Device to run on
        num_batches: Number of batches to evaluate
        
    Returns:
        Dict with faithfulness metrics
    """
    metrics = {
        'accuracy': 0.0,
        'kl_divergence': 0.0,
        'mse': 0.0,
    }
    count = 0
    
    batch_iter = iter(dataloader)
    for _ in range(num_batches):
        try:
            batch = next(batch_iter)
        except StopIteration:
            break
        
        batch = batch.to(device)
        
        with torch.no_grad():
            output = model(
                inputs=batch.inputs,
                outputs=batch.outputs,
                output_types=output_types,
            )
        
        for name, pred in output.predictions.items():
            if name in batch.outputs:
                target = batch.outputs[name]
                out_type = output_types.get(name, 'node_mask')
                
                # Accuracy
                if out_type in ['node_mask']:
                    pred_binary = (torch.sigmoid(pred[:, :target.shape[-1]]) > 0.5)
                    acc = (pred_binary == target.bool()).float().mean()
                    metrics['accuracy'] += acc.item()
                elif out_type == 'node_pointer':
                    pred_idx = pred.argmax(dim=-1)
                    target_idx = target.long() if target.dim() < pred.dim() else target.argmax(dim=-1)
                    acc = (pred_idx == target_idx).float().mean()
                    metrics['accuracy'] += acc.item()
                else:
                    metrics['accuracy'] += 1.0
                
                # MSE
                mse = F.mse_loss(pred[:, :target.shape[-1]], target.float())
                metrics['mse'] += mse.item()
                
                # KL Divergence (for probability outputs)
                if out_type in ['node_mask', 'node_categorical']:
                    pred_probs = torch.sigmoid(pred[:, :target.shape[-1]]).clamp(1e-7, 1-1e-7)
                    target_probs = target.float().clamp(1e-7, 1-1e-7)
                    kl = F.kl_div(torch.log(pred_probs), target_probs, reduction='batchmean')
                    metrics['kl_divergence'] += kl.item()
                
                count += 1
    
    # Average metrics
    for key in metrics:
        metrics[key] /= max(count, 1)
    
    return metrics


def compute_minimality(circuit: Circuit) -> Dict[str, float]:
    """
    Compute minimality metrics for a circuit.
    
    Minimality measures how small and simple the circuit is.
    
    Args:
        circuit: Circuit to evaluate
        
    Returns:
        Dict with minimality metrics
    """
    stats = circuit.compute_statistics()
    
    # Compute compression ratio (would need original size)
    num_nodes = stats['num_nodes']
    num_edges = stats['num_edges']
    
    # Density (edges per possible edges)
    max_edges = num_nodes * (num_nodes - 1) if num_nodes > 1 else 1
    density = num_edges / max_edges
    
    # Depth (number of layers)
    depth = stats['num_layers']
    
    # Average degree
    avg_degree = (stats['avg_in_degree'] + stats['avg_out_degree']) / 2
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'density': density,
        'depth': depth,
        'avg_degree': avg_degree,
        'is_connected': float(stats['is_connected']),
    }


def compute_specificity(
    circuit: Circuit,
    other_circuits: List[Circuit],
) -> Dict[str, float]:
    """
    Compute specificity metrics for a circuit.
    
    Specificity measures how unique/specialized the circuit is
    compared to circuits for other tasks.
    
    Args:
        circuit: Circuit to evaluate
        other_circuits: Other circuits for comparison
        
    Returns:
        Dict with specificity metrics
    """
    if not other_circuits:
        return {
            'uniqueness': 1.0,
            'avg_similarity': 0.0,
            'min_similarity': 0.0,
            'max_similarity': 0.0,
        }
    
    similarities = []
    for other in other_circuits:
        comparison = compare_circuits(circuit, other)
        similarities.append(comparison['overall_similarity'])
    
    return {
        'uniqueness': 1.0 - np.mean(similarities),
        'avg_similarity': np.mean(similarities),
        'min_similarity': np.min(similarities),
        'max_similarity': np.max(similarities),
    }


def compute_all_metrics(
    model: nn.Module,
    circuit: Circuit,
    dataloader,
    output_types: Dict[str, str],
    device: torch.device,
    other_circuits: List[Circuit] = None,
    num_batches: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Compute all interpretability metrics for a circuit.
    
    Args:
        model: Full model
        circuit: Circuit to evaluate
        dataloader: Data for evaluation
        output_types: Types of model outputs
        device: Device to run on
        other_circuits: Other circuits for specificity comparison
        num_batches: Number of batches for evaluation
        
    Returns:
        Dict with all metrics grouped by category
    """
    return {
        'faithfulness': compute_faithfulness(
            model, circuit, dataloader, output_types, device, num_batches
        ),
        'minimality': compute_minimality(circuit),
        'specificity': compute_specificity(circuit, other_circuits or []),
    }


class CircuitEvaluator:
    """
    Evaluator for comparing and analyzing multiple circuits.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
    ):
        self.model = model
        self.device = device
        self.circuits: Dict[str, Circuit] = {}
        self.metrics: Dict[str, Dict] = {}
    
    def add_circuit(
        self,
        name: str,
        circuit: Circuit,
        dataloader=None,
        output_types: Dict[str, str] = None,
    ):
        """Add a circuit for evaluation."""
        self.circuits[name] = circuit
        
        if dataloader is not None and output_types is not None:
            self.metrics[name] = compute_all_metrics(
                self.model,
                circuit,
                dataloader,
                output_types,
                self.device,
                [c for n, c in self.circuits.items() if n != name],
            )
    
    def get_comparison_matrix(self) -> np.ndarray:
        """Get pairwise similarity matrix for all circuits."""
        names = list(self.circuits.keys())
        n = len(names)
        matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                comparison = compare_circuits(
                    self.circuits[names[i]],
                    self.circuits[names[j]],
                )
                similarity = comparison['overall_similarity']
                matrix[i, j] = similarity
                matrix[j, i] = similarity
        
        return matrix
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all circuits."""
        return {
            'circuits': {
                name: {
                    'num_nodes': c.num_nodes,
                    'num_edges': c.num_edges,
                    'num_layers': c.num_layers,
                }
                for name, c in self.circuits.items()
            },
            'metrics': self.metrics,
            'comparison_matrix': self.get_comparison_matrix().tolist(),
        }
