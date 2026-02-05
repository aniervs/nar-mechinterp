#!/usr/bin/env python3
"""
Run ACDC circuit discovery on trained NAR models.

Usage:
    python run_acdc.py --checkpoint checkpoints/bfs/best.pt --algorithm bfs
    python run_acdc.py --checkpoint checkpoints/dijkstra/best.pt --algorithm dijkstra --threshold 0.005
"""

import argparse
import sys
from pathlib import Path
import json

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import get_clrs_dataloader, get_algorithm_spec
from models import NARModel
from interp import ACDC, ACDCConfig
from utils import plot_circuit_graph, plot_edge_importance, save_circuit_to_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--metric", type=str, default="kl_divergence")
    parser.add_argument("--ablation_type", type=str, default="mean")
    parser.add_argument("--max_iterations", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_samples", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="circuits")
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> NARModel:
    """Load trained NAR model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint.get('args', {})
    
    model = NARModel(
        hidden_dim=args.get('hidden_dim', 128),
        num_layers=args.get('num_layers', 4),
        num_heads=args.get('num_heads', 8),
        processor_type=args.get('processor_type', 'mpnn'),
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Get algorithm spec
    spec = get_algorithm_spec(args.algorithm)
    output_types = spec.get('output_types', {'reach': 'node_mask'})
    
    print(f"\nAlgorithm: {args.algorithm}")
    print(f"Output types: {output_types}")
    
    # Create dataloader
    dataloader = get_clrs_dataloader(
        args.algorithm, "train", args.batch_size, args.data_samples, [16], args.seed
    )
    
    # ACDC configuration
    config = ACDCConfig(
        threshold=args.threshold,
        metric=args.metric,
        ablation_type=args.ablation_type,
        max_iterations=args.max_iterations,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    
    print(f"\nACDC Configuration:")
    print(f"  Threshold: {config.threshold}")
    print(f"  Metric: {config.metric}")
    print(f"  Ablation: {config.ablation_type}")
    print(f"  Max iterations: {config.max_iterations}")
    
    # Run ACDC
    print(f"\n{'='*60}")
    print(f"Running ACDC for {args.algorithm}")
    print(f"{'='*60}")
    
    acdc = ACDC(model, config, device)
    result = acdc.run(dataloader, output_types, args.algorithm, show_progress=True)
    
    # Results summary
    print(f"\n{'='*60}")
    print(f"Results Summary")
    print(f"{'='*60}")
    print(f"  Circuit nodes: {len(result.circuit.nodes)}")
    print(f"  Circuit edges: {len(result.circuit.edges)}")
    print(f"  Final fidelity: {result.fidelity:.4f}")
    print(f"  Pruning iterations: {len(result.pruning_history)}")
    
    # Save circuit
    circuit_path = output_dir / f"{args.algorithm}_circuit.json"
    save_circuit_to_json(result.circuit.to_dict(), circuit_path)
    print(f"\nCircuit saved to: {circuit_path}")
    
    # Save full results
    results_path = output_dir / f"{args.algorithm}_acdc_results.json"
    result.save(str(results_path))
    print(f"Full results saved to: {results_path}")
    
    # Save plots
    if args.save_plots:
        try:
            plot_circuit_graph(
                result.circuit.to_dict(),
                title=f"Circuit for {args.algorithm}",
                save_path=str(output_dir / f"{args.algorithm}_circuit.png"),
            )
            print(f"Circuit plot saved")
            
            plot_edge_importance(
                result.edge_scores,
                title=f"Edge Importance for {args.algorithm}",
                save_path=str(output_dir / f"{args.algorithm}_importance.png"),
            )
            print(f"Importance plot saved")
        except Exception as e:
            print(f"Warning: Could not create plots: {e}")
    
    print(f"\n{'='*60}")
    print("ACDC Complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
