#!/usr/bin/env python3
"""
Analyze and compare discovered circuits across algorithms.

Usage:
    python analyze_circuits.py --circuits circuits/bfs_circuit.json circuits/dfs_circuit.json
    python analyze_circuits.py --circuit_dir circuits/ --compare
"""

import argparse
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from interp import (
    Circuit,
    compare_circuits,
    merge_circuits,
    find_shared_subcircuit,
    CircuitEvaluator,
)
from utils import (
    plot_circuit_graph,
    plot_edge_importance,
    load_circuit_from_json,
    save_circuit_to_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuits", type=str, nargs="+", help="Circuit JSON files")
    parser.add_argument("--circuit_dir", type=str, help="Directory with circuit files")
    parser.add_argument("--compare", action="store_true", help="Compare all circuits")
    parser.add_argument("--find_shared", action="store_true", help="Find shared subcircuit")
    parser.add_argument("--merge", action="store_true", help="Merge all circuits")
    parser.add_argument("--output_dir", type=str, default="analysis")
    parser.add_argument("--save_plots", action="store_true")
    return parser.parse_args()


def load_circuits(args) -> dict:
    """Load circuits from files or directory."""
    circuits = {}
    
    if args.circuits:
        for path in args.circuits:
            path = Path(path)
            if path.exists():
                data = load_circuit_from_json(path)
                circuit = Circuit.from_dict(data)
                name = path.stem.replace("_circuit", "")
                circuits[name] = circuit
                print(f"Loaded: {name} ({circuit.num_nodes} nodes, {circuit.num_edges} edges)")
    
    if args.circuit_dir:
        circuit_dir = Path(args.circuit_dir)
        for path in circuit_dir.glob("*_circuit.json"):
            if path.stem not in [c.replace("_circuit", "") for c in circuits]:
                data = load_circuit_from_json(path)
                circuit = Circuit.from_dict(data)
                name = path.stem.replace("_circuit", "")
                circuits[name] = circuit
                print(f"Loaded: {name} ({circuit.num_nodes} nodes, {circuit.num_edges} edges)")
    
    return circuits


def analyze_single_circuit(name: str, circuit: Circuit, output_dir: Path, save_plots: bool):
    """Analyze a single circuit."""
    print(f"\n{'='*60}")
    print(f"Analysis: {name}")
    print(f"{'='*60}")
    
    stats = circuit.compute_statistics()
    
    print(f"  Nodes: {stats['num_nodes']}")
    print(f"  Edges: {stats['num_edges']}")
    print(f"  Layers: {stats['num_layers']}")
    print(f"  Connected: {stats['is_connected']}")
    print(f"  Avg in-degree: {stats['avg_in_degree']:.2f}")
    print(f"  Avg out-degree: {stats['avg_out_degree']:.2f}")
    
    # Node type distribution
    print("\n  Node types:")
    for node_type, count in stats['node_type_distribution'].items():
        print(f"    {node_type}: {count}")
    
    # Edge weight stats
    if 'edge_weight_mean' in stats:
        print(f"\n  Edge weights:")
        print(f"    Mean: {stats['edge_weight_mean']:.4f}")
        print(f"    Std: {stats['edge_weight_std']:.4f}")
        print(f"    Range: [{stats['edge_weight_min']:.4f}, {stats['edge_weight_max']:.4f}]")
    
    # Save stats
    stats_path = output_dir / f"{name}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\n  Stats saved to: {stats_path}")
    
    # Plot
    if save_plots:
        try:
            plot_circuit_graph(
                circuit.to_dict(),
                title=f"Circuit: {name}",
                save_path=str(output_dir / f"{name}_graph.png"),
            )
            print(f"  Graph saved to: {output_dir / f'{name}_graph.png'}")
        except Exception as e:
            print(f"  Warning: Could not create plot: {e}")
    
    return stats


def compare_all_circuits(circuits: dict, output_dir: Path):
    """Compare all pairs of circuits."""
    print(f"\n{'='*60}")
    print("Circuit Comparisons")
    print(f"{'='*60}")
    
    names = list(circuits.keys())
    comparisons = {}
    
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            comparison = compare_circuits(
                circuits[name1], circuits[name2], name1, name2
            )
            key = f"{name1}_vs_{name2}"
            comparisons[key] = comparison
            
            print(f"\n{name1} vs {name2}:")
            print(f"  Node similarity (Jaccard): {comparison['node_comparison']['jaccard_similarity']:.4f}")
            print(f"  Edge similarity (Jaccard): {comparison['edge_comparison']['jaccard_similarity']:.4f}")
            print(f"  Overall similarity: {comparison['overall_similarity']:.4f}")
            print(f"  Common nodes: {len(comparison['node_comparison']['common'])}")
            print(f"  Common edges: {comparison['edge_comparison']['num_common']}")
    
    # Save comparisons
    comparisons_path = output_dir / "circuit_comparisons.json"
    with open(comparisons_path, 'w') as f:
        json.dump(comparisons, f, indent=2, default=list)
    print(f"\nComparisons saved to: {comparisons_path}")
    
    return comparisons


def find_and_analyze_shared(circuits: dict, output_dir: Path, save_plots: bool):
    """Find shared subcircuit across all algorithms."""
    print(f"\n{'='*60}")
    print("Shared Subcircuit Analysis")
    print(f"{'='*60}")
    
    circuit_list = list(circuits.values())
    shared = find_shared_subcircuit(circuit_list)
    
    print(f"  Shared nodes: {shared.num_nodes}")
    print(f"  Shared edges: {shared.num_edges}")
    
    if shared.num_nodes > 0:
        # Analyze shared circuit
        stats = shared.compute_statistics()
        
        print(f"\n  Shared node types:")
        for node_type, count in stats['node_type_distribution'].items():
            print(f"    {node_type}: {count}")
        
        # Save shared circuit
        shared.save(str(output_dir / "shared_circuit.json"))
        print(f"\n  Shared circuit saved to: {output_dir / 'shared_circuit.json'}")
        
        if save_plots:
            try:
                plot_circuit_graph(
                    shared.to_dict(),
                    title="Shared Subcircuit",
                    save_path=str(output_dir / "shared_circuit.png"),
                )
                print(f"  Plot saved to: {output_dir / 'shared_circuit.png'}")
            except Exception as e:
                print(f"  Warning: Could not create plot: {e}")
    else:
        print("  No shared components found across all circuits.")
    
    return shared


def merge_all_circuits(circuits: dict, output_dir: Path, save_plots: bool):
    """Merge all circuits into one."""
    print(f"\n{'='*60}")
    print("Merged Circuit")
    print(f"{'='*60}")
    
    circuit_list = list(circuits.values())
    merged = merge_circuits(circuit_list)
    
    print(f"  Total nodes: {merged.num_nodes}")
    print(f"  Total edges: {merged.num_edges}")
    
    stats = merged.compute_statistics()
    print(f"\n  Node types:")
    for node_type, count in stats['node_type_distribution'].items():
        print(f"    {node_type}: {count}")
    
    # Save merged circuit
    merged.save(str(output_dir / "merged_circuit.json"))
    print(f"\n  Merged circuit saved to: {output_dir / 'merged_circuit.json'}")
    
    if save_plots:
        try:
            plot_circuit_graph(
                merged.to_dict(),
                title="Merged Circuit (All Algorithms)",
                save_path=str(output_dir / "merged_circuit.png"),
            )
            print(f"  Plot saved to: {output_dir / 'merged_circuit.png'}")
        except Exception as e:
            print(f"  Warning: Could not create plot: {e}")
    
    return merged


def main():
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load circuits
    print("Loading circuits...")
    circuits = load_circuits(args)
    
    if not circuits:
        print("No circuits found! Provide --circuits or --circuit_dir")
        return
    
    print(f"\nLoaded {len(circuits)} circuits: {list(circuits.keys())}")
    
    # Analyze each circuit
    all_stats = {}
    for name, circuit in circuits.items():
        stats = analyze_single_circuit(name, circuit, output_dir, args.save_plots)
        all_stats[name] = stats
    
    # Compare circuits
    if args.compare and len(circuits) > 1:
        compare_all_circuits(circuits, output_dir)
    
    # Find shared subcircuit
    if args.find_shared and len(circuits) > 1:
        find_and_analyze_shared(circuits, output_dir, args.save_plots)
    
    # Merge circuits
    if args.merge and len(circuits) > 1:
        merge_all_circuits(circuits, output_dir, args.save_plots)
    
    # Save summary
    summary = {
        'circuits': list(circuits.keys()),
        'stats': {name: {k: v for k, v in s.items() if not isinstance(v, dict)} 
                  for name, s in all_stats.items()},
    }
    
    with open(output_dir / "analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
