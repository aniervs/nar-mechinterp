"""
Analyze SAE features and correlate them with algorithmic concepts.

Usage:
    uv run python -m experiments.analyze_features \
        --algorithm bfs \
        --sae-checkpoint results/sae/bfs/sae.pt \
        --activations results/sae/bfs/activations.pt
"""

import argparse
import json
from pathlib import Path

import torch

from interp.sae import SparseAutoencoder, BatchTopKSAE, Transcoder
from interp.feature_analysis import FeatureAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze SAE features")
    parser.add_argument("--algorithm", type=str, default="bfs")
    parser.add_argument("--sae-checkpoint", type=str, required=True)
    parser.add_argument("--activations", type=str, required=True)
    parser.add_argument("--concept-labels", type=str, default=None,
                        help="Path to concept labels .pt file (dict of tensors)")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SAE (dispatch on sae_type saved in checkpoint)
    print(f"Loading SAE from {args.sae_checkpoint}...")
    checkpoint = torch.load(args.sae_checkpoint, map_location=device, weights_only=True)
    sae_type = checkpoint.get("sae_type", "standard")
    SAE_CLASSES = {
        "standard": SparseAutoencoder,
        "batchtopk": BatchTopKSAE,
        "transcoder": Transcoder,
    }
    sae_cls = SAE_CLASSES.get(sae_type, SparseAutoencoder)
    sae = sae_cls.from_config(checkpoint["config"]).to(device)
    sae.load_state_dict(checkpoint["state_dict"])
    sae.eval()
    is_transcoder = sae_type == "transcoder"
    print(f"SAE type: {sae_type}")

    # Load activations
    print(f"Loading activations from {args.activations}...")
    activations = torch.load(args.activations, map_location=device, weights_only=True)
    print(f"Activations shape: {activations.shape}")

    analyzer = FeatureAnalyzer(sae)

    # --- Basic feature stats ---
    print("\n=== Feature Statistics ===")
    stats = analyzer.compute_feature_stats(activations)

    # Sort by activation frequency
    stats_sorted = sorted(stats, key=lambda s: s.activation_frequency, reverse=True)
    dead = [s for s in stats if s.activation_frequency == 0]
    print(f"Total features: {len(stats)}")
    print(f"Dead features: {len(dead)} ({100*len(dead)/len(stats):.1f}%)")

    active_freqs = [s.activation_frequency for s in stats if s.activation_frequency > 0]
    if active_freqs:
        print(f"Active feature freq: min={min(active_freqs):.4f}, "
              f"median={sorted(active_freqs)[len(active_freqs)//2]:.4f}, "
              f"max={max(active_freqs):.4f}")

    # Top features by frequency
    print(f"\nTop-{args.top_k} most active features:")
    for s in stats_sorted[:args.top_k]:
        print(f"  Feature {s.feature_idx:4d}: freq={s.activation_frequency:.4f}, "
              f"mean_act={s.mean_activation:.4f}, max_act={s.max_activation:.4f}")

    # --- Concept correlation (if labels provided) ---
    if args.concept_labels:
        print(f"\n=== Concept Correlations ===")
        concept_data = torch.load(args.concept_labels, map_location=device, weights_only=True)
        # train_sae.py saves {"labels": {...}, "descriptions": {...}, ...}
        # Extract the actual label tensors
        if "labels" in concept_data:
            concept_labels = concept_data["labels"]
        else:
            concept_labels = concept_data
        print(f"Concepts: {list(concept_labels.keys())}")

        result = analyzer.compute_concept_correlations(activations, concept_labels)
        print(f"Monosemantic features: {len(result.monosemantic_features)}")

        for concept_name in result.concept_names:
            top = analyzer.find_features_for_concept(
                activations, concept_labels[concept_name], top_k=5
            )
            print(f"\n  Concept '{concept_name}' — top correlated features:")
            for feat_idx, corr in top:
                print(f"    Feature {feat_idx:4d}: correlation={corr:+.4f}")

    # --- L0 analysis ---
    print("\n=== Sparsity Analysis ===")
    with torch.no_grad():
        features = sae.encode(activations)
        l0_per_sample = (features > 0).float().sum(dim=-1)
        print(f"L0 (avg active features per sample): {l0_per_sample.mean():.1f}")
        print(f"L0 std: {l0_per_sample.std():.1f}")
        print(f"L0 min: {l0_per_sample.min():.0f}, max: {l0_per_sample.max():.0f}")

    # --- Reconstruction quality ---
    print("\n=== Reconstruction Quality ===")
    with torch.no_grad():
        sample = activations[:10000]
        if is_transcoder:
            output = sae(sample, target=sample)
            reconstructed = output.predicted_output
        else:
            output = sae(sample)
            reconstructed = output.reconstructed
        print(f"Reconstruction MSE: {output.reconstruction_loss.item():.6f}")
        # Fraction of variance explained
        total_var = sample.var(dim=0).sum()
        residual_var = (sample - reconstructed).var(dim=0).sum()
        fve = 1 - residual_var / total_var
        print(f"Fraction of variance explained: {fve.item():.4f}")

    # --- Save results ---
    output_dir = args.output_dir or str(Path(args.sae_checkpoint).parent)
    output_path = Path(output_dir) / "feature_analysis.json"
    results = {
        "algorithm": args.algorithm,
        "num_features": len(stats),
        "num_dead": len(dead),
        "avg_l0": l0_per_sample.mean().item(),
        "reconstruction_mse": output.reconstruction_loss.item(),
        "fve": fve.item(),
        "top_features": [
            {
                "idx": s.feature_idx,
                "freq": s.activation_frequency,
                "mean_act": s.mean_activation,
                "max_act": s.max_activation,
            }
            for s in stats_sorted[:args.top_k]
        ],
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved analysis to {output_path}")


if __name__ == "__main__":
    main()
