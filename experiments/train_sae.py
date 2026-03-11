"""
Train SAE variants on NAR processor activations.

Supports three SAE variants:
  - standard: ReLU + L1 penalty (baseline)
  - batchtopk: BatchTopK sparsity (recommended)
  - transcoder: Maps processor input -> output (for circuit analysis)

Usage:
    # BatchTopK SAE (recommended)
    uv run python -m experiments.train_sae \
        --algorithm bfs \
        --model-checkpoint path/to/model.pt \
        --sae-type batchtopk \
        --expansion-factor 8 \
        --k 32 \
        --steps 50000

    # Standard SAE (baseline)
    uv run python -m experiments.train_sae \
        --algorithm bfs \
        --sae-type standard \
        --sparsity-coeff 1e-3

    # Transcoder
    uv run python -m experiments.train_sae \
        --algorithm bfs \
        --sae-type transcoder \
        --use-skip
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from data.clrs_dataset import (
    get_clrs_dataset,
    get_clrs_dataloader,
    get_algorithm_spec,
    spec_to_model_types,
)
from interp.sae import SparseAutoencoder, BatchTopKSAE, Transcoder, SAETrainer
from interp.activation_collector import ActivationCollector, make_activation_dataloader
from interp.concept_labels import collect_concept_labels
from models.nar_model import NARModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAE variants on NAR activations")
    # Data
    parser.add_argument("--algorithm", type=str, default="bfs")
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of CLRS samples for activation collection")
    parser.add_argument("--num-nodes", type=int, default=16)
    parser.add_argument("--edge-probability", type=float, default=0.2)
    parser.add_argument("--data-dir", type=str, default=None)
    # Model
    parser.add_argument("--model-checkpoint", type=str, default=None,
                        help="Path to trained NAR model checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--processor-type", type=str, default="mpnn")
    # SAE variant
    parser.add_argument("--sae-type", type=str, default="batchtopk",
                        choices=["standard", "batchtopk", "transcoder"],
                        help="SAE variant to train")
    parser.add_argument("--expansion-factor", type=int, default=8,
                        help="Dictionary size = hidden_dim * expansion_factor")
    # Standard SAE params
    parser.add_argument("--sparsity-coeff", type=float, default=1e-3,
                        help="L1 sparsity coefficient (standard SAE and L1 transcoder)")
    # BatchTopK params
    parser.add_argument("--k", type=int, default=32,
                        help="Average number of active features per sample (BatchTopK)")
    # Transcoder params
    parser.add_argument("--use-skip", action="store_true",
                        help="Use skip connection in transcoder (Skip Transcoder)")
    parser.add_argument("--transcoder-sparsity", type=str, default="batchtopk",
                        choices=["l1", "batchtopk"],
                        help="Sparsity mode for transcoder")
    # Training
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--resample-dead-every", type=int, default=25000)
    # Output
    parser.add_argument("--output-dir", type=str, default="results/sae")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def create_sae(args, device):
    """Create the SAE variant specified by args."""
    dict_size = args.hidden_dim * args.expansion_factor

    if args.sae_type == "standard":
        sae = SparseAutoencoder(
            input_dim=args.hidden_dim,
            dict_size=dict_size,
            sparsity_coeff=args.sparsity_coeff,
        )
        print(f"Standard SAE: input_dim={args.hidden_dim}, dict_size={dict_size}, "
              f"sparsity={args.sparsity_coeff}")

    elif args.sae_type == "batchtopk":
        sae = BatchTopKSAE(
            input_dim=args.hidden_dim,
            dict_size=dict_size,
            k=args.k,
        )
        print(f"BatchTopK SAE: input_dim={args.hidden_dim}, dict_size={dict_size}, "
              f"k={args.k}")

    elif args.sae_type == "transcoder":
        sae = Transcoder(
            input_dim=args.hidden_dim,
            output_dim=args.hidden_dim,
            dict_size=dict_size,
            sparsity_coeff=args.sparsity_coeff,
            use_skip=args.use_skip,
            sparsity_mode=args.transcoder_sparsity,
            k=args.k,
        )
        print(f"Transcoder: input_dim={args.hidden_dim}, output_dim={args.hidden_dim}, "
              f"dict_size={dict_size}, skip={args.use_skip}, "
              f"sparsity_mode={args.transcoder_sparsity}")
    else:
        raise ValueError(f"Unknown SAE type: {args.sae_type}")

    return sae.to(device), dict_size


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) / args.algorithm / args.sae_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Load dataset and spec ---
    print(f"Loading {args.algorithm} dataset ({args.num_samples} samples, "
          f"{args.num_nodes} nodes)...")
    ds = get_clrs_dataset(
        args.algorithm, split="train",
        num_samples=args.num_samples,
        num_nodes=args.num_nodes,
        edge_probability=args.edge_probability,
        data_dir=args.data_dir,
    )
    spec = get_algorithm_spec(args.algorithm, dataset=ds)
    output_types, _ = spec_to_model_types(spec)

    dataloader = get_clrs_dataloader(
        args.algorithm, split="train",
        batch_size=32,
        num_samples=args.num_samples,
        num_nodes=args.num_nodes,
        edge_probability=args.edge_probability,
        data_dir=args.data_dir,
    )

    # --- 2. Load or create NAR model ---
    print(f"Setting up NAR model (hidden_dim={args.hidden_dim}, {args.processor_type})...")
    model = NARModel(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        processor_type=args.processor_type,
    ).to(device)

    if args.model_checkpoint:
        print(f"Loading checkpoint: {args.model_checkpoint}")
        ckpt = torch.load(args.model_checkpoint, map_location=device, weights_only=True)
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict)

    model.eval()

    # --- 3. Collect activations ---
    print("Collecting activations...")
    collector = ActivationCollector(model, spec=spec, device=device)
    activations = collector.collect(
        dataloader,
        output_types=output_types,
    )
    print(f"Collected {activations.shape[0]} activation vectors of dim {activations.shape[1]}")

    act_path = output_dir / "activations.pt"
    torch.save(activations, act_path)
    print(f"Saved activations to {act_path}")

    # --- 4. Collect concept labels ---
    print("Collecting concept labels...")
    dataloader = get_clrs_dataloader(
        args.algorithm, split="train",
        batch_size=32,
        num_samples=args.num_samples,
        num_nodes=args.num_nodes,
        edge_probability=args.edge_probability,
        data_dir=args.data_dir,
        shuffle=False,
    )
    concept_result = collect_concept_labels(dataloader, spec, args.algorithm)
    concept_path = output_dir / "concept_labels.pt"
    torch.save({
        "labels": concept_result.labels,
        "descriptions": concept_result.concept_descriptions,
        "algorithm": concept_result.algorithm,
        "num_samples": concept_result.num_samples,
    }, concept_path)
    print(f"Saved concept labels to {concept_path}")
    print(f"  Concepts: {list(concept_result.labels.keys())}")
    print(f"  Samples: {concept_result.num_samples}")

    # --- 5. Create and train SAE variant ---
    sae, dict_size = create_sae(args, device)

    trainer = SAETrainer(
        sae,
        lr=args.lr,
        resample_dead_every=args.resample_dead_every,
    )

    act_loader = make_activation_dataloader(activations, batch_size=args.batch_size)

    # For transcoder, we need input-output pairs
    # In this simplified version, we use consecutive activations as proxy
    # (proper usage requires collecting both processor input and output)
    is_transcoder = args.sae_type == "transcoder"

    print(f"\nTraining {args.sae_type} for {args.steps} steps...")
    step = 0
    pbar = tqdm(total=args.steps, desc=f"{args.sae_type} Training")
    while step < args.steps:
        for (batch,) in act_loader:
            if step >= args.steps:
                break
            batch = batch.to(device)

            if is_transcoder:
                # For transcoder: use same activations as both input and target
                # (placeholder — proper usage needs processor input/output pairs)
                output = trainer.train_step(batch, targets=batch)
            else:
                output = trainer.train_step(batch)

            if step % 1000 == 0:
                pbar.set_postfix({
                    "loss": f"{output.loss.item():.4f}",
                    "recon": f"{output.reconstruction_loss.item():.4f}",
                    "L0": f"{output.l0.item():.1f}",
                })

            step += 1
            pbar.update(1)

    pbar.close()

    # --- 6. Save results ---
    stats = trainer.get_training_stats()
    print(f"\nFinal stats: {stats}")

    sae_path = output_dir / "sae.pt"
    torch.save({
        "state_dict": sae.state_dict(),
        "config": sae.get_config(),
        "sae_type": args.sae_type,
    }, sae_path)
    print(f"Saved {args.sae_type} to {sae_path}")

    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "args": vars(args),
            "final_stats": stats,
            "log_history": trainer.log_history,
        }, f, indent=2)
    print(f"Saved training log to {log_path}")

    # Quick dead feature analysis
    with torch.no_grad():
        dead_mask = sae.get_dead_features(activations[:10000].to(device))
        num_dead = dead_mask.sum().item()
        print(f"\nDead features: {num_dead}/{dict_size} ({100*num_dead/dict_size:.1f}%)")


if __name__ == "__main__":
    main()
