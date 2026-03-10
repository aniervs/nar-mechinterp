#!/usr/bin/env python3
"""
Train Neural Algorithmic Reasoning model on CLRS-30.

Usage:
    uv run python -m experiments.train_nar --algorithm bfs --epochs 100
    uv run python -m experiments.train_nar --algorithm dijkstra --hidden_dim 256
"""

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data import (
    get_clrs_dataset,
    get_clrs_dataloader,
    get_algorithm_spec,
    spec_to_model_types,
    batch_to_model_inputs,
)
from models import NARModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="bfs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--processor_type", type=str, default="mpnn")
    parser.add_argument("--num_nodes", type=int, default=16)
    parser.add_argument("--edge_probability", type=float, default=0.2)
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--val_samples", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_epoch(model, loader, optimizer, spec, output_types, hint_types, device):
    model.train()
    total_loss = 0
    cnt = 0
    for batch in tqdm(loader, desc="Training"):
        cnt += 1
        batch = batch.to(device)
        inputs, outputs, hints = batch_to_model_inputs(batch, spec, device)

        optimizer.zero_grad()

        output = model(
            inputs=inputs,
            hints=hints,
            outputs=outputs,
            output_types=output_types,
            hint_types=hint_types,
            num_steps=batch.lengths.max().item(),
        )

        output.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += output.total_loss.item()

        avg_loss = total_loss / cnt
        if torch.isnan(torch.tensor(avg_loss)):
            print(f"NaN loss at batch {cnt}, stopping training.")
            exit(-1)

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, spec, output_types, hint_types, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    count = 0

    for batch in loader:
        batch = batch.to(device)
        inputs, outputs, hints = batch_to_model_inputs(batch, spec, device)

        output = model(
            inputs=inputs,
            outputs=outputs,
            output_types=output_types,
            hint_types=hint_types,
            num_steps=batch.lengths.max().item(),
        )
        total_loss += output.total_loss.item()

        for name, pred in output.predictions.items():
            if name in outputs:
                target = outputs[name]
                if output_types.get(name) == 'node_mask':
                    pred_bin = torch.sigmoid(pred[:, :target.shape[-1]]) > 0.5
                    acc = (pred_bin == target.bool()).float().mean()
                elif output_types.get(name) == 'node_pointer':
                    pred_idx = pred.argmax(-1)
                    tgt_idx = target.long() if target.dim() < pred.dim() else target.argmax(-1)
                    acc = (pred_idx == tgt_idx).float().mean()
                else:
                    acc = torch.tensor(0.0)
                total_acc += acc.item()
                count += 1

    return total_loss / max(len(loader), 1), total_acc / max(count, 1)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Get algorithm spec and derive model type dicts
    ds = get_clrs_dataset(
        args.algorithm, split="train",
        num_samples=args.train_samples,
        num_nodes=args.num_nodes,
        edge_probability=args.edge_probability,
        data_dir=args.data_dir,
    )
    spec = get_algorithm_spec(args.algorithm, dataset=ds)
    output_types, hint_types = spec_to_model_types(spec)

    print(f"\nAlgorithm: {args.algorithm}")
    print(f"Output types: {output_types}")
    print(f"Hint types: {hint_types}")

    train_loader = get_clrs_dataloader(
        args.algorithm, "train",
        batch_size=args.batch_size,
        num_samples=args.train_samples,
        num_nodes=args.num_nodes,
        edge_probability=args.edge_probability,
        data_dir=args.data_dir,
        seed=args.seed,
    )
    val_loader = get_clrs_dataloader(
        args.algorithm, "val",
        batch_size=args.batch_size,
        num_samples=args.val_samples,
        num_nodes=args.num_nodes,
        edge_probability=args.edge_probability,
        data_dir=args.data_dir,
        seed=args.seed + 1,
    )

    model = NARModel(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        processor_type=args.processor_type,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = Path(args.checkpoint_dir) / args.algorithm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model, train_loader, optimizer, spec, output_types, hint_types, device
        )
        val_loss, val_acc = validate(
            model, val_loader, spec, output_types, hint_types, device
        )
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'args': vars(args),
                'epoch': epoch,
            }, ckpt_dir / "best.pt")

    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
    }, ckpt_dir / "final.pt")

    print(f"\nDone! Models saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
