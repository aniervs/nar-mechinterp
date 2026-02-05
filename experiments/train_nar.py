#!/usr/bin/env python3
"""
Train Neural Algorithmic Reasoning model on CLRS-30.

Usage:
    python train_nar.py --algorithm bfs --epochs 100
    python train_nar.py --algorithm dijkstra --hidden_dim 256
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import get_clrs_dataloader, get_algorithm_spec
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
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--val_samples", type=int, default=128)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_epoch(model, loader, optimizer, output_types, hint_types, device):
    model.train()
    total_loss = 0
    cnt = 0
    for batch in tqdm(loader, desc="Training"):
        cnt += 1
        batch = batch.to(device)
        optimizer.zero_grad()
        
        output = model(
            inputs=batch.inputs,
            hints=batch.hints,
            outputs=batch.outputs,
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
def validate(model, loader, output_types, hint_types, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    count = 0
    
    for batch in loader:
        batch = batch.to(device)
        output = model(
            inputs=batch.inputs,
            outputs=batch.outputs,
            output_types=output_types,
            hint_types=hint_types,
            num_steps=batch.lengths.max().item(),
        )
        total_loss += output.total_loss.item()
        
        for name, pred in output.predictions.items():
            if name in batch.outputs:
                target = batch.outputs[name]
                if output_types.get(name) in ['node_mask']:
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
    
    return total_loss / len(loader), total_acc / max(count, 1)


def main():
    args = parse_args()
    
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    spec = get_algorithm_spec(args.algorithm)
    output_types = spec.get('output_types', {'reach': 'node_mask'})
    hint_types = spec.get('hint_types', {'reach': 'node_mask'})
    
    print(f"\nAlgorithm: {args.algorithm}")
    print(f"Output types: {output_types}")
    
    train_loader = get_clrs_dataloader(
        args.algorithm, "train", args.batch_size, args.train_samples, [16], args.seed
    )
    val_loader = get_clrs_dataloader(
        args.algorithm, "val", args.batch_size, args.val_samples, [16], args.seed + 1
    )
    
    model = NARModel(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        processor_type="mpnn",
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    ckpt_dir = Path(args.checkpoint_dir) / args.algorithm
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, output_types, hint_types, device)
        val_loss, val_acc = validate(model, val_loader, output_types, hint_types, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.4f}")
        
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
