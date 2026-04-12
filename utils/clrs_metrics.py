"""CLRS-30 benchmark evaluation metrics.

Ports the official dm-clrs evaluation protocol to PyTorch tensors.
Follows the metric definitions from clrs._src.evaluation exactly:
    - MASK   → F1 score (handles class imbalance)
    - POINTER → index accuracy
    - MASK_ONE / CATEGORICAL → argmax accuracy
    - SCALAR → negative MSE (higher is better, matching CLRS convention)
"""

import torch
import numpy as np
from typing import Optional


def mask_f1(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """F1 score for MASK outputs (binary classification with imbalance).

    This is the official CLRS metric for mask-type outputs.
    Masked values (truth == -1) are excluded.
    """
    pred_np = pred.detach().cpu().float().numpy()
    truth_np = truth.detach().cpu().float().numpy()

    mask = (truth_np != -1).astype(np.float32)
    pred_bin = (pred_np > 0.5).astype(np.float32)
    truth_bin = (truth_np > 0.5).astype(np.float32)

    tp = np.sum(pred_bin * truth_bin * mask)
    fp = np.sum(pred_bin * (1 - truth_bin) * mask)
    fn = np.sum((1 - pred_bin) * truth_bin * mask)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0

    if precision + recall > 0:
        return float(2.0 * precision * recall / (precision + recall))
    return 0.0


def pointer_accuracy(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """Accuracy for POINTER outputs (index prediction).

    pred: logits of shape (..., N) or indices of shape (...)
    truth: indices of shape (...)
    """
    if pred.dim() > truth.dim():
        pred_idx = pred.argmax(dim=-1)
    else:
        pred_idx = pred
    truth_idx = truth.long()
    return float((pred_idx == truth_idx).float().mean().item())


def categorical_accuracy(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """Accuracy for MASK_ONE / CATEGORICAL outputs (argmax match).

    Masked values (all -1 along last dim) are excluded.
    """
    pred_np = pred.detach().cpu().float().numpy()
    truth_np = truth.detach().cpu().float().numpy()

    if pred_np.ndim < 2 or truth_np.ndim < 2:
        # Fallback: treat as direct comparison
        return float(np.mean(np.argmax(pred_np, -1) == np.argmax(truth_np, -1)))

    mask = np.all(truth_np != -1, axis=-1)
    if mask.sum() == 0:
        return 1.0
    correct = (np.argmax(pred_np, -1) == np.argmax(truth_np, -1))
    return float(np.sum(correct * mask) / np.sum(mask))


def scalar_mse(pred: torch.Tensor, truth: torch.Tensor) -> float:
    """MSE for SCALAR outputs. Lower is better.

    Returns negative MSE so that higher = better (consistent with other metrics).
    """
    return -float(((pred.float() - truth.float()) ** 2).mean().item())


# Map CLRS type names to evaluation functions
EVAL_FN = {
    'mask': mask_f1,
    'node_mask': mask_f1,
    'edge_mask': mask_f1,
    'pointer': pointer_accuracy,
    'node_pointer': pointer_accuracy,
    'edge_pointer': pointer_accuracy,
    'mask_one': categorical_accuracy,
    'node_mask_one': categorical_accuracy,
    'categorical': categorical_accuracy,
    'node_categorical': categorical_accuracy,
    'scalar': scalar_mse,
    'node_scalar': scalar_mse,
    'edge_scalar': scalar_mse,
}


def evaluate_outputs(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    output_types: dict[str, str],
) -> dict[str, float]:
    """Evaluate all output predictions using the correct CLRS metric per type.

    Args:
        predictions: {field_name: predicted tensor}
        targets: {field_name: ground truth tensor}
        output_types: {field_name: type string (e.g. 'node_pointer', 'node_mask')}

    Returns:
        Dict with per-field scores and an overall 'score' (mean of all fields).
    """
    evals = {}

    for name, pred in predictions.items():
        if name not in targets:
            continue
        target = targets[name]
        otype = output_types.get(name, '')

        # Skip if shapes are incompatible (edge-level outputs not converted to dense)
        if target.shape[0] != pred.shape[0]:
            continue

        eval_fn = EVAL_FN.get(otype)
        if eval_fn is None:
            continue

        # For mask outputs, apply sigmoid to logits
        if 'mask' in otype and 'mask_one' not in otype:
            pred = torch.sigmoid(pred)
            # Trim prediction to target size if needed
            if pred.shape != target.shape:
                pred = pred[:, :target.shape[-1]]

        # For pointer outputs, trim to target node count
        if 'pointer' in otype and pred.dim() > target.dim():
            pred = pred[:, :target.shape[-1], :]

        evals[name] = eval_fn(pred, target)

    if evals:
        # For scalars, exclude from the mean score (CLRS convention: hints with
        # scalars don't contribute to the "score"). For outputs, include all.
        evals['score'] = sum(evals.values()) / len(evals)
    else:
        evals['score'] = 0.0

    return evals
