"""SAE evaluation helpers: reconstruction quality, sparsity, concept analysis."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .correlation import pearson_correlation_matrix, find_monosemantic_features


def evaluate_sae(
    sae: nn.Module,
    activations: torch.Tensor,
    batch_size: int = 256,
    device: Optional[torch.device] = None,
    max_samples: Optional[int] = None,
) -> dict:
    """Compute reconstruction quality and sparsity metrics for an SAE.

    Args:
        sae: Any SAE variant with a forward() returning an object with
            .reconstructed and .features attributes.
        activations: (N, input_dim) activation tensor.
        batch_size: Evaluation batch size.
        device: Device for computation.
        max_samples: Cap on number of samples to evaluate (None = all).

    Returns:
        Dict with keys: fve, mse, dead_count, dead_pct, l0_mean, l0_std,
        features (Tensor), reconstructed (Tensor).
    """
    if device is None:
        device = next(sae.parameters()).device
    sae.eval()

    n = activations.shape[0] if max_samples is None else min(max_samples, activations.shape[0])
    sample = activations[:n]

    recon_list, feat_list = [], []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = sample[i : i + batch_size].to(device)
            out = sae(batch)
            recon_list.append(out.reconstructed.cpu())
            feat_list.append(out.features.cpu())
            del out, batch

    reconstructed = torch.cat(recon_list)
    features = torch.cat(feat_list).float()
    del recon_list, feat_list

    mse = F.mse_loss(reconstructed, sample).item()
    total_var = sample.var(dim=0).sum()
    residual_var = (sample - reconstructed).var(dim=0).sum()
    fve = (1 - residual_var / total_var).item()

    dict_size = features.shape[1]
    dead_count = (features.sum(dim=0) == 0).sum().item()
    l0_per = (features > 0).float().sum(dim=-1)

    return {
        "fve": fve,
        "mse": mse,
        "dead_count": int(dead_count),
        "dead_pct": 100 * dead_count / dict_size,
        "l0_mean": l0_per.mean().item(),
        "l0_std": l0_per.std().item(),
        "features": features,
        "reconstructed": reconstructed,
    }


def compute_concept_analysis(
    features: torch.Tensor,
    concept_labels: dict[str, torch.Tensor],
    concept_names: Optional[list[str]] = None,
    mono_thresh: float = 0.4,
    max_samples: Optional[int] = None,
) -> dict:
    """Compute feature-concept correlations and identify monosemantic features.

    Args:
        features: (N, D) SAE feature activations.
        concept_labels: {concept_name: (N,) label tensor}.
        concept_names: Ordered list of concept names. Defaults to dict keys.
        mono_thresh: Threshold for monosemantic classification.
        max_samples: Cap on samples used for correlation.

    Returns:
        Dict with keys: cm (correlation matrix), concept_names, mono (list),
        dead (list), features (truncated to max_samples).
    """
    if concept_names is None:
        concept_names = list(concept_labels.keys())

    n = features.shape[0]
    if max_samples is not None:
        n = min(n, max_samples, min(v.shape[0] for v in concept_labels.values()))
    else:
        n = min(n, min(v.shape[0] for v in concept_labels.values()))

    labels = torch.stack(
        [concept_labels[name][:n].float().cpu() for name in concept_names], dim=1
    )
    feats = features[:n].float().cpu()

    cm = pearson_correlation_matrix(feats, labels)
    mono = find_monosemantic_features(cm, thresh=mono_thresh)
    dead = (feats.sum(dim=0) == 0).nonzero(as_tuple=True)[0].tolist()

    return {
        "cm": cm,
        "concept_names": concept_names,
        "mono": mono,
        "dead": dead,
        "features": feats,
    }
