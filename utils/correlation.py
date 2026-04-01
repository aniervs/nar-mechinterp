"""Pearson correlation utilities for feature-concept analysis."""

import torch


def pearson_correlation_matrix(
    features: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute Pearson correlation between each feature and each label column.

    Args:
        features: (N, D) tensor of feature activations.
        labels: (N, C) tensor of concept labels.

    Returns:
        (D, C) tensor of Pearson correlations.
    """
    n = features.shape[0]
    feat_centered = features - features.mean(dim=0, keepdim=True)
    label_centered = labels - labels.mean(dim=0, keepdim=True)
    feat_norm = feat_centered / feat_centered.std(dim=0, keepdim=True).clamp(min=1e-8)
    label_norm = label_centered / label_centered.std(dim=0, keepdim=True).clamp(min=1e-8)
    return (feat_norm.T @ label_norm) / n


def find_monosemantic_features(
    corr_matrix: torch.Tensor,
    thresh: float = 0.4,
    max_shared: int = 2,
) -> list[int]:
    """Find features strongly correlated with at most *max_shared* concepts.

    A feature is considered monosemantic if its max |correlation| exceeds
    *thresh* and at most *max_shared* concepts exceed a secondary threshold
    of 0.3.

    Args:
        corr_matrix: (D, C) Pearson correlation matrix.
        thresh: Minimum |r| to qualify.
        max_shared: Maximum number of concepts above 0.3.

    Returns:
        List of monosemantic feature indices.
    """
    mono = []
    for i in range(corr_matrix.shape[0]):
        corrs = corr_matrix[i].abs()
        if corrs.max() > thresh and (corrs > 0.3).sum() <= max_shared:
            mono.append(i)
    return mono
