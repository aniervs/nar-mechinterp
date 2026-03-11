"""
Feature analysis tools for SAE-discovered features in NAR models.

Provides utilities to:
- Compute feature-to-algorithm-concept correlations
- Identify monosemantic vs polysemantic features
- Measure feature sharing across algorithms
- Find features that correspond to specific algorithmic concepts
  (e.g., "frontier node in BFS", "relaxed edge in Bellman-Ford")
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


@dataclass
class FeatureStats:
    """Statistics for a single SAE feature."""
    feature_idx: int
    activation_frequency: float  # Fraction of samples where feature fires
    mean_activation: float       # Mean activation when active
    max_activation: float        # Maximum activation observed
    concept_correlations: dict = field(default_factory=dict)  # concept_name -> correlation


@dataclass
class FeatureAnalysisResult:
    """Result of analyzing SAE features."""
    feature_stats: list[FeatureStats]
    concept_matrix: Optional[torch.Tensor] = None  # (num_features, num_concepts) correlations
    concept_names: list[str] = field(default_factory=list)
    dead_features: list[int] = field(default_factory=list)
    monosemantic_features: list[int] = field(default_factory=list)


class FeatureAnalyzer:
    """
    Analyzes features discovered by an SAE trained on NAR activations.

    Given an SAE and a labeled dataset (activations + algorithm state labels),
    computes how well each feature correlates with algorithmic concepts.
    """

    def __init__(self, sae: nn.Module):
        """
        Args:
            sae: Any SAE variant (SparseAutoencoder, BatchTopKSAE, or Transcoder).
                Must have an `encode(x)` method and a `dict_size` attribute.
        """
        self.sae = sae
        self.sae.eval()

    @torch.no_grad()
    def compute_feature_stats(
        self,
        activations: torch.Tensor,
        activation_threshold: float = 0.0,
    ) -> list[FeatureStats]:
        """
        Compute basic statistics for each feature.

        Args:
            activations: Collected activations, shape (N, input_dim).
            activation_threshold: Minimum value to consider a feature "active".

        Returns:
            List of FeatureStats, one per dictionary feature.
        """
        features = self.sae.encode(activations)  # (N, dict_size)
        active_mask = features > activation_threshold

        stats = []
        for i in range(self.sae.dict_size):
            feat_vals = features[:, i]
            is_active = active_mask[:, i]
            freq = is_active.float().mean().item()
            mean_act = feat_vals[is_active].mean().item() if is_active.any() else 0.0
            max_act = feat_vals.max().item()

            stats.append(FeatureStats(
                feature_idx=i,
                activation_frequency=freq,
                mean_activation=mean_act,
                max_activation=max_act,
            ))

        return stats

    @torch.no_grad()
    def compute_concept_correlations(
        self,
        activations: torch.Tensor,
        concept_labels: dict[str, torch.Tensor],
        activation_threshold: float = 0.0,
    ) -> FeatureAnalysisResult:
        """
        Compute correlations between SAE features and algorithmic concepts.

        Args:
            activations: Node activations, shape (N, input_dim).
            concept_labels: Dict mapping concept names to binary/continuous labels
                of shape (N,). E.g., {"is_frontier": tensor, "is_visited": tensor}.
            activation_threshold: Threshold for active features.

        Returns:
            FeatureAnalysisResult with per-feature stats and concept correlation matrix.
        """
        features = self.sae.encode(activations)  # (N, dict_size)
        concept_names = list(concept_labels.keys())
        num_concepts = len(concept_names)
        concept_matrix = torch.zeros(self.sae.dict_size, num_concepts)

        # Stack concept labels
        labels = torch.stack([concept_labels[name].float() for name in concept_names], dim=1)  # (N, C)

        # Compute Pearson correlation between each feature and each concept
        feat_centered = features - features.mean(dim=0, keepdim=True)
        label_centered = labels - labels.mean(dim=0, keepdim=True)

        feat_std = feat_centered.std(dim=0, keepdim=True).clamp(min=1e-8)
        label_std = label_centered.std(dim=0, keepdim=True).clamp(min=1e-8)

        feat_norm = feat_centered / feat_std  # (N, D)
        label_norm = label_centered / label_std  # (N, C)

        # Correlation matrix: (D, C)
        concept_matrix = (feat_norm.T @ label_norm) / activations.shape[0]

        # Compute per-feature stats with concept info
        stats = self.compute_feature_stats(activations, activation_threshold)
        for stat in stats:
            for j, name in enumerate(concept_names):
                stat.concept_correlations[name] = concept_matrix[stat.feature_idx, j].item()

        # Identify dead features
        dead = [s.feature_idx for s in stats if s.activation_frequency == 0.0]

        # Identify monosemantic features: high correlation with exactly one concept
        mono = []
        for i in range(self.sae.dict_size):
            corrs = concept_matrix[i].abs()
            if corrs.max() > 0.5 and (corrs > 0.3).sum() == 1:
                mono.append(i)

        return FeatureAnalysisResult(
            feature_stats=stats,
            concept_matrix=concept_matrix,
            concept_names=concept_names,
            dead_features=dead,
            monosemantic_features=mono,
        )

    @torch.no_grad()
    def find_features_for_concept(
        self,
        activations: torch.Tensor,
        concept_label: torch.Tensor,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Find the top-k features most correlated with a given concept.

        Args:
            activations: Shape (N, input_dim).
            concept_label: Binary or continuous label, shape (N,).
            top_k: Number of top features to return.

        Returns:
            List of (feature_idx, correlation) tuples, sorted by |correlation|.
        """
        features = self.sae.encode(activations)

        # Pearson correlation
        feat_centered = features - features.mean(dim=0, keepdim=True)
        label_centered = (concept_label - concept_label.mean()).unsqueeze(1)

        feat_std = feat_centered.std(dim=0).clamp(min=1e-8)
        label_std = label_centered.std().clamp(min=1e-8)

        corr = (feat_centered * label_centered).mean(dim=0) / (feat_std * label_std)

        # Top-k by absolute correlation
        values, indices = corr.abs().topk(min(top_k, len(corr)))
        return [(idx.item(), corr[idx].item()) for idx, val in zip(indices, values)]

    @torch.no_grad()
    def compute_feature_sharing(
        self,
        activations_per_algo: dict[str, torch.Tensor],
        activation_threshold: float = 0.0,
        sharing_threshold: float = 0.01,
    ) -> dict:
        """
        Analyze which features are shared vs algorithm-specific.

        Args:
            activations_per_algo: Dict mapping algorithm name to activation tensor (N_i, input_dim).
            activation_threshold: Minimum activation to count as "active".
            sharing_threshold: Minimum activation frequency to count as "used".

        Returns:
            Dict with:
                - shared_features: feature indices active across all algorithms
                - algo_specific: dict mapping algo name to its unique features
                - usage_matrix: (num_features, num_algos) activation frequency matrix
        """
        algo_names = list(activations_per_algo.keys())
        usage_matrix = torch.zeros(self.sae.dict_size, len(algo_names))

        for j, (algo, acts) in enumerate(activations_per_algo.items()):
            features = self.sae.encode(acts)
            active = (features > activation_threshold).float()
            usage_matrix[:, j] = active.mean(dim=0)

        # Shared: active in all algorithms
        used_mask = usage_matrix > sharing_threshold
        shared = used_mask.all(dim=1).nonzero(as_tuple=True)[0].tolist()

        # Algorithm-specific: active in exactly one
        algo_specific = {}
        for j, algo in enumerate(algo_names):
            specific = (used_mask[:, j] & ~used_mask[:, [k for k in range(len(algo_names)) if k != j]].any(dim=1))
            algo_specific[algo] = specific.nonzero(as_tuple=True)[0].tolist()

        return {
            "shared_features": shared,
            "algo_specific": algo_specific,
            "usage_matrix": usage_matrix,
            "algo_names": algo_names,
        }
