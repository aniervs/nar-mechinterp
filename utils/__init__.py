"""Utility functions for NAR mechanistic interpretability."""

from .hooks import HookManager, ActivationStore, ActivationPatcher, create_activation_hooks
from .visualization import (
    plot_attention_patterns,
    plot_activation_heatmap,
    plot_circuit_graph,
    plot_edge_importance,
    plot_training_curves,
    create_interactive_circuit,
    save_circuit_to_json,
    load_circuit_from_json,
)
from .correlation import pearson_correlation_matrix, find_monosemantic_features
from .evaluation import evaluate_sae, compute_concept_analysis
from .clrs_metrics import evaluate_outputs, mask_f1, pointer_accuracy, EVAL_FN

__all__ = [
    "HookManager",
    "ActivationStore",
    "ActivationPatcher",
    "create_activation_hooks",
    "plot_attention_patterns",
    "plot_activation_heatmap",
    "plot_circuit_graph",
    "plot_edge_importance",
    "plot_training_curves",
    "create_interactive_circuit",
    "save_circuit_to_json",
    "load_circuit_from_json",
    "pearson_correlation_matrix",
    "find_monosemantic_features",
    "evaluate_sae",
    "compute_concept_analysis",
    "evaluate_outputs",
    "mask_f1",
    "pointer_accuracy",
    "EVAL_FN",
]
