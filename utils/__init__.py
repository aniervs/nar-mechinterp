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
]
