"""Interpretability modules for NAR mechanistic interpretability."""

from .activation_patching import (
    ActivationPatchingExperiment,
    PatchingResult,
    PathPatching,
    compute_direct_effect,
)
from .acdc import ACDC, ACDCConfig, ACDCResult, run_acdc_multi_algorithm
from .circuit import (
    Circuit,
    CircuitNode,
    CircuitEdge,
    compare_circuits,
    merge_circuits,
    find_shared_subcircuit,
)
from .metrics import (
    compute_faithfulness,
    compute_minimality,
    compute_specificity,
    compute_all_metrics,
    CircuitEvaluator,
)

__all__ = [
    # Activation patching
    "ActivationPatchingExperiment",
    "PatchingResult",
    "PathPatching",
    "compute_direct_effect",
    # ACDC
    "ACDC",
    "ACDCConfig",
    "ACDCResult",
    "run_acdc_multi_algorithm",
    # Circuit
    "Circuit",
    "CircuitNode",
    "CircuitEdge",
    "compare_circuits",
    "merge_circuits",
    "find_shared_subcircuit",
    # Metrics
    "compute_faithfulness",
    "compute_minimality",
    "compute_specificity",
    "compute_all_metrics",
    "CircuitEvaluator",
]
