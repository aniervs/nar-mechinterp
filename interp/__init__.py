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
    create_full_circuit_from_model,
)
from .metrics import (
    compute_faithfulness,
    compute_minimality,
    compute_specificity,
    compute_all_metrics,
    CircuitEvaluator,
    kl_divergence,
    edge_importance_score,
)
from .sae import SparseAutoencoder, SAEOutput, SAETrainer
from .activation_collector import ActivationCollector, make_activation_dataloader
from .feature_analysis import FeatureAnalyzer, FeatureAnalysisResult, FeatureStats
from .concept_labels import ConceptLabels, extract_concept_labels, collect_concept_labels

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
    "create_full_circuit_from_model",
    # Metrics
    "compute_faithfulness",
    "compute_minimality",
    "compute_specificity",
    "compute_all_metrics",
    "CircuitEvaluator",
    "kl_divergence",
    "edge_importance_score",
    # SAE
    "SparseAutoencoder",
    "SAEOutput",
    "SAETrainer",
    # Activation collector
    "ActivationCollector",
    "make_activation_dataloader",
    # Feature analysis
    "FeatureAnalyzer",
    "FeatureAnalysisResult",
    "FeatureStats",
    # Concept labels
    "ConceptLabels",
    "extract_concept_labels",
    "collect_concept_labels",
]
