"""
NAR Mechanistic Interpretability

A toolkit for mechanistic interpretability research on Neural Algorithmic Reasoning
models trained on CLRS-30.

Modules:
    data: CLRS-30 dataset loading and processing
    models: Neural Algorithmic Reasoning model architectures
    interp: Interpretability tools (ACDC, activation patching, circuits)
    utils: Hooks, visualization, and utility functions
    experiments: Training and analysis scripts
"""

__version__ = "0.1.0"

# Subpackages are imported directly: `from models import NARModel`, etc.
# This file exists to mark the root as a Python package for relative
# imports within subpackages.
