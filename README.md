# NAR Mechanistic Interpretability

Mechanistic interpretability of Neural Algorithmic Reasoning (NAR) models using Sparse Autoencoders (SAEs). We train NAR models on CLRS-30 algorithmic tasks, then use SAEs to extract interpretable features from processor activations and correlate them with ground-truth algorithmic concepts (e.g., "is this node visited?", "is this node on the BFS frontier?").

## Overview

1. **Train** an NAR model on a CLRS-30 algorithm (BFS, DFS, Dijkstra, etc.)
2. **Collect** intermediate processor activations across all message-passing steps
3. **Train a Sparse Autoencoder** (BatchTopK SAE) on these activations
4. **Correlate** SAE features with algorithmic concept labels extracted from hints
5. **Analyze** which features are monosemantic (cleanly map to one concept)

## Quick Start

### Installation

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync
```

### Run Experiments

The main experiment pipeline is in a single notebook:

```bash
# Run locally
uv run jupyter lab experiments/run_experiments.ipynb
```

Or open directly in **Google Colab** — the notebook includes a setup cell that clones the repo and installs dependencies. It also mounts Google Drive for persistent checkpoints.

Set `LOCAL_DEBUG = True` in the config cell for a quick sanity check on CPU (~2 min), or `False` for full-scale GPU training.

### Run Tests

```bash
uv run python -m pytest
```

## Project Structure

```
nar-mechinterp/
├── data/
│   └── clrs_dataset.py            # CLRS-30 data loading via salsa-clrs
├── models/
│   ├── nar_model.py               # NAR model (Encoder-Processor-Decoder)
│   └── processor.py               # MPNN and Transformer processors
├── interp/
│   ├── sae.py                     # SAE variants (Standard, BatchTopK, Transcoder)
│   ├── activation_collector.py    # Collect processor activations per (node, step)
│   ├── concept_labels.py          # Extract algorithmic concept labels from hints
│   └── feature_analysis.py        # Feature-concept correlation analysis
├── experiments/
│   └── run_experiments.ipynb       # End-to-end experiment notebook
└── tests/                          # Unit and integration tests
```

## Architecture

### NAR Model

Encoder-Processor-Decoder architecture following Velickovic et al.:

- **Encoder**: Embeds node features, edge weights, and graph structure
- **Processor**: Multi-step MPNN with attention and gating (configurable: MPNN or Transformer)
- **Decoder**: Produces algorithm-specific outputs (pointers, masks, scalars)
- **Hint supervision**: Intermediate processor steps are supervised with algorithmic hints

### Sparse Autoencoders

Three SAE variants for analyzing processor activations:

- **SparseAutoencoder**: Standard L1-penalized SAE
- **BatchTopKSAE** (recommended): Sparsity via global top-k selection across the batch — no activation shrinkage, direct sparsity control
- **Transcoder**: Maps processor input to output through sparse features (for circuit analysis)

### Concept Labels

Ground-truth algorithmic concepts extracted from CLRS hints:

| Algorithm | Concepts |
|-----------|----------|
| BFS | `is_source`, `is_visited`, `is_frontier` |
| DFS | `is_source`, `is_visited`, `is_active`, `is_finished` |
| Dijkstra | `is_source`, `is_settled`, `is_in_queue`, `is_current`, `distance_estimate` |
| Prim's MST | `is_source`, `is_in_tree`, `is_in_queue`, `is_current`, `key_value` |

## Requirements

- Python >= 3.11
- PyTorch 2.0+
- salsa-clrs (installed from git automatically)

## References

- CLRS-30 Benchmark: [Velickovic et al., 2022](https://arxiv.org/abs/2205.15659)
- SALSA-CLRS: [Minder et al., 2024](https://github.com/jkminder/SALSA-CLRS)
- BatchTopK SAEs: [Bussmann et al., 2024](https://arxiv.org/abs/2412.06410)
- Neural Algorithmic Reasoning: [Velickovic, 2023](https://arxiv.org/abs/2105.02761)

## License

MIT License
