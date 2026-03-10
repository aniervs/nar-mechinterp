# NAR Mechanistic Interpretability

Mechanistic interpretability research on Neural Algorithmic Reasoning (NAR) models trained on the CLRS-30 benchmark, using Automatic Circuit Discovery (ACDC) techniques.

## Overview

This codebase provides tools to:
1. **Train** NAR models on algorithmic reasoning tasks from CLRS-30
2. **Discover** minimal circuits using ACDC (Automatic Circuit Discovery)
3. **Analyze** and compare circuits across different algorithms
4. **Visualize** attention patterns, activations, and circuit structures

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync
```

To run scripts and tests, use `uv run`:

```bash
uv run python experiments/train_nar.py --algorithm bfs
uv run python experiments/run_acdc.py --checkpoint checkpoints/bfs/best.pt --algorithm bfs
uv run python -m pytest
```

### Requirements
- Python 3.13+
- PyTorch 2.0+
- NetworkX (for circuit visualization)
- Matplotlib, Plotly (visualization)
- salsa-clrs (CLRS-30 dataset) - optional, mock data available

## Project Structure

```
nar_mechinterp/
├── configs/
│   └── default_config.yaml      # Training and experiment configuration
├── data/
│   └── clrs_dataset.py          # CLRS-30 dataset loader with mock fallback
├── models/
│   ├── nar_model.py             # Full NAR model (Encoder-Processor-Decoder)
│   └── processor.py             # Message passing and transformer processors
├── interp/
│   ├── acdc.py                  # ACDC algorithm implementation
│   ├── activation_patching.py  # Activation patching experiments
│   ├── circuit.py               # Circuit data structures and analysis
│   └── metrics.py               # Interpretability metrics
├── utils/
│   ├── hooks.py                 # PyTorch hook utilities for activation collection
│   └── visualization.py         # Circuit and attention visualization
├── experiments/
│   ├── train_nar.py             # Training script
│   ├── run_acdc.py              # Circuit discovery script
│   └── analyze_circuits.py      # Circuit analysis and comparison
└── requirements.txt
```

## Quick Start

### 1. Train a NAR Model

```bash
# Train on BFS algorithm
python experiments/train_nar.py --algorithm bfs --epochs 100

# Train on Dijkstra's algorithm with custom parameters
python experiments/train_nar.py --algorithm dijkstra --hidden_dim 256 --num_layers 6

# Available algorithms: bfs, dfs, dijkstra, bellman_ford, insertion_sort, 
#                       bubble_sort, heapsort, quicksort, and more
```

### 2. Run Circuit Discovery

```bash
# Discover circuit for BFS
python experiments/run_acdc.py \
    --checkpoint checkpoints/bfs/best.pt \
    --algorithm bfs \
    --threshold 0.01 \
    --save_plots

# With different ablation strategy
python experiments/run_acdc.py \
    --checkpoint checkpoints/dijkstra/best.pt \
    --algorithm dijkstra \
    --ablation_type resample \
    --threshold 0.005
```

### 3. Analyze and Compare Circuits

```bash
# Analyze individual circuits
python experiments/analyze_circuits.py \
    --circuits circuits/bfs_circuit.json circuits/dfs_circuit.json \
    --compare \
    --find_shared \
    --save_plots

# Analyze all circuits in a directory
python experiments/analyze_circuits.py \
    --circuit_dir circuits/ \
    --compare \
    --merge \
    --output_dir analysis/
```

## Model Architecture

The NAR model follows the Encoder-Processor-Decoder architecture:

### Encoder
- Embeds various input types (scalars, node features, edges, pointers, graphs)
- Separate embedding layers for each input type
- Projects to hidden dimension

### Processor
- **MPNN (default)**: Message Passing Neural Network
  - Node attention mechanism
  - Edge update MLP
  - Message aggregation
  - Node update with gating
- **Transformer**: Standard transformer with self-attention
- Supports multi-step reasoning (hint supervision)

### Decoder
- Produces algorithm-specific outputs
- Supports: node masks, pointers, scalars, edge predictions

## ACDC Algorithm

The ACDC (Automatic Circuit Discovery) implementation:

1. **Build computational graph** from model structure
2. **Score edges** using activation patching
3. **Iteratively prune** low-importance edges
4. **Verify fidelity** of resulting circuit

### Ablation Types
- `mean`: Replace activations with dataset mean
- `zero`: Replace with zeros
- `resample`: Replace with activations from different inputs

### Metrics
- `kl_divergence`: KL divergence between patched and clean outputs
- `mse`: Mean squared error
- `accuracy`: Task accuracy difference

## Circuit Analysis

### Comparing Circuits
```python
from interp import Circuit, compare_circuits

bfs_circuit = Circuit.load("circuits/bfs_circuit.json")
dfs_circuit = Circuit.load("circuits/dfs_circuit.json")

comparison = compare_circuits(bfs_circuit, dfs_circuit, "BFS", "DFS")
print(f"Similarity: {comparison['overall_similarity']:.4f}")
```

### Finding Shared Components
```python
from interp import find_shared_subcircuit

circuits = [bfs_circuit, dfs_circuit, dijkstra_circuit]
shared = find_shared_subcircuit(circuits)
print(f"Shared nodes: {shared.num_nodes}")
```

## Visualization

```python
from utils import plot_circuit_graph, plot_attention_patterns

# Visualize circuit
plot_circuit_graph(
    circuit.to_dict(),
    title="BFS Circuit",
    save_path="bfs_circuit.png"
)

# Visualize attention patterns
plot_attention_patterns(
    attention_weights,
    layer_idx=2,
    save_path="attention_layer2.png"
)
```

## Configuration

Edit `configs/default_config.yaml`:

```yaml
model:
  hidden_dim: 128
  num_layers: 4
  num_heads: 8
  processor_type: mpnn
  use_gating: true

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  scheduler: cosine

acdc:
  threshold: 0.01
  metric: kl_divergence
  ablation_type: mean
  max_iterations: 100
```

## Key Research Questions

This codebase enables investigation of:

1. **Circuit Specialization**: Do different algorithms use distinct circuits?
2. **Shared Computation**: What components are shared across algorithms?
3. **Algorithm Families**: Do similar algorithms (BFS/DFS, sorting variants) share circuits?
4. **Generalization**: How do circuits change with problem size?
5. **Emergent Structure**: Do circuits reflect algorithmic structure?

## References

- CLRS-30 Benchmark: [Deepmind CLRS](https://github.com/deepmind/clrs)
- ACDC Paper: "Towards Automated Circuit Discovery for Mechanistic Interpretability" (Conmy et al., 2023)
- NAR Survey: "Neural Algorithmic Reasoning" (Veličković, 2023)

## License

MIT License
