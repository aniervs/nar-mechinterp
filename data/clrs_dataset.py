"""
CLRS-30 Dataset Loader using salsa-clrs (PyTorch/PyG implementation).

This module wraps the SALSACLRSDataset from the salsa-clrs package,
providing utilities for loading CLRS-30 algorithmic reasoning data
in PyTorch Geometric format.

Data format:
    Each sample is a PyG Data object (CLRSData) with:
    - edge_index: (2, num_edges) graph structure
    - Node-level fields: shape (num_nodes,) or (num_nodes, num_steps) for hints
    - Edge-level fields: shape (num_edges,) or (num_edges, num_steps) for hints
    - inputs/hints/outputs: lists of field names belonging to each stage
    - length: number of algorithm steps
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import DataLoader
import salsaclrs
from salsaclrs.data import SALSACLRSDataset, CLRSData
from torch_geometric.data.storage import GlobalStorage

# Whitelist salsaclrs/PyG classes so torch.load works with weights_only=True
# DataEdgeAttr/DataTensorAttr are dynamically generated in some PyG versions
_safe_globals = [CLRSData, GlobalStorage]
try:
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    _safe_globals.extend([DataEdgeAttr, DataTensorAttr])
except ImportError:
    pass
torch.serialization.add_safe_globals(_safe_globals)


# Algorithms available in salsa-clrs
AVAILABLE_ALGORITHMS = ['bfs', 'dfs', 'dijkstra', 'mst_prim', 'fast_mis', 'eccentricity']


@dataclass
class AlgorithmSpec:
    """Specification for a CLRS algorithm's data fields."""
    input_fields: List[str]
    hint_fields: List[str]
    output_fields: List[str]
    specs: Dict[str, tuple] = field(default_factory=dict)


def get_clrs_dataset(
    algorithm: str,
    split: str = "train",
    num_samples: int = 1000,
    num_nodes: int = 16,
    edge_probability: float = 0.2,
    data_dir: Optional[str] = None,
    seed: int = 42,
) -> SALSACLRSDataset:
    """
    Load a CLRS-30 dataset using salsa-clrs.

    Args:
        algorithm: Algorithm name (one of AVAILABLE_ALGORITHMS).
        split: Data split ("train", "val", "test").
        num_samples: Number of problem instances to generate.
        num_nodes: Number of nodes per graph.
        edge_probability: Edge probability for Erdos-Renyi graph generator.
        data_dir: Root directory for cached data. Defaults to ./clrs_data.
        seed: Random seed for data generation.

    Returns:
        SALSACLRSDataset (a PyG InMemoryDataset).
    """
    if algorithm not in AVAILABLE_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Available: {AVAILABLE_ALGORITHMS}"
        )

    if data_dir is None:
        data_dir = os.path.join(os.getcwd(), "clrs_data")

    dataset = SALSACLRSDataset(
        root=data_dir,
        split=split,
        algorithm=algorithm,
        num_samples=num_samples,
        seed=seed,
        graph_generator="er",
        graph_generator_kwargs={"n": num_nodes, "p": edge_probability},
    )

    return dataset


@dataclass
class CLRSBatch:
    """A batched collection of CLRS graph data with padded hints."""
    edge_index: torch.Tensor       # (2, total_edges) concatenated edge indices
    batch: torch.Tensor            # (total_nodes,) graph assignment for each node
    num_graphs: int
    lengths: torch.Tensor          # (batch_size,) algorithm steps per graph
    node_data: Dict[str, torch.Tensor]   # field_name -> (total_nodes,) or (total_nodes, max_steps)
    edge_data: Dict[str, torch.Tensor]   # field_name -> (total_edges,) or (total_edges, max_steps)
    input_fields: List[str]
    hint_fields: List[str]
    output_fields: List[str]
    specs: Dict[str, tuple] = field(default_factory=dict)

    def to(self, device: torch.device) -> 'CLRSBatch':
        return CLRSBatch(
            edge_index=self.edge_index.to(device),
            batch=self.batch.to(device),
            num_graphs=self.num_graphs,
            lengths=self.lengths.to(device),
            node_data={k: v.to(device) for k, v in self.node_data.items()},
            edge_data={k: v.to(device) for k, v in self.edge_data.items()},
            input_fields=self.input_fields,
            hint_fields=self.hint_fields,
            output_fields=self.output_fields,
            specs=self.specs,
        )


def _collate_clrs_batch(data_list) -> CLRSBatch:
    """Collate CLRSData objects into a batch, padding hints to max time steps."""
    if len(data_list) == 0:
        raise ValueError("Empty batch")

    # Determine max time steps across the batch
    max_steps = max(d.length.item() for d in data_list)

    # Gather field metadata from first item
    first = data_list[0]
    input_fields = list(first.inputs) if hasattr(first, 'inputs') else []
    hint_fields = list(first.hints) if hasattr(first, 'hints') else []
    output_fields = list(first.outputs) if hasattr(first, 'outputs') else []
    all_fields = input_fields + hint_fields + output_fields

    # Identify which fields are node-level vs edge-level
    node_fields = {}
    edge_fields = {}
    for name in all_fields:
        val = first[name]
        if val.shape[0] == first.num_nodes:
            node_fields[name] = val
        elif val.shape[0] == first.num_edges:
            edge_fields[name] = val

    # Build batch by concatenating nodes/edges and padding time dims
    edge_indices = []
    batch_vec = []
    lengths = []
    node_data = {name: [] for name in node_fields}
    edge_data = {name: [] for name in edge_fields}
    node_offset = 0

    for i, data in enumerate(data_list):
        n = data.num_nodes
        e = data.num_edges
        steps = data.length.item()

        # Offset edge indices
        edge_indices.append(data.edge_index + node_offset)
        batch_vec.append(torch.full((n,), i, dtype=torch.long))
        lengths.append(steps)

        for name in node_fields:
            val = data[name]
            if val.dim() == 2 and val.shape[1] < max_steps:
                # Pad time dimension
                pad = torch.zeros(n, max_steps - val.shape[1], dtype=val.dtype)
                val = torch.cat([val, pad], dim=1)
            node_data[name].append(val)

        for name in edge_fields:
            val = data[name]
            if val.dim() == 2 and val.shape[1] < max_steps:
                pad = torch.zeros(e, max_steps - val.shape[1], dtype=val.dtype)
                val = torch.cat([val, pad], dim=1)
            edge_data[name].append(val)

        node_offset += n

    return CLRSBatch(
        edge_index=torch.cat(edge_indices, dim=1),
        batch=torch.cat(batch_vec),
        num_graphs=len(data_list),
        lengths=torch.tensor(lengths, dtype=torch.long),
        node_data={k: torch.cat(v, dim=0) for k, v in node_data.items()},
        edge_data={k: torch.cat(v, dim=0) for k, v in edge_data.items()},
        input_fields=input_fields,
        hint_fields=hint_fields,
        output_fields=output_fields,
    )


def get_clrs_dataloader(
    algorithm: str,
    split: str = "train",
    batch_size: int = 32,
    num_samples: int = 1000,
    num_nodes: int = 16,
    edge_probability: float = 0.2,
    data_dir: Optional[str] = None,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """
    Create a DataLoader for CLRS-30 data with custom collation.

    Args:
        algorithm: Algorithm name.
        split: Data split.
        batch_size: Batch size.
        num_samples: Number of samples.
        num_nodes: Number of nodes per graph.
        edge_probability: Edge probability for ER graphs.
        data_dir: Data cache directory.
        seed: Random seed.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for GPU transfer.
        shuffle: Whether to shuffle. Defaults to True for train split.

    Returns:
        PyG DataLoader yielding batched CLRSData objects.
    """
    dataset = get_clrs_dataset(
        algorithm=algorithm,
        split=split,
        num_samples=num_samples,
        num_nodes=num_nodes,
        edge_probability=edge_probability,
        data_dir=data_dir,
        seed=seed,
    )

    if shuffle is None:
        shuffle = (split == "train")

    # Hints have variable time steps across samples, so we need to pad them.
    # Use a standard DataLoader with a custom collate that pads the time dimension.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate_clrs_batch,
    )


def get_algorithm_spec(algorithm: str, dataset: Optional[SALSACLRSDataset] = None) -> AlgorithmSpec:
    """
    Get the specification for an algorithm's data fields.

    If a dataset is provided, reads specs directly from it.
    Otherwise returns specs from a hardcoded table.

    Args:
        algorithm: Algorithm name.
        dataset: Optional dataset to read specs from.

    Returns:
        AlgorithmSpec with input/hint/output field names and type info.
    """
    if dataset is not None and hasattr(dataset, 'specs'):
        specs = dataset.specs
        inputs = [k for k, v in specs.items() if v[0] == 'input']
        hints = [k for k, v in specs.items() if v[0] == 'hint']
        outputs = [k for k, v in specs.items() if v[0] == 'output']
        return AlgorithmSpec(
            input_fields=inputs,
            hint_fields=hints,
            output_fields=outputs,
            specs=specs,
        )

    # Hardcoded specs for common algorithms (stage, location, type, dim)
    SPECS = {
        "bfs": {
            'pos': ('input', 'node', 'scalar', None),
            's': ('input', 'node', 'mask_one', None),
            'pi': ('output', 'node', 'pointer', None),
            'reach_h': ('hint', 'node', 'mask', None),
            'pi_h': ('hint', 'node', 'pointer', None),
        },
        "dfs": {
            'pos': ('input', 'node', 'scalar', None),
            's': ('input', 'node', 'mask_one', None),
            'pi': ('output', 'node', 'pointer', None),
            'color': ('hint', 'node', 'categorical', None),
            'pi_h': ('hint', 'node', 'pointer', None),
            'topo_h': ('hint', 'node', 'mask', None),
        },
        "dijkstra": {
            'pos': ('input', 'node', 'scalar', None),
            's': ('input', 'node', 'mask_one', None),
            'pi': ('output', 'node', 'pointer', None),
            'pi_h': ('hint', 'node', 'pointer', None),
            'd': ('hint', 'node', 'scalar', None),
            'mark': ('hint', 'node', 'mask', None),
            'in_queue': ('hint', 'node', 'mask', None),
            'u': ('hint', 'node', 'mask_one', None),
        },
        "mst_prim": {
            'pos': ('input', 'node', 'scalar', None),
            's': ('input', 'node', 'mask_one', None),
            'pi': ('output', 'node', 'pointer', None),
            'pi_h': ('hint', 'node', 'pointer', None),
            'key': ('hint', 'node', 'scalar', None),
            'mark': ('hint', 'node', 'mask', None),
            'in_queue': ('hint', 'node', 'mask', None),
            'u': ('hint', 'node', 'mask_one', None),
        },
    }

    algo_specs = SPECS.get(algorithm, {})
    inputs = [k for k, v in algo_specs.items() if v[0] == 'input']
    hints = [k for k, v in algo_specs.items() if v[0] == 'hint']
    outputs = [k for k, v in algo_specs.items() if v[0] == 'output']
    return AlgorithmSpec(
        input_fields=inputs,
        hint_fields=hints,
        output_fields=outputs,
        specs=algo_specs,
    )


def spec_to_model_types(spec: AlgorithmSpec) -> tuple[Dict[str, str], Dict[str, str]]:
    """
    Convert an AlgorithmSpec to output_types and hint_types dicts
    in the format the NAR model expects.

    Maps salsaclrs type names to model type names:
        pointer -> node_pointer, mask -> node_mask, mask_one -> node_mask,
        scalar -> node_scalar, categorical -> node_categorical

    Returns:
        (output_types, hint_types) dicts mapping field_name -> model_type_str.
    """
    TYPE_MAP = {
        ('node', 'pointer'): 'node_pointer',
        ('node', 'mask'): 'node_mask',
        ('node', 'mask_one'): 'node_mask',
        ('node', 'scalar'): 'node_scalar',
        ('node', 'categorical'): 'node_categorical',
        ('edge', 'pointer'): 'edge_pointer',
        ('edge', 'mask'): 'edge_mask',
        ('edge', 'scalar'): 'edge_scalar',
    }

    output_types = {}
    hint_types = {}

    for name, (stage, location, dtype, _) in spec.specs.items():
        model_type = TYPE_MAP.get((location, dtype), f"{location}_{dtype}")
        if stage == 'output':
            output_types[name] = model_type
        elif stage == 'hint':
            hint_types[name] = model_type

    return output_types, hint_types


def batch_to_model_inputs(
    batch: 'CLRSBatch',
    spec: AlgorithmSpec,
    device: Optional[torch.device] = None,
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Convert a CLRSBatch to the dense format the NAR model expects.

    Returns:
        (inputs, outputs, hints) — each a dict of tensors shaped
        (batch_size, num_nodes, ...) for node-level data.

    For pointer fields stored on edges, converts to dense
    (batch_size, num_nodes) integer index tensors.
    """
    if device is None:
        device = batch.edge_index.device

    edge_index = batch.edge_index.to(device)
    batch_vec = batch.batch.to(device)
    num_graphs = batch.num_graphs
    nodes_per_graph = torch.bincount(batch_vec)
    max_nodes = nodes_per_graph.max().item()

    # --- Build adjacency matrices ---
    adj = torch.zeros(num_graphs, max_nodes, max_nodes, device=device)
    node_offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
    node_offsets[1:] = nodes_per_graph.cumsum(0)

    for g in range(num_graphs):
        mask = batch_vec[edge_index[0]] == g
        src = edge_index[0, mask] - node_offsets[g]
        tgt = edge_index[1, mask] - node_offsets[g]
        adj[g, src, tgt] = 1.0

    # --- Helper: pad concatenated node data to (batch, max_nodes, ...) ---
    def _pad_node_field(flat_tensor):
        extra_dims = flat_tensor.shape[1:] if flat_tensor.dim() > 1 else ()
        padded = torch.zeros(num_graphs, max_nodes, *extra_dims,
                             dtype=flat_tensor.dtype, device=device)
        for g in range(num_graphs):
            n = nodes_per_graph[g].item()
            start = node_offsets[g].item()
            padded[g, :n] = flat_tensor[start:start + n]
        return padded

    # --- Helper: decode edge-level pointer to node-level index tensor ---
    def _decode_edge_pointer(edge_vals, step_dim=False):
        """Convert edge-stored pointer to (batch, max_nodes) or (batch, max_nodes, steps) indices."""
        if step_dim and edge_vals.dim() == 2:
            # edge_vals: (total_edges, steps)
            num_steps = edge_vals.shape[1]
            result = torch.zeros(num_graphs, max_nodes, num_steps,
                                 dtype=torch.long, device=device)
            for g in range(num_graphs):
                n = nodes_per_graph[g].item()
                mask = batch_vec[edge_index[0]] == g
                src = edge_index[0, mask] - node_offsets[g]
                tgt = edge_index[1, mask] - node_offsets[g]
                ev = edge_vals[mask].to(device)  # (edges_in_g, steps)
                for t in range(num_steps):
                    for node in range(n):
                        node_mask = src == node
                        if node_mask.any():
                            result[g, node, t] = tgt[node_mask][ev[node_mask, t].argmax()]
            return result
        else:
            result = torch.zeros(num_graphs, max_nodes, dtype=torch.long, device=device)
            for g in range(num_graphs):
                n = nodes_per_graph[g].item()
                mask = batch_vec[edge_index[0]] == g
                src = edge_index[0, mask] - node_offsets[g]
                tgt = edge_index[1, mask] - node_offsets[g]
                ev = edge_vals[mask].to(device)
                for node in range(n):
                    node_mask = src == node
                    if node_mask.any():
                        result[g, node] = tgt[node_mask][ev[node_mask].argmax()]
            return result

    # --- Build inputs dict ---
    inputs = {'adjacency': adj}

    # Concatenate node-level input features
    node_feat_list = []
    for name in batch.input_fields:
        if name in batch.node_data:
            val = batch.node_data[name].to(device)
            if val.dim() == 1:
                val = val.unsqueeze(-1)
            node_feat_list.append(_pad_node_field(val))

    if node_feat_list:
        inputs['node_features'] = torch.cat(node_feat_list, dim=-1)
    else:
        inputs['node_features'] = torch.zeros(num_graphs, max_nodes, 1, device=device)

    # Edge weights if present
    if 'weights' in batch.edge_data:
        weight_matrix = torch.zeros(num_graphs, max_nodes, max_nodes, device=device)
        weights = batch.edge_data['weights'].to(device)
        for g in range(num_graphs):
            mask = batch_vec[edge_index[0]] == g
            src = edge_index[0, mask] - node_offsets[g]
            tgt = edge_index[1, mask] - node_offsets[g]
            weight_matrix[g, src, tgt] = weights[mask].float()
        inputs['edge_weights'] = weight_matrix

    # --- Build outputs dict ---
    outputs = {}
    for name in batch.output_fields:
        field_spec = spec.specs.get(name, ('output', 'node', 'scalar', None))
        _, location, dtype, _ = field_spec

        if name in batch.node_data:
            outputs[name] = _pad_node_field(batch.node_data[name].to(device))
        elif name in batch.edge_data:
            if dtype == 'pointer':
                outputs[name] = _decode_edge_pointer(batch.edge_data[name].to(device))
            else:
                # Edge-level non-pointer outputs — store as edge attribute
                outputs[name] = batch.edge_data[name].to(device)

    # --- Build hints dict (padded to max_steps) ---
    hints = {}
    for name in batch.hint_fields:
        field_spec = spec.specs.get(name, ('hint', 'node', 'scalar', None))
        _, location, dtype, _ = field_spec

        if name in batch.node_data:
            # Node hint: (total_nodes, max_steps) -> (batch, max_nodes, max_steps)
            hints[name] = _pad_node_field(batch.node_data[name].to(device))
        elif name in batch.edge_data:
            if dtype == 'pointer':
                hints[name] = _decode_edge_pointer(
                    batch.edge_data[name].to(device), step_dim=True
                )
            else:
                hints[name] = batch.edge_data[name].to(device)

    return inputs, outputs, hints


def pyg_to_dense(data, num_nodes: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """
    Convert a PyG Data object to dense tensor format.

    Useful for models that expect adjacency matrices rather than edge_index.

    Args:
        data: PyG Data object from salsaclrs.
        num_nodes: Number of nodes. Inferred from data if not provided.

    Returns:
        Dict with 'adjacency' (num_nodes, num_nodes), 'node_features' (num_nodes, d),
        plus any other node-level fields.
    """
    if num_nodes is None:
        num_nodes = data.num_nodes

    # Build adjacency matrix from edge_index
    edge_index = data.edge_index
    adj = torch.zeros(num_nodes, num_nodes)
    adj[edge_index[0], edge_index[1]] = 1.0

    # Collect node-level input fields
    input_names = data.inputs if hasattr(data, 'inputs') else []
    node_features = []
    for name in input_names:
        val = data[name]
        if val.dim() == 1 and val.shape[0] == num_nodes:
            node_features.append(val.unsqueeze(-1))

    if node_features:
        node_feat = torch.cat(node_features, dim=-1)
    else:
        node_feat = torch.zeros(num_nodes, 1)

    result = {
        'adjacency': adj,
        'node_features': node_feat,
    }

    # Include edge weights if present
    if hasattr(data, 'weights'):
        weight_matrix = torch.zeros(num_nodes, num_nodes)
        weight_matrix[edge_index[0], edge_index[1]] = data.weights.float()
        result['edge_weights'] = weight_matrix

    return result
