"""
CLRS-30 Dataset Loader using salsa-clrs (PyTorch implementation)

This module provides a clean interface to load CLRS-30 algorithmic reasoning data
for training and interpretability experiments.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum

# Import from salsa-clrs (PyTorch CLRS implementation)
try:
    from salsa_clrs import clrs
    from salsa_clrs.clrs import specs
    SALSA_AVAILABLE = True
except ImportError:
    SALSA_AVAILABLE = False
    print("Warning: salsa-clrs not installed. Using mock data for development.")


class Algorithm(Enum):
    """CLRS-30 algorithm identifiers."""
    # Sorting
    INSERTION_SORT = "insertion_sort"
    BUBBLE_SORT = "bubble_sort"
    HEAPSORT = "heapsort"
    QUICKSORT = "quicksort"
    MERGE_SORT = "merge_sort"
    
    # Searching
    BINARY_SEARCH = "binary_search"
    LINEAR_SEARCH = "linear_search"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    QUICKSELECT = "quickselect"
    
    # Graphs - Traversal
    BFS = "bfs"
    DFS = "dfs"
    TOPOLOGICAL_SORT = "topological_sort"
    ARTICULATION_POINTS = "articulation_points"
    BRIDGES = "bridges"
    STRONGLY_CONNECTED = "strongly_connected_components"
    
    # Graphs - Shortest Paths
    DIJKSTRA = "dijkstra"
    BELLMAN_FORD = "bellman_ford"
    FLOYD_WARSHALL = "floyd_warshall"
    DAG_SHORTEST_PATHS = "dag_shortest_paths"
    
    # Graphs - MST
    MST_PRIM = "mst_prim"
    MST_KRUSKAL = "mst_kruskal"
    
    # Dynamic Programming
    LCS_LENGTH = "lcs_length"
    OPTIMAL_BST = "optimal_bst"
    ACTIVITY_SELECTOR = "activity_selector"
    MATRIX_CHAIN_ORDER = "matrix_chain_order"
    TASK_SCHEDULING = "task_scheduling"
    
    # Strings
    KMP = "naive_string_matcher"
    
    # Geometry
    SEGMENTS_INTERSECT = "segments_intersect"
    GRAHAM_SCAN = "graham_scan"
    JARVIS_MARCH = "jarvis_march"


@dataclass
class CLRSBatch:
    """Batch of CLRS algorithmic reasoning data."""
    # Inputs
    inputs: Dict[str, torch.Tensor]
    # Hints (intermediate algorithm states)
    hints: Dict[str, torch.Tensor]
    # Outputs (final algorithm results)
    outputs: Dict[str, torch.Tensor]
    # Lengths of sequences (for masking)
    lengths: torch.Tensor
    # Algorithm identifier
    algorithm: str
    # Problem metadata
    metadata: Dict[str, Any]
    
    def to(self, device: torch.device) -> 'CLRSBatch':
        """Move batch to device."""
        return CLRSBatch(
            inputs={k: v.to(device) for k, v in self.inputs.items()},
            hints={k: v.to(device) for k, v in self.hints.items()},
            outputs={k: v.to(device) for k, v in self.outputs.items()},
            lengths=self.lengths.to(device),
            algorithm=self.algorithm,
            metadata=self.metadata
        )


class CLRSDataset(Dataset):
    """PyTorch Dataset wrapper for CLRS-30 data."""
    
    def __init__(
        self,
        algorithm: str,
        split: str = "train",
        num_samples: int = 1000,
        lengths: List[int] = [16],
        seed: int = 42,
    ):
        """
        Initialize CLRS dataset.
        
        Args:
            algorithm: Algorithm name (e.g., "bfs", "dijkstra")
            split: Data split ("train", "val", "test")
            num_samples: Number of samples to generate
            lengths: Problem sizes to sample from
            seed: Random seed for reproducibility
        """
        self.algorithm = algorithm
        self.split = split
        self.num_samples = num_samples
        self.lengths = lengths
        self.seed = seed
        
        # Load or generate data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict]:
        """Load CLRS data using salsa-clrs or generate mock data."""
        if SALSA_AVAILABLE:
            return self._load_salsa_data()
        else:
            return self._generate_mock_data()
    
    def _load_salsa_data(self) -> List[Dict]:
        """Load data from salsa-clrs."""
        np.random.seed(self.seed)
        
        # Get algorithm sampler from CLRS
        sampler, spec = clrs.build_sampler(
            self.algorithm,
            seed=self.seed,
            num_samples=self.num_samples,
            length=self.lengths[0] if len(self.lengths) == 1 else max(self.lengths),
        )
        
        data = []
        for _ in range(self.num_samples):
            # Sample a problem instance
            feedback = sampler.next(batch_size=1)
            
            # Convert to PyTorch tensors
            sample = {
                'inputs': {k: torch.tensor(v.data) for k, v in feedback.features.inputs.items()},
                'hints': {k: torch.tensor(v.data) for k, v in feedback.features.hints.items()},
                'outputs': {k: torch.tensor(v.data) for k, v in feedback.outputs.items()},
                'length': self.lengths[np.random.randint(len(self.lengths))],
            }
            data.append(sample)
        
        return data
    
    def _generate_mock_data(self) -> List[Dict]:
        """Generate mock data for development when salsa-clrs is not available."""
        np.random.seed(self.seed)
        data = []
        
        for i in range(self.num_samples):
            length = self.lengths[i % len(self.lengths)]
            
            # Generate mock graph data (common to many algorithms)
            num_nodes = length
            num_edges = min(length * 2, length * (length - 1))
            
            # Node features
            node_features = torch.randn(num_nodes, 8)
            
            # Adjacency matrix (sparse)
            adj = torch.zeros(num_nodes, num_nodes)
            edges = torch.randperm(num_nodes * num_nodes)[:num_edges]
            for e in edges:
                i_idx, j_idx = e // num_nodes, e % num_nodes
                if i_idx != j_idx:
                    adj[i_idx, j_idx] = 1
            
            # Edge weights
            edge_weights = torch.rand(num_nodes, num_nodes) * adj
            
            # Source node (for graph traversal algorithms)
            source = torch.zeros(num_nodes)
            source[0] = 1
            
            # Mock hints (intermediate states)
            num_steps = length
            reach_hints = torch.zeros(num_steps, num_nodes)
            for t in range(num_steps):
                reach_hints[t, :min(t+1, num_nodes)] = 1
            
            # Mock output (e.g., reachability or distances)
            output = torch.randint(0, 2, (num_nodes,)).float()
            
            sample = {
                'inputs': {
                    'node_features': node_features,
                    'adjacency': adj,
                    'edge_weights': edge_weights,
                    'source': source,
                },
                'hints': {
                    'reach': reach_hints,
                    'predecessor': torch.randint(0, num_nodes, (num_steps, num_nodes)),
                },
                'outputs': {
                    'reach': output,
                    'distances': torch.rand(num_nodes) * length,
                },
                'length': length,
            }
            data.append(sample)
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]


def collate_clrs_batch(
    batch: List[Dict],
    algorithm: str
) -> CLRSBatch:
    """
    Collate function for CLRS data.
    
    Handles variable-length sequences and creates padded batches.
    """
    batch_size = len(batch)
    max_length = max(b['length'] for b in batch)
    
    # Collect all keys
    input_keys = batch[0]['inputs'].keys()
    hint_keys = batch[0]['hints'].keys()
    output_keys = batch[0]['outputs'].keys()
    
    # Initialize padded tensors
    inputs = {}
    hints = {}
    outputs = {}
    
    for key in input_keys:
        shapes = [b['inputs'][key].shape for b in batch]
        max_shape = tuple(max(s[i] for s in shapes) for i in range(len(shapes[0])))
        padded = torch.zeros(batch_size, *max_shape)
        for i, b in enumerate(batch):
            slices = tuple(slice(0, s) for s in b['inputs'][key].shape)
            padded[(i,) + slices] = b['inputs'][key]
        inputs[key] = padded
    
    for key in hint_keys:
        shapes = [b['hints'][key].shape for b in batch]
        max_shape = tuple(max(s[i] for s in shapes) for i in range(len(shapes[0])))
        padded = torch.zeros(batch_size, *max_shape)
        for i, b in enumerate(batch):
            slices = tuple(slice(0, s) for s in b['hints'][key].shape)
            padded[(i,) + slices] = b['hints'][key]
        hints[key] = padded
    
    for key in output_keys:
        shapes = [b['outputs'][key].shape for b in batch]
        max_shape = tuple(max(s[i] for s in shapes) for i in range(len(shapes[0])))
        padded = torch.zeros(batch_size, *max_shape)
        for i, b in enumerate(batch):
            slices = tuple(slice(0, s) for s in b['outputs'][key].shape)
            padded[(i,) + slices] = b['outputs'][key]
        outputs[key] = padded
    
    lengths = torch.tensor([b['length'] for b in batch])
    
    return CLRSBatch(
        inputs=inputs,
        hints=hints,
        outputs=outputs,
        lengths=lengths,
        algorithm=algorithm,
        metadata={'batch_size': batch_size, 'max_length': max_length}
    )


def get_clrs_dataloader(
    algorithm: str,
    split: str = "train",
    batch_size: int = 32,
    num_samples: int = 1000,
    lengths: List[int] = [16],
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for CLRS-30 data.
    
    Args:
        algorithm: Algorithm name
        split: Data split
        batch_size: Batch size
        num_samples: Number of samples
        lengths: Problem sizes
        seed: Random seed
        num_workers: DataLoader workers
        pin_memory: Pin memory for GPU
        
    Returns:
        DataLoader for CLRS data
    """
    dataset = CLRSDataset(
        algorithm=algorithm,
        split=split,
        num_samples=num_samples,
        lengths=lengths,
        seed=seed,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda b: collate_clrs_batch(b, algorithm),
    )


def get_algorithm_spec(algorithm: str) -> Dict:
    """
    Get specification for an algorithm including input/output types.
    
    Returns dict with:
        - input_types: Dict mapping input name to type
        - hint_types: Dict mapping hint name to type
        - output_types: Dict mapping output name to type
        - category: Algorithm category (e.g., "graphs", "sorting")
    """
    # Algorithm specifications based on CLRS-30
    ALGORITHM_SPECS = {
        "bfs": {
            "input_types": {
                "adjacency": "graph",
                "source": "node_pointer",
            },
            "hint_types": {
                "reach": "node_mask",
                "predecessor": "node_pointer",
            },
            "output_types": {
                "reach": "node_mask",
                "predecessor": "node_pointer",
            },
            "category": "graphs",
        },
        "dfs": {
            "input_types": {
                "adjacency": "graph",
                "source": "node_pointer",
            },
            "hint_types": {
                "color": "node_categorical",
                "predecessor": "node_pointer",
                "time": "node_scalar",
            },
            "output_types": {
                "predecessor": "node_pointer",
                "discovery_time": "node_scalar",
                "finish_time": "node_scalar",
            },
            "category": "graphs",
        },
        "dijkstra": {
            "input_types": {
                "adjacency": "graph",
                "edge_weights": "edge_scalar",
                "source": "node_pointer",
            },
            "hint_types": {
                "in_queue": "node_mask",
                "distance": "node_scalar",
                "predecessor": "node_pointer",
            },
            "output_types": {
                "distance": "node_scalar",
                "predecessor": "node_pointer",
            },
            "category": "graphs",
        },
        "bellman_ford": {
            "input_types": {
                "adjacency": "graph",
                "edge_weights": "edge_scalar",
                "source": "node_pointer",
            },
            "hint_types": {
                "distance": "node_scalar",
                "predecessor": "node_pointer",
            },
            "output_types": {
                "distance": "node_scalar",
                "predecessor": "node_pointer",
            },
            "category": "graphs",
        },
        "insertion_sort": {
            "input_types": {
                "array": "array_scalar",
            },
            "hint_types": {
                "key": "scalar",
                "sorted_prefix": "array_mask",
            },
            "output_types": {
                "sorted_array": "array_scalar",
            },
            "category": "sorting",
        },
        "bubble_sort": {
            "input_types": {
                "array": "array_scalar",
            },
            "hint_types": {
                "swapped": "scalar",
                "current_array": "array_scalar",
            },
            "output_types": {
                "sorted_array": "array_scalar",
            },
            "category": "sorting",
        },
        "heapsort": {
            "input_types": {
                "array": "array_scalar",
            },
            "hint_types": {
                "heap": "array_scalar",
                "heap_size": "scalar",
            },
            "output_types": {
                "sorted_array": "array_scalar",
            },
            "category": "sorting",
        },
    }
    
    return ALGORITHM_SPECS.get(algorithm, {
        "input_types": {},
        "hint_types": {},
        "output_types": {},
        "category": "unknown",
    })
