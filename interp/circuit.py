"""
Circuit Representation and Analysis.

Defines data structures for representing discovered circuits
and utilities for analyzing and comparing them.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np


@dataclass
class CircuitNode:
    """A node in the computational circuit."""
    id: str
    type: str  # attention, mlp, message_passing, etc.
    layer: int = 0
    module: Optional[nn.Module] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'type': self.type,
            'layer': self.layer,
            'metadata': self.metadata,
        }


@dataclass
class CircuitEdge:
    """An edge (connection) in the computational circuit."""
    id: str
    source: str
    target: str
    weight: float = 1.0
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'source': self.source,
            'target': self.target,
            'weight': self.weight,
            'active': self.active,
            'metadata': self.metadata,
        }


@dataclass
class Circuit:
    """
    A computational circuit discovered through interpretability analysis.

    Represents a subgraph of the full model that is sufficient
    for performing a specific task.
    """
    nodes: List[CircuitNode] = field(default_factory=list)
    edges: List[CircuitEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self):
        """Build lookup dictionaries."""
        self._node_dict = {n.id: n for n in self.nodes}
        self._edge_dict = {e.id: e for e in self.edges}
        self._adjacency = self._build_adjacency()

    def add_node(self, id: str, type: str, layer: int = 0, component: str = "", **kwargs) -> CircuitNode:
        """Add a node to the circuit."""
        metadata = kwargs.copy()
        if component:
            metadata['component'] = component
        node = CircuitNode(id=id, type=type, layer=layer, metadata=metadata)
        self.nodes.append(node)
        self._node_dict[id] = node
        self._adjacency[id] = self._adjacency.get(id, [])
        return node

    def add_edge(self, source: str, target: str, weight: float = 1.0, edge_type: str = "", **kwargs) -> CircuitEdge:
        """Add an edge to the circuit."""
        edge_id = f"{source}->{target}"
        metadata = kwargs.copy()
        if edge_type:
            metadata['edge_type'] = edge_type
        edge = CircuitEdge(id=edge_id, source=source, target=target, weight=weight, metadata=metadata)
        self.edges.append(edge)
        self._edge_dict[edge_id] = edge
        if source in self._adjacency:
            self._adjacency[source].append(target)
        return edge

    def remove_edge(self, source: str, target: str):
        """Remove an edge by source and target."""
        self.edges = [e for e in self.edges if not (e.source == source and e.target == target)]
        edge_id = f"{source}->{target}"
        self._edge_dict.pop(edge_id, None)
        self._adjacency = self._build_adjacency()

    def remove_node(self, node_id: str):
        """Remove a node and all its incident edges."""
        self.nodes = [n for n in self.nodes if n.id != node_id]
        self._node_dict.pop(node_id, None)
        self.edges = [e for e in self.edges if e.source != node_id and e.target != node_id]
        self._edge_dict = {e.id: e for e in self.edges}
        self._adjacency = self._build_adjacency()
    
    def _build_adjacency(self) -> Dict[str, List[str]]:
        """Build adjacency list."""
        adj = {n.id: [] for n in self.nodes}
        for edge in self.edges:
            if edge.active and edge.source in adj:
                adj[edge.source].append(edge.target)
        return adj
    
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)
    
    @property
    def num_edges(self) -> int:
        return len([e for e in self.edges if e.active])
    
    @property
    def num_layers(self) -> int:
        if not self.nodes:
            return 0
        return max(n.layer for n in self.nodes) + 1
    
    def get_node(self, node_id: str) -> Optional[CircuitNode]:
        return self._node_dict.get(node_id)
    
    def get_edge(self, edge_id: str) -> Optional[CircuitEdge]:
        return self._edge_dict.get(edge_id)
    
    def get_edges_from(self, node_id: str) -> List[CircuitEdge]:
        """Get all edges originating from a node."""
        return [e for e in self.edges if e.source == node_id and e.active]
    
    def get_edges_to(self, node_id: str) -> List[CircuitEdge]:
        """Get all edges targeting a node."""
        return [e for e in self.edges if e.target == node_id and e.active]
    
    def get_nodes_by_type(self, node_type: str) -> List[CircuitNode]:
        """Get all nodes of a specific type."""
        return [n for n in self.nodes if n.type == node_type]
    
    def get_nodes_at_layer(self, layer: int) -> List[CircuitNode]:
        """Get all nodes at a specific layer."""
        return [n for n in self.nodes if n.layer == layer]
    
    def is_connected(self) -> bool:
        """Check if the circuit is connected."""
        if not self.nodes:
            return True
        
        visited = set()
        stack = [self.nodes[0].id]
        
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            
            # Add neighbors (both directions for undirected check)
            for edge in self.edges:
                if edge.active:
                    if edge.source == node and edge.target not in visited:
                        stack.append(edge.target)
                    if edge.target == node and edge.source not in visited:
                        stack.append(edge.source)
        
        return len(visited) == len(self.nodes)
    
    def get_paths(
        self,
        source: str,
        target: str,
        max_length: int = 10,
    ) -> List[List[str]]:
        """Find all paths between two nodes."""
        paths = []
        
        def dfs(current: str, path: List[str]):
            if len(path) > max_length:
                return
            if current == target:
                paths.append(path.copy())
                return
            
            for neighbor in self._adjacency.get(current, []):
                if neighbor not in path:
                    path.append(neighbor)
                    dfs(neighbor, path)
                    path.pop()
        
        dfs(source, [source])
        return paths
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute various statistics about the circuit."""
        stats = {
            'num_nodes': self.num_nodes,
            'num_edges': self.num_edges,
            'num_layers': self.num_layers,
            'is_connected': self.is_connected(),
        }
        
        # Node type distribution
        type_counts = {}
        for node in self.nodes:
            type_counts[node.type] = type_counts.get(node.type, 0) + 1
        stats['node_type_distribution'] = type_counts
        
        # Edge weight statistics
        weights = [e.weight for e in self.edges if e.active]
        if weights:
            stats['edge_weight_mean'] = np.mean(weights)
            stats['edge_weight_std'] = np.std(weights)
            stats['edge_weight_min'] = np.min(weights)
            stats['edge_weight_max'] = np.max(weights)
        
        # Degree statistics
        in_degrees = {n.id: 0 for n in self.nodes}
        out_degrees = {n.id: 0 for n in self.nodes}
        for edge in self.edges:
            if edge.active:
                if edge.source in out_degrees:
                    out_degrees[edge.source] += 1
                if edge.target in in_degrees:
                    in_degrees[edge.target] += 1
        
        stats['avg_in_degree'] = np.mean(list(in_degrees.values())) if in_degrees else 0
        stats['avg_out_degree'] = np.mean(list(out_degrees.values())) if out_degrees else 0
        
        return stats
    
    def to_dict(self) -> Dict:
        """Convert circuit to dictionary for serialization."""
        return {
            'nodes': [n.to_dict() for n in self.nodes],
            'edges': [e.to_dict() for e in self.edges],
            'metadata': self.metadata,
            'statistics': self.compute_statistics(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Circuit':
        """Create circuit from dictionary."""
        nodes = [
            CircuitNode(
                id=n['id'],
                type=n['type'],
                layer=n.get('layer', 0),
                metadata=n.get('metadata', {}),
            )
            for n in data['nodes']
        ]
        edges = [
            CircuitEdge(
                id=e['id'],
                source=e['source'],
                target=e['target'],
                weight=e.get('weight', 1.0),
                active=e.get('active', True),
                metadata=e.get('metadata', {}),
            )
            for e in data['edges']
        ]
        return cls(
            nodes=nodes,
            edges=edges,
            metadata=data.get('metadata', {}),
        )
    
    def save(self, path: str):
        """Save circuit to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'Circuit':
        """Load circuit from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def compare_circuits(
    circuit1: Circuit,
    circuit2: Circuit,
    name1: str = "Circuit 1",
    name2: str = "Circuit 2",
) -> Dict[str, Any]:
    """
    Compare two circuits and compute similarity metrics.
    
    Args:
        circuit1: First circuit
        circuit2: Second circuit
        name1: Name for first circuit
        name2: Name for second circuit
        
    Returns:
        Dict with comparison metrics
    """
    # Node comparison
    nodes1 = set(n.id for n in circuit1.nodes)
    nodes2 = set(n.id for n in circuit2.nodes)
    
    common_nodes = nodes1 & nodes2
    only_in_1 = nodes1 - nodes2
    only_in_2 = nodes2 - nodes1
    
    node_jaccard = len(common_nodes) / len(nodes1 | nodes2) if (nodes1 | nodes2) else 1.0
    
    # Edge comparison
    edges1 = set((e.source, e.target) for e in circuit1.edges if e.active)
    edges2 = set((e.source, e.target) for e in circuit2.edges if e.active)
    
    common_edges = edges1 & edges2
    only_edges_1 = edges1 - edges2
    only_edges_2 = edges2 - edges1
    
    edge_jaccard = len(common_edges) / len(edges1 | edges2) if (edges1 | edges2) else 1.0
    
    # Type distribution comparison
    types1 = {n.type for n in circuit1.nodes}
    types2 = {n.type for n in circuit2.nodes}
    
    return {
        'node_comparison': {
            'common': list(common_nodes),
            f'only_in_{name1}': list(only_in_1),
            f'only_in_{name2}': list(only_in_2),
            'jaccard_similarity': node_jaccard,
        },
        'edge_comparison': {
            'num_common': len(common_edges),
            f'num_only_in_{name1}': len(only_edges_1),
            f'num_only_in_{name2}': len(only_edges_2),
            'jaccard_similarity': edge_jaccard,
        },
        'type_comparison': {
            'common_types': list(types1 & types2),
            f'only_in_{name1}': list(types1 - types2),
            f'only_in_{name2}': list(types2 - types1),
        },
        'size_comparison': {
            f'{name1}_nodes': circuit1.num_nodes,
            f'{name2}_nodes': circuit2.num_nodes,
            f'{name1}_edges': circuit1.num_edges,
            f'{name2}_edges': circuit2.num_edges,
        },
        'overall_similarity': (node_jaccard + edge_jaccard) / 2,
    }


def merge_circuits(circuits: List[Circuit]) -> Circuit:
    """
    Merge multiple circuits into one (union of all nodes and edges).
    
    Args:
        circuits: List of circuits to merge
        
    Returns:
        Merged circuit
    """
    all_nodes = {}
    all_edges = {}
    
    for circuit in circuits:
        for node in circuit.nodes:
            if node.id not in all_nodes:
                all_nodes[node.id] = node
        
        for edge in circuit.edges:
            if edge.id not in all_edges:
                all_edges[edge.id] = edge
            elif edge.active:
                # If edge appears in multiple circuits, keep it active
                all_edges[edge.id].active = True
    
    return Circuit(
        nodes=list(all_nodes.values()),
        edges=list(all_edges.values()),
        metadata={'merged_from': len(circuits)},
    )


def find_shared_subcircuit(circuits: List[Circuit]) -> Circuit:
    """
    Find the shared subcircuit (intersection) across multiple circuits.
    
    Args:
        circuits: List of circuits
        
    Returns:
        Shared subcircuit
    """
    if not circuits:
        return Circuit(nodes=[], edges=[])
    
    # Start with first circuit
    shared_nodes = set(n.id for n in circuits[0].nodes)
    shared_edges = set((e.source, e.target) for e in circuits[0].edges if e.active)
    
    # Intersect with remaining circuits
    for circuit in circuits[1:]:
        circuit_nodes = set(n.id for n in circuit.nodes)
        circuit_edges = set((e.source, e.target) for e in circuit.edges if e.active)
        
        shared_nodes &= circuit_nodes
        shared_edges &= circuit_edges
    
    # Build shared circuit
    nodes = [n for n in circuits[0].nodes if n.id in shared_nodes]
    edges = [
        e for e in circuits[0].edges 
        if e.active and (e.source, e.target) in shared_edges
    ]
    
    return Circuit(
        nodes=nodes,
        edges=edges,
        metadata={'shared_across': len(circuits)},
    )


def create_full_circuit_from_model(model: nn.Module, name: str = "full_model") -> Circuit:
    """
    Build a full circuit representation from a PyTorch model.

    Traverses model modules and creates a circuit node for each leaf module,
    with edges connecting adjacent layers.

    Args:
        model: PyTorch model to analyze
        name: Name for the circuit

    Returns:
        Circuit representing the full model
    """
    circuit = Circuit(name=name)

    # Add input node
    circuit.add_node("input", "input", layer=0, component="input")

    # Traverse model to find leaf components
    for mod_name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue  # Skip container modules

        # Determine component type
        lower = mod_name.lower()
        if 'attention' in lower:
            node_type = "attention"
        elif 'mlp' in lower or 'linear' in lower:
            node_type = "mlp"
        elif 'message' in lower:
            node_type = "message_passing"
        elif 'node' in lower:
            node_type = "node_update"
        elif 'edge' in lower:
            node_type = "edge_update"
        else:
            node_type = "unknown"

        # Extract layer number from name
        layer = 0
        for part in mod_name.split('.'):
            if part.isdigit():
                layer = int(part) + 1
                break

        circuit.add_node(mod_name, node_type, layer=layer, component=mod_name)

    # Add output node
    circuit.add_node("output", "output", layer=100, component="output")

    # Build edges based on layer structure
    nodes_by_layer: Dict[int, List[str]] = {}
    for node in circuit.nodes:
        layer = node.layer
        if layer not in nodes_by_layer:
            nodes_by_layer[layer] = []
        nodes_by_layer[layer].append(node.id)

    sorted_layers = sorted(nodes_by_layer.keys())
    for i, layer in enumerate(sorted_layers[:-1]):
        next_layer = sorted_layers[i + 1]
        for src in nodes_by_layer[layer]:
            for tgt in nodes_by_layer[next_layer]:
                circuit.add_edge(src, tgt, weight=1.0)

    return circuit
