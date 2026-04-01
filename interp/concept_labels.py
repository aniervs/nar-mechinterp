"""
Concept label extractor for CLRS algorithmic reasoning data.

Extracts binary/continuous concept labels from algorithm hints and aligns
them with collected activations. Each (node, step) pair gets a set of
concept labels, matching the activation collector's flattening convention.

Supported concepts per algorithm:

BFS:
    - is_visited: node reached by this step (reach_h)
    - is_frontier: node just became visited this step
    - is_source: source node indicator (s)

Dijkstra:
    - is_settled: node permanently labeled (mark)
    - is_in_queue: node in priority queue (in_queue)
    - is_current: node being processed this step (u)
    - distance: current distance estimate (d)

DFS:
    - is_visited: node discovered (color > 0)
    - is_active: node on the stack / being explored (color == 1, "gray")
    - is_finished: node fully explored (color == 2, "black")

MST (Prim):
    - is_in_tree: node added to MST (mark)
    - is_in_queue: node in candidate queue (in_queue)
    - is_current: node being processed (u)
    - key_value: current key/priority (key)
"""

from dataclasses import dataclass, field
from typing import Optional

import torch

from data.clrs_dataset import (
    CLRSBatch,
    AlgorithmSpec,
    batch_to_model_inputs,
)


@dataclass
class ConceptLabels:
    """Concept labels aligned with flattened activations."""
    labels: dict[str, torch.Tensor]  # concept_name -> (N,) tensor
    num_samples: int                  # total (node, step) pairs
    algorithm: str
    concept_descriptions: dict[str, str] = field(default_factory=dict)


def extract_concept_labels(
    batch: CLRSBatch,
    spec: AlgorithmSpec,
    algorithm: str,
) -> ConceptLabels:
    """
    Extract concept labels from a CLRSBatch, flattened to match
    the activation collector's output convention.

    The activation collector flattens as:
        for step in steps:
            for graph in batch:
                for node in nodes:
                    -> one activation vector

    So we flatten hints the same way: (steps, batch, nodes) -> (N,).

    Args:
        batch: A CLRSBatch from the dataloader.
        spec: AlgorithmSpec for the algorithm.
        algorithm: Algorithm name (determines which concepts to extract).

    Returns:
        ConceptLabels with flattened label tensors.
    """
    # Convert batch to dense format to get padded hint tensors
    _, _, hints = batch_to_model_inputs(batch, spec)
    num_graphs = batch.num_graphs
    batch_vec = batch.batch
    nodes_per_graph = torch.bincount(batch_vec)
    max_nodes = nodes_per_graph.max().item()
    max_steps = batch.lengths.max().item()

    # Source node indicator (same for all steps)
    inputs, _, _ = batch_to_model_inputs(batch, spec)
    source = _extract_source(inputs, batch, max_nodes)

    # Extract algorithm-specific concepts
    _EXTRACTORS = {
        "bfs": _extract_bfs_concepts,
        "dijkstra": _extract_dijkstra_concepts,
        "dfs": _extract_dfs_concepts,
        "mst_prim": _extract_mst_prim_concepts,
        "bellman_ford": _extract_bellman_ford_concepts,
        "mst_kruskal": _extract_mst_kruskal_concepts,
        "articulation_points": _extract_articulation_concepts,
        "bridges": _extract_bridges_concepts,
        "fast_mis": _extract_fast_mis_concepts,
        "eccentricity": _extract_eccentricity_concepts,
    }

    extractor = _EXTRACTORS.get(algorithm)
    if extractor is not None:
        labels, descriptions = extractor(hints, source, max_steps)
    else:
        # Generic: just flatten whatever hints are available
        labels, descriptions = _extract_generic_concepts(hints, max_steps)

    # Flatten all labels: (steps, batch, nodes) -> (N,)
    flat_labels = {}
    num_samples = 0
    for name, tensor in labels.items():
        flat = _flatten_step_batch_node(tensor, max_steps)
        flat_labels[name] = flat
        num_samples = flat.shape[0]

    return ConceptLabels(
        labels=flat_labels,
        num_samples=num_samples,
        algorithm=algorithm,
        concept_descriptions=descriptions,
    )


def _extract_source(inputs: dict, batch: CLRSBatch, max_nodes: int) -> torch.Tensor:
    """Extract source node indicator as (batch, max_nodes) binary tensor."""
    node_feats = inputs.get('node_features')
    if node_feats is None:
        return torch.zeros(batch.num_graphs, max_nodes)

    # The 's' field (mask_one) is usually the second input feature
    # It's encoded as one column in node_features
    # Try to find it from batch data directly
    if 's' in batch.node_data:
        s_flat = batch.node_data['s']
        batch_vec = batch.batch
        nodes_per_graph = torch.bincount(batch_vec)
        node_offsets = torch.zeros(batch.num_graphs + 1, dtype=torch.long)
        node_offsets[1:] = nodes_per_graph.cumsum(0)
        source = torch.zeros(batch.num_graphs, max_nodes)
        for g in range(batch.num_graphs):
            n = nodes_per_graph[g].item()
            start = node_offsets[g].item()
            source[g, :n] = s_flat[start:start + n]
        return source

    return torch.zeros(batch.num_graphs, max_nodes)


def _flatten_step_batch_node(tensor: torch.Tensor, max_steps: int) -> torch.Tensor:
    """
    Flatten a (batch, nodes, steps) or (batch, nodes) tensor to match
    the activation collector convention: iterate steps outer, then batch, then nodes.

    Output shape: (max_steps * batch * nodes,) for time-varying concepts,
    or (max_steps * batch * nodes,) repeating for static concepts.
    """
    if tensor.dim() == 3:
        # (batch, nodes, steps) -> (steps, batch, nodes) -> flat
        t = tensor.permute(2, 0, 1)  # (steps, batch, nodes)
        # Only take up to max_steps
        t = t[:max_steps]
        return t.reshape(-1).float()
    elif tensor.dim() == 2:
        # (batch, nodes) — static concept, repeat for each step
        t = tensor.unsqueeze(0).expand(max_steps, -1, -1)  # (steps, batch, nodes)
        return t.reshape(-1).float()
    else:
        return tensor.reshape(-1).float()


def _extract_bfs_concepts(
    hints: dict, source: torch.Tensor, max_steps: int
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Extract BFS-specific concepts from hints."""
    labels = {}
    descriptions = {}

    # is_source: static across steps
    labels['is_source'] = source
    descriptions['is_source'] = 'Source node of BFS traversal'

    if 'reach_h' in hints:
        reach = hints['reach_h']  # (batch, nodes, steps)

        # is_visited: reach_h == 1 at this step
        labels['is_visited'] = reach
        descriptions['is_visited'] = 'Node has been visited by BFS at this step'

        # is_frontier: visited at step t but not at step t-1
        if reach.dim() == 3 and reach.shape[2] > 1:
            frontier = torch.zeros_like(reach)
            frontier[:, :, 0] = reach[:, :, 0]  # Step 0: frontier = initial visited
            frontier[:, :, 1:] = reach[:, :, 1:] * (1 - reach[:, :, :-1])
            labels['is_frontier'] = frontier
            descriptions['is_frontier'] = 'Node just entered the BFS frontier this step'

        # is_unvisited: NOT visited
        labels['is_unvisited'] = 1.0 - reach
        descriptions['is_unvisited'] = 'Node has not yet been reached by BFS'

    return labels, descriptions


def _extract_dijkstra_concepts(
    hints: dict, source: torch.Tensor, max_steps: int
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Extract Dijkstra-specific concepts from hints."""
    labels = {}
    descriptions = {}

    labels['is_source'] = source
    descriptions['is_source'] = 'Source node of Dijkstra'

    if 'mark' in hints:
        labels['is_settled'] = hints['mark']
        descriptions['is_settled'] = 'Node has been permanently settled (shortest path found)'

    if 'in_queue' in hints:
        labels['is_in_queue'] = hints['in_queue']
        descriptions['is_in_queue'] = 'Node is currently in the priority queue'

    if 'u' in hints:
        labels['is_current'] = hints['u']
        descriptions['is_current'] = 'Node is currently being processed (extracted from queue)'

    if 'd' in hints:
        labels['distance_estimate'] = hints['d']
        descriptions['distance_estimate'] = 'Current shortest distance estimate to this node'

    # Derived: just_settled = settled at t but not at t-1
    if 'mark' in hints and hints['mark'].dim() == 3 and hints['mark'].shape[2] > 1:
        mark = hints['mark']
        just_settled = torch.zeros_like(mark)
        just_settled[:, :, 0] = mark[:, :, 0]
        just_settled[:, :, 1:] = mark[:, :, 1:] * (1 - mark[:, :, :-1])
        labels['just_settled'] = just_settled
        descriptions['just_settled'] = 'Node was just settled this step (relaxation complete)'

    return labels, descriptions


def _extract_dfs_concepts(
    hints: dict, source: torch.Tensor, max_steps: int
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Extract DFS-specific concepts from hints."""
    labels = {}
    descriptions = {}

    labels['is_source'] = source
    descriptions['is_source'] = 'Source node of DFS'

    if 'color' in hints:
        color = hints['color']  # categorical: 0=white, 1=gray, 2=black
        labels['is_visited'] = (color > 0).float()
        descriptions['is_visited'] = 'Node has been discovered (gray or black)'

        labels['is_active'] = (color == 1).float()
        descriptions['is_active'] = 'Node is on the DFS stack (gray, being explored)'

        labels['is_finished'] = (color == 2).float()
        descriptions['is_finished'] = 'Node is fully explored (black, all neighbors visited)'

    return labels, descriptions


def _extract_mst_prim_concepts(
    hints: dict, source: torch.Tensor, max_steps: int
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Extract Prim's MST-specific concepts from hints."""
    labels = {}
    descriptions = {}

    labels['is_source'] = source
    descriptions['is_source'] = 'Source/root node of MST'

    if 'mark' in hints:
        labels['is_in_tree'] = hints['mark']
        descriptions['is_in_tree'] = 'Node has been added to the MST'

    if 'in_queue' in hints:
        labels['is_in_queue'] = hints['in_queue']
        descriptions['is_in_queue'] = 'Node is a candidate for MST inclusion'

    if 'u' in hints:
        labels['is_current'] = hints['u']
        descriptions['is_current'] = 'Node being added to MST this step'

    if 'key' in hints:
        labels['key_value'] = hints['key']
        descriptions['key_value'] = 'Current minimum edge weight connecting node to MST'

    return labels, descriptions


def _extract_bellman_ford_concepts(
    hints: dict, source: torch.Tensor, max_steps: int
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Extract Bellman-Ford shortest-path concepts from hints."""
    labels = {}
    descriptions = {}

    labels['is_source'] = source
    descriptions['is_source'] = 'Source node of Bellman-Ford'

    if 'msk' in hints:
        labels['is_relaxed'] = hints['msk']
        descriptions['is_relaxed'] = 'Node has been relaxed (distance updated)'

    if 'd' in hints:
        labels['distance_estimate'] = hints['d']
        descriptions['distance_estimate'] = 'Current shortest distance estimate'

    # Derived: just_relaxed = relaxed at t but not at t-1
    if 'msk' in hints and hints['msk'].dim() == 3 and hints['msk'].shape[2] > 1:
        msk = hints['msk']
        just_relaxed = torch.zeros_like(msk)
        just_relaxed[:, :, 0] = msk[:, :, 0]
        just_relaxed[:, :, 1:] = msk[:, :, 1:] * (1 - msk[:, :, :-1])
        labels['just_relaxed'] = just_relaxed
        descriptions['just_relaxed'] = 'Node was just relaxed this step'

    return labels, descriptions


def _extract_mst_kruskal_concepts(
    hints: dict, source: torch.Tensor, max_steps: int
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Extract Kruskal MST concepts from hints."""
    labels = {}
    descriptions = {}

    if 'mask_u' in hints:
        labels['in_component_u'] = hints['mask_u']
        descriptions['in_component_u'] = 'Node belongs to the component of endpoint u'

    if 'mask_v' in hints:
        labels['in_component_v'] = hints['mask_v']
        descriptions['in_component_v'] = 'Node belongs to the component of endpoint v'

    if 'u' in hints:
        labels['is_u'] = hints['u']
        descriptions['is_u'] = 'Node is the u-endpoint of the current edge'

    if 'v' in hints:
        labels['is_v'] = hints['v']
        descriptions['is_v'] = 'Node is the v-endpoint of the current edge'

    if 'root_u' in hints:
        labels['is_root_u'] = hints['root_u']
        descriptions['is_root_u'] = 'Node is the root of component containing u'

    if 'root_v' in hints:
        labels['is_root_v'] = hints['root_v']
        descriptions['is_root_v'] = 'Node is the root of component containing v'

    return labels, descriptions


def _extract_articulation_concepts(
    hints: dict, source: torch.Tensor, max_steps: int
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Extract articulation point detection concepts from hints."""
    labels = {}
    descriptions = {}

    if 'is_cut_h' in hints:
        labels['is_cut'] = hints['is_cut_h']
        descriptions['is_cut'] = 'Node is identified as an articulation point so far'

    if 'color' in hints:
        color = hints['color']
        labels['is_visited'] = (color > 0).float()
        descriptions['is_visited'] = 'Node has been discovered (gray or black)'
        labels['is_active'] = (color == 1).float()
        descriptions['is_active'] = 'Node is on the DFS stack (gray)'
        labels['is_finished'] = (color == 2).float()
        descriptions['is_finished'] = 'Node is fully explored (black)'

    if 'low' in hints:
        labels['low_value'] = hints['low']
        descriptions['low_value'] = 'Lowest discovery time reachable from subtree'

    if 'd' in hints:
        labels['discovery_time'] = hints['d']
        descriptions['discovery_time'] = 'DFS discovery time'

    return labels, descriptions


def _extract_bridges_concepts(
    hints: dict, source: torch.Tensor, max_steps: int
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Extract bridge detection concepts from hints."""
    labels = {}
    descriptions = {}

    if 'color' in hints:
        color = hints['color']
        labels['is_visited'] = (color > 0).float()
        descriptions['is_visited'] = 'Node has been discovered'
        labels['is_active'] = (color == 1).float()
        descriptions['is_active'] = 'Node is on the DFS stack (gray)'
        labels['is_finished'] = (color == 2).float()
        descriptions['is_finished'] = 'Node is fully explored (black)'

    if 'low' in hints:
        labels['low_value'] = hints['low']
        descriptions['low_value'] = 'Lowest discovery time reachable from subtree'

    if 'd' in hints:
        labels['discovery_time'] = hints['d']
        descriptions['discovery_time'] = 'DFS discovery time'

    return labels, descriptions


def _extract_fast_mis_concepts(
    hints: dict, source: torch.Tensor, max_steps: int
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Extract Fast MIS (maximal independent set) concepts from hints."""
    labels = {}
    descriptions = {}

    if 'alive_h' in hints:
        labels['is_alive'] = hints['alive_h']
        descriptions['is_alive'] = 'Node is still a candidate (not yet decided)'

    if 'inmis_h' in hints:
        labels['in_mis'] = hints['inmis_h']
        descriptions['in_mis'] = 'Node has been added to the independent set'

    if 'phase_h' in hints:
        labels['in_phase'] = hints['phase_h']
        descriptions['in_phase'] = 'Node is participating in the current phase'

    # Derived: just_added = in MIS at t but not at t-1
    if 'inmis_h' in hints and hints['inmis_h'].dim() == 3 and hints['inmis_h'].shape[2] > 1:
        inmis = hints['inmis_h']
        just_added = torch.zeros_like(inmis)
        just_added[:, :, 0] = inmis[:, :, 0]
        just_added[:, :, 1:] = inmis[:, :, 1:] * (1 - inmis[:, :, :-1])
        labels['just_added_to_mis'] = just_added
        descriptions['just_added_to_mis'] = 'Node was just added to MIS this step'

    return labels, descriptions


def _extract_eccentricity_concepts(
    hints: dict, source: torch.Tensor, max_steps: int
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Extract eccentricity computation concepts from hints."""
    labels = {}
    descriptions = {}

    labels['is_source'] = source
    descriptions['is_source'] = 'Source node for eccentricity computation'

    if 'visited_h' in hints:
        labels['is_visited'] = hints['visited_h']
        descriptions['is_visited'] = 'Node has been visited in BFS flood'

    if 'eccentricity_h' in hints:
        labels['eccentricity'] = hints['eccentricity_h']
        descriptions['eccentricity'] = 'Current eccentricity estimate'

    return labels, descriptions


def _extract_generic_concepts(
    hints: dict, max_steps: int
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Extract concepts generically from any hint fields."""
    labels = {}
    descriptions = {}

    for name, tensor in hints.items():
        labels[name] = tensor
        descriptions[name] = f'Hint field: {name}'

    return labels, descriptions


def collect_concept_labels(
    dataloader,
    spec: AlgorithmSpec,
    algorithm: str,
    num_batches: Optional[int] = None,
) -> ConceptLabels:
    """
    Collect concept labels across an entire dataloader.

    Args:
        dataloader: DataLoader yielding CLRSBatch objects.
        spec: AlgorithmSpec for the algorithm.
        algorithm: Algorithm name.
        num_batches: Max batches to process. None = all.

    Returns:
        ConceptLabels with concatenated labels across all batches.
    """
    all_labels: dict[str, list[torch.Tensor]] = {}
    total_samples = 0
    descriptions = {}

    for i, batch in enumerate(dataloader):
        if num_batches is not None and i >= num_batches:
            break

        result = extract_concept_labels(batch, spec, algorithm)

        for name, tensor in result.labels.items():
            if name not in all_labels:
                all_labels[name] = []
            all_labels[name].append(tensor)

        total_samples += result.num_samples
        descriptions.update(result.concept_descriptions)

    concat_labels = {name: torch.cat(tensors, dim=0)
                     for name, tensors in all_labels.items()}

    return ConceptLabels(
        labels=concat_labels,
        num_samples=total_samples,
        algorithm=algorithm,
        concept_descriptions=descriptions,
    )
