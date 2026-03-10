"""
Activation collector for NAR models.

Collects processor hidden states from forward passes over CLRS data,
yielding flat activation tensors suitable for SAE training. Each
(node, message-passing step) pair is treated as an independent sample.
"""

from typing import Optional, Iterator

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data.clrs_dataset import CLRSBatch, AlgorithmSpec, batch_to_model_inputs, pyg_to_dense


class ActivationCollector:
    """
    Collects intermediate activations from a NAR model's processor.

    Runs the model in eval mode over a dataset, extracts the node
    embeddings at each processor step, and stores them as a flat
    tensor of shape (num_samples * num_nodes * num_steps, hidden_dim).

    Usage:
        collector = ActivationCollector(model)
        activations = collector.collect(dataloader, num_batches=100)
        # activations.shape = (N, hidden_dim)
    """

    def __init__(
        self,
        model: nn.Module,
        spec: Optional[AlgorithmSpec] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.spec = spec
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def collect(
        self,
        dataloader,
        num_batches: Optional[int] = None,
        output_types: Optional[dict] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Collect processor activations from the dataloader.

        Args:
            dataloader: DataLoader yielding CLRSBatch or PyG batch objects.
            num_batches: Max number of batches to process. None = all.
            output_types: Output types dict for the model. Defaults to empty.
            num_steps: Number of processor steps per forward pass.
                If None, uses batch.lengths.max() for CLRSBatch, else 1.

        Returns:
            Tensor of shape (total_samples, hidden_dim) where each row is
            a node embedding from one processor step.
        """
        self.model.eval()
        all_activations = []

        if output_types is None:
            output_types = {}

        for i, batch in enumerate(dataloader):
            if num_batches is not None and i >= num_batches:
                break

            acts = self._extract_from_batch(batch, output_types, num_steps)
            if acts is not None:
                all_activations.append(acts.cpu())

        if not all_activations:
            raise RuntimeError("No activations collected. Check model and data compatibility.")

        return torch.cat(all_activations, dim=0)

    def _extract_from_batch(
        self,
        batch,
        output_types: dict,
        num_steps: Optional[int],
    ) -> Optional[torch.Tensor]:
        """Extract node activations from a single batch."""
        # Convert batch to the format the NAR model expects
        if isinstance(batch, CLRSBatch):
            if self.spec is not None:
                inputs, _, _ = batch_to_model_inputs(batch, self.spec, self.device)
            else:
                inputs = self._clrs_batch_to_model_inputs_fallback(batch)
            if num_steps is None:
                num_steps = batch.lengths.max().item()
        elif isinstance(batch, dict):
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}
            if num_steps is None:
                num_steps = 1
        else:
            # Try PyG data object — convert to dense
            inputs = self._pyg_to_model_inputs(batch)
            if num_steps is None:
                num_steps = 1

        # Run forward pass with activation collection
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output = self.model(
                inputs=inputs,
                output_types=output_types,
                num_steps=num_steps,
                return_activations=True,
            )

        activations = output.activations
        if not activations or 'node_features' not in activations:
            return None

        # node_features is a list of tensors, one per step
        # Each has shape (batch_size, num_nodes, hidden_dim)
        step_acts = activations['node_features']
        if not step_acts:
            return None

        # Stack steps and flatten: (steps, batch, nodes, hidden) -> (steps*batch*nodes, hidden)
        stacked = torch.stack(step_acts, dim=0)  # (steps, batch, nodes, hidden)
        flat = stacked.reshape(-1, stacked.shape[-1])
        return flat

    def _clrs_batch_to_model_inputs_fallback(self, batch: CLRSBatch) -> dict:
        """Fallback conversion when no spec is available."""
        num_graphs = batch.num_graphs
        batch_vec = batch.batch.to(self.device)
        edge_index = batch.edge_index.to(self.device)

        nodes_per_graph = torch.bincount(batch_vec)
        max_nodes = nodes_per_graph.max().item()

        # Build adjacency
        adj = torch.zeros(num_graphs, max_nodes, max_nodes, device=self.device)
        node_offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=self.device)
        node_offsets[1:] = nodes_per_graph.cumsum(0)
        for g in range(num_graphs):
            mask = batch_vec[edge_index[0]] == g
            src = edge_index[0, mask] - node_offsets[g]
            tgt = edge_index[1, mask] - node_offsets[g]
            adj[g, src, tgt] = 1.0

        # Build node features
        node_feat_list = []
        for name in batch.input_fields:
            if name in batch.node_data:
                val = batch.node_data[name].to(self.device)
                if val.dim() == 1:
                    val = val.unsqueeze(-1)
                padded = torch.zeros(num_graphs, max_nodes, *val.shape[1:], device=self.device)
                for g in range(num_graphs):
                    n = nodes_per_graph[g].item()
                    start = node_offsets[g].item()
                    padded[g, :n] = val[start:start + n]
                node_feat_list.append(padded)

        if node_feat_list:
            node_features = torch.cat(node_feat_list, dim=-1)
        else:
            node_features = torch.zeros(num_graphs, max_nodes, 1, device=self.device)

        return {
            'adjacency': adj,
            'node_features': node_features,
        }

    def _pyg_to_model_inputs(self, data) -> dict:
        """Convert a single PyG data object to model inputs."""
        dense = pyg_to_dense(data)
        return {k: v.unsqueeze(0).to(self.device) if v.dim() < 3 else v.to(self.device)
                for k, v in dense.items()}

    @torch.no_grad()
    def collect_streaming(
        self,
        dataloader,
        output_types: Optional[dict] = None,
        num_steps: Optional[int] = None,
    ) -> Iterator[torch.Tensor]:
        """
        Yield activation batches one at a time (memory-efficient).

        Each yielded tensor has shape (batch_nodes * num_steps, hidden_dim).
        """
        self.model.eval()
        if output_types is None:
            output_types = {}

        for batch in dataloader:
            acts = self._extract_from_batch(batch, output_types, num_steps)
            if acts is not None:
                yield acts.cpu()


def make_activation_dataloader(
    activations: torch.Tensor,
    batch_size: int = 256,
    shuffle: bool = True,
) -> DataLoader:
    """
    Wrap collected activations in a DataLoader for SAE training.

    Args:
        activations: Tensor of shape (N, hidden_dim).
        batch_size: Training batch size.
        shuffle: Whether to shuffle.

    Returns:
        DataLoader yielding (activation_batch,) tuples.
    """
    dataset = TensorDataset(activations)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
