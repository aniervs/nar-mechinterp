"""Multi-algorithm dataloader for training a shared-processor NAR on several
CLRS-30 algorithms simultaneously.

Each yielded item is a tuple:
    (batch, algorithm_name, output_types, hint_types)

so the training loop can pass the correct types to the model per batch.
"""

import random
from typing import Optional

from torch.utils.data import DataLoader

from .clrs_dataset import (
    AlgorithmSpec,
    CLRSBatch,
    get_clrs_dataloader,
    get_algorithm_spec,
    spec_to_model_types,
)


class MultiAlgorithmLoader:
    """Round-robin dataloader over multiple CLRS algorithms.

    For each algorithm a separate ``DataLoader`` is created. Iteration
    yields batches from all algorithms in round-robin order (one full
    pass over each algorithm per epoch).

    Args:
        algorithms: List of algorithm names.
        split: Data split ("train", "val", "test").
        batch_size: Batch size per algorithm.
        num_samples: Samples per algorithm.
        num_nodes: Number of graph nodes.
        edge_probability: ER edge probability.
        data_dir: Data cache directory.
        seed: Base random seed (each algorithm gets ``seed + i``).
        shuffle: Shuffle within each algorithm loader.
    """

    def __init__(
        self,
        algorithms: list[str],
        split: str = "train",
        batch_size: int = 32,
        num_samples: int = 1000,
        num_nodes: int = 16,
        edge_probability: float = 0.2,
        data_dir: Optional[str] = None,
        seed: int = 42,
        shuffle: Optional[bool] = None,
    ):
        if not algorithms:
            raise ValueError("algorithms list cannot be empty")
        self.algorithms = algorithms
        self.loaders: dict[str, DataLoader] = {}
        self.specs: dict[str, AlgorithmSpec] = {}
        self.output_types: dict[str, dict] = {}
        self.hint_types: dict[str, dict] = {}

        for i, algo in enumerate(algorithms):
            loader = get_clrs_dataloader(
                algorithm=algo,
                split=split,
                batch_size=batch_size,
                num_samples=num_samples,
                num_nodes=num_nodes,
                edge_probability=edge_probability,
                data_dir=data_dir,
                seed=seed + i,
                shuffle=shuffle,
            )
            self.loaders[algo] = loader
            spec = get_algorithm_spec(algo)
            self.specs[algo] = spec
            ot, ht = spec_to_model_types(spec)
            self.output_types[algo] = ot
            self.hint_types[algo] = ht

    def __iter__(self):
        """Yield (batch, algo_name, output_types, hint_types) round-robin."""
        iterators = {algo: iter(loader) for algo, loader in self.loaders.items()}
        active = set(self.algorithms)

        while active:
            for algo in list(active):
                try:
                    batch = next(iterators[algo])
                except StopIteration:
                    active.discard(algo)
                    continue
                yield batch, algo, self.output_types[algo], self.hint_types[algo]

    def __len__(self):
        """Total number of batches across all algorithms."""
        return sum(len(loader) for loader in self.loaders.values())

    @property
    def num_algorithms(self) -> int:
        return len(self.algorithms)
