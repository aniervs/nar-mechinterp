"""Linear probe baseline for concept decodability from NAR activations."""

from typing import Optional

import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split


def fit_linear_probes(
    per_layer_acts: dict[int, torch.Tensor],
    concept_labels: dict[str, torch.Tensor],
    num_layers: int,
    concepts: Optional[list[str]] = None,
    n_samples: int = 20_000,
    test_size: float = 0.3,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Fit logistic regression probes per layer and concept.

    Args:
        per_layer_acts: {layer_idx: (N, hidden_dim) tensor}.
        concept_labels: {concept_name: (N,) tensor}.
        num_layers: Number of processor layers.
        concepts: Which concepts to probe. Defaults to all in concept_labels.
        n_samples: Max samples for training (subsampled for speed).
        test_size: Fraction held out for evaluation.
        seed: Random seed for train/test split.
        verbose: Print per-probe results.

    Returns:
        Dict with:
            results: {(layer, concept): {"auroc": float, "f1": float}}
            concepts: list of concept names
            num_layers: int
    """
    if concepts is None:
        concepts = list(concept_labels.keys())

    results: dict[tuple[int, str], dict] = {}

    for layer_idx in range(num_layers):
        if layer_idx not in per_layer_acts:
            continue
        acts = per_layer_acts[layer_idx]
        n = min(n_samples, acts.shape[0])
        X = acts[:n].numpy().astype(np.float32)

        for concept in concepts:
            if concept not in concept_labels:
                continue
            y = concept_labels[concept][:n].numpy().astype(np.int32)

            pos_rate = y.mean()
            if pos_rate < 0.001 or pos_rate > 0.999:
                if verbose:
                    print(f"L{layer_idx} | {concept:20s} | skipped (prevalence={pos_rate:.4f})")
                continue

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y
            )

            clf = LogisticRegression(
                class_weight="balanced", max_iter=1000, solver="lbfgs", C=1.0, n_jobs=-1
            )
            clf.fit(X_tr, y_tr)

            y_prob = clf.predict_proba(X_te)[:, 1]
            y_pred = clf.predict(X_te)

            auroc = roc_auc_score(y_te, y_prob)
            f1 = f1_score(y_te, y_pred, zero_division=0)

            results[(layer_idx, concept)] = {"auroc": auroc, "f1": f1}
            if verbose:
                print(
                    f"L{layer_idx} | {concept:20s} | AUROC={auroc:.3f} | F1={f1:.3f} | "
                    f"prevalence={pos_rate:.3f}"
                )

    return {"results": results, "concepts": concepts, "num_layers": num_layers}


def save_probe_results(probe_output: dict, path) -> None:
    """Save probe results to disk (converts tuple keys to strings)."""
    serializable = {
        str(k): {"auroc": v["auroc"], "f1": v["f1"]}
        for k, v in probe_output["results"].items()
    }
    torch.save(
        {"results": serializable, "concepts": probe_output["concepts"],
         "num_layers": probe_output["num_layers"]},
        path,
    )


def load_probe_results(path) -> dict:
    """Load probe results from disk (restores tuple keys)."""
    raw = torch.load(path, weights_only=False)
    results = {eval(k): v for k, v in raw["results"].items()}
    return {"results": results, "concepts": raw["concepts"], "num_layers": raw["num_layers"]}
