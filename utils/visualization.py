"""
Visualization Utilities for Mechanistic Interpretability.

Provides functions for:
- Circuit visualization as graphs
- Attention pattern visualization
- Activation heatmaps
- Training curves
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple, Any, Union
import json
from pathlib import Path

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def plot_attention_patterns(
    attention_weights: torch.Tensor,
    layer_idx: int = 0,
    head_idx: Optional[int] = None,
    title: str = "Attention Patterns",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """Visualize attention patterns."""
    weights = attention_weights.detach().cpu().numpy()
    
    if weights.ndim == 4:
        weights = weights[0]
    if weights.ndim == 3:
        num_heads = weights.shape[0]
    else:
        num_heads = 1
        weights = weights[np.newaxis, ...]
    
    if head_idx is not None:
        weights = weights[head_idx:head_idx+1]
        num_heads = 1
    
    ncols = min(4, num_heads)
    nrows = (num_heads + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    
    for i in range(num_heads):
        row, col = i // ncols, i % ncols
        ax = axes[row, col]
        im = ax.imshow(weights[i], cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title(f'Head {i if head_idx is None else head_idx}')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    for i in range(num_heads, nrows * ncols):
        axes[i // ncols, i % ncols].axis('off')
    
    fig.suptitle(f'{title} - Layer {layer_idx}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_activation_heatmap(
    activations: torch.Tensor,
    title: str = "Activations",
    xlabel: str = "Hidden Dimension",
    ylabel: str = "Position",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Plot activation heatmap."""
    acts = activations.detach().cpu().numpy()
    if acts.ndim == 3:
        acts = acts[0]
    
    fig, ax = plt.subplots(figsize=figsize)
    vmax = np.abs(acts).max()
    im = ax.imshow(acts, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Activation')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_circuit_graph(
    circuit: Dict[str, Any],
    title: str = "Discovered Circuit",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """Visualize a circuit as a graph."""
    if not HAS_NETWORKX:
        raise ImportError("networkx required")
    
    G = nx.DiGraph()
    for node in circuit.get('nodes', []):
        G.add_node(node['id'], **node)
    for edge in circuit.get('edges', []):
        G.add_edge(edge['source'], edge['target'], weight=edge.get('weight', 1.0))
    
    # Hierarchical layout
    layers = {}
    for node in G.nodes():
        layer = G.nodes[node].get('layer', 0)
        layers.setdefault(layer, []).append(node)
    
    pos = {}
    for layer, nodes in sorted(layers.items()):
        for i, node in enumerate(nodes):
            pos[node] = (layer, -i)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    type_colors = {
        'attention': '#FF6B6B', 'mlp': '#4ECDC4', 'message_passing': '#45B7D1',
        'node_update': '#96CEB4', 'edge_update': '#FFEAA7', 'input': '#DFE6E9',
        'output': '#6C5CE7', 'unknown': '#B2BEC3',
    }
    
    node_colors = [type_colors.get(G.nodes[n].get('type', 'unknown'), '#B2BEC3') for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=2000, alpha=0.9)
    
    edges = list(G.edges())
    weights = [G[u][v].get('weight', 1.0) for u, v in edges]
    max_weight = max(weights) if weights else 1
    widths = [3 * w / max_weight for w in weights]
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths, alpha=0.6, arrows=True, arrowsize=20)
    
    labels = {n: n.split('.')[-1] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
    
    legend_patches = [mpatches.Patch(color=c, label=n) for n, c in type_colors.items()]
    ax.legend(handles=legend_patches, loc='upper left')
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_edge_importance(
    importance_scores: Dict[str, float],
    top_k: int = 30,
    title: str = "Edge Importance Scores",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """Plot edge importance scores."""
    sorted_edges = sorted(importance_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    names, scores = zip(*sorted_edges) if sorted_edges else ([], [])
    
    fig, ax = plt.subplots(figsize=figsize)
    colors = ['#FF6B6B' if s > 0 else '#4ECDC4' for s in scores]
    ax.barh(range(len(names)), scores, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_training_curves(
    metrics: Dict[str, List[float]],
    title: str = "Training Progress",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Plot training curves."""
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]
    
    colors = plt.cm.Set2(np.linspace(0, 1, n))
    for ax, (name, values), color in zip(axes, metrics.items(), colors):
        ax.plot(values, color=color, linewidth=2, label=name)
        ax.set_ylabel(name)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Step')
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def create_interactive_circuit(
    circuit: Dict[str, Any],
    title: str = "Interactive Circuit",
    save_path: Optional[str] = None,
) -> Any:
    """Create interactive Plotly visualization."""
    if not HAS_PLOTLY or not HAS_NETWORKX:
        raise ImportError("plotly and networkx required")
    
    G = nx.DiGraph()
    for node in circuit.get('nodes', []):
        G.add_node(node['id'], **node)
    for edge in circuit.get('edges', []):
        G.add_edge(edge['source'], edge['target'], weight=edge.get('weight', 1.0))
    
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'),
                            hoverinfo='none', mode='lines')
    
    type_colors = {
        'attention': '#FF6B6B', 'mlp': '#4ECDC4', 'message_passing': '#45B7D1',
        'node_update': '#96CEB4', 'edge_update': '#FFEAA7', 'input': '#DFE6E9',
        'output': '#6C5CE7', 'unknown': '#B2BEC3',
    }
    
    node_x, node_y, node_text, node_colors = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_type = G.nodes[node].get('type', 'unknown')
        node_text.append(f"{node}<br>Type: {node_type}")
        node_colors.append(type_colors.get(node_type, '#B2BEC3'))
    
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', hoverinfo='text',
        text=[n.split('.')[-1] for n in G.nodes()], textposition='top center',
        hovertext=node_text, marker=dict(size=20, color=node_colors, line=dict(width=2, color='white'))
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(title=title, showlegend=False, hovermode='closest',
                                     xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                     yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    if save_path:
        fig.write_html(save_path)
    return fig


def save_circuit_to_json(circuit: Dict[str, Any], path: Union[str, Path]):
    """Save circuit to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, (torch.Tensor, np.ndarray)):
            return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(path, 'w') as f:
        json.dump(convert(circuit), f, indent=2)


def load_circuit_from_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load circuit from JSON."""
    with open(path, 'r') as f:
        return json.load(f)
