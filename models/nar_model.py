"""
Neural Algorithmic Reasoning Model

Implements the encode-process-decode architecture for learning algorithms
from the CLRS-30 benchmark.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .processor import Processor, TransformerProcessor


@dataclass
class NAROutput:
    """Output from NAR model."""
    # Predicted outputs
    predictions: Dict[str, torch.Tensor]
    # Predicted hints at each step
    hint_predictions: Optional[Dict[str, List[torch.Tensor]]]
    # Loss components
    output_loss: torch.Tensor
    hint_loss: Optional[torch.Tensor]
    total_loss: torch.Tensor
    # Activations for interpretability
    activations: Dict[str, Any]


class Encoder(nn.Module):
    """
    Encoder network that embeds algorithm inputs.
    
    Handles different input types:
    - Scalars, node features, edge features
    - Pointers (categorical over nodes)
    - Graphs (adjacency matrices)
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        input_specs: Optional[Dict] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Default embeddings for common input types
        # Scalar embedding
        self.scalar_encoder = nn.Linear(1, hidden_dim)
        
        # Node feature embedding
        self.node_encoder = nn.Linear(8, hidden_dim)  # Assume 8-dim node features
        
        # Edge weight embedding
        self.edge_encoder = nn.Linear(1, hidden_dim)
        
        # Pointer embedding (learned embedding for pointer positions)
        self.pointer_encoder = nn.Embedding(256, hidden_dim)  # Max 256 nodes
        
        # Position embedding
        self.position_encoder = nn.Embedding(256, hidden_dim)
        
        # Projection to combine multiple input encodings
        self.combine_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        num_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode algorithm inputs.
        
        Args:
            inputs: Dict of input tensors
            num_nodes: Number of nodes in the graph
            
        Returns:
            node_encodings: (batch, num_nodes, hidden_dim)
            edge_encodings: (batch, num_nodes, num_nodes, hidden_dim)
        """
        batch_size = next(iter(inputs.values())).shape[0]
        device = next(iter(inputs.values())).device
        
        # Initialize encodings
        node_encodings = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=device)
        edge_encodings = torch.zeros(batch_size, num_nodes, num_nodes, self.hidden_dim, device=device)
        
        # Add position encoding
        positions = torch.arange(num_nodes, device=device)
        pos_enc = self.position_encoder(positions)
        node_encodings = node_encodings + pos_enc.unsqueeze(0)
        
        # Encode each input type
        for name, tensor in inputs.items():
            if 'node_features' in name or 'features' in name:
                # Node features: (batch, num_nodes, feat_dim)
                if tensor.dim() == 2:
                    tensor = tensor.unsqueeze(-1)
                if tensor.shape[-1] != 8:
                    # Pad or project to expected dimension
                    tensor = F.pad(tensor, (0, max(0, 8 - tensor.shape[-1])))[:, :, :8]
                enc = self.node_encoder(tensor[:, :num_nodes])
                node_encodings = node_encodings + enc
                
            elif 'adjacency' in name or 'adj' in name:
                # Adjacency matrix: (batch, num_nodes, num_nodes)
                adj = tensor[:, :num_nodes, :num_nodes]
                edge_encodings = edge_encodings + adj.unsqueeze(-1) * self.hidden_dim ** 0.5
                
            elif 'edge_weights' in name or 'weights' in name:
                # Edge weights: (batch, num_nodes, num_nodes)
                weights = tensor[:, :num_nodes, :num_nodes]
                enc = self.edge_encoder(weights.unsqueeze(-1))
                edge_encodings = edge_encodings + enc
                
            elif 'source' in name or 'pointer' in name:
                # Pointer input (one-hot or index)
                if tensor.dim() == 1:
                    # Index form
                    ptr_enc = self.pointer_encoder(tensor.long())
                    # Broadcast to all nodes
                    node_encodings = node_encodings + ptr_enc.unsqueeze(1)
                else:
                    # One-hot form: (batch, num_nodes)
                    ptr_idx = tensor[:, :num_nodes].argmax(dim=1)
                    ptr_enc = self.pointer_encoder(ptr_idx)
                    node_encodings = node_encodings + ptr_enc.unsqueeze(1)
                    # Also mark the source node specifically
                    source_mask = tensor[:, :num_nodes].unsqueeze(-1)
                    node_encodings = node_encodings + source_mask * ptr_enc.unsqueeze(1)
        
        return node_encodings, edge_encodings


class Decoder(nn.Module):
    """
    Decoder network that produces algorithm outputs and hints.
    
    Handles different output types:
    - Node predictions (per-node classification/regression)
    - Edge predictions
    - Pointer predictions (softmax over nodes)
    - Scalar outputs
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        output_specs: Optional[Dict] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Output heads for different types
        # Node classification (e.g., reachability)
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Node regression (e.g., distances)
        self.node_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Pointer prediction (attention over nodes)
        self.pointer_query = nn.Linear(hidden_dim, hidden_dim)
        self.pointer_key = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge prediction
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        output_types: Dict[str, str],
    ) -> Dict[str, torch.Tensor]:
        """
        Decode outputs from processed features.
        
        Args:
            node_features: (batch, num_nodes, hidden_dim)
            edge_features: (batch, num_nodes, num_nodes, hidden_dim)
            output_types: Dict mapping output names to types
            
        Returns:
            predictions: Dict of output predictions
        """
        predictions = {}
        
        for name, out_type in output_types.items():
            if out_type in ['node_mask', 'node_categorical']:
                # Binary/categorical prediction per node
                pred = self.node_classifier(node_features).squeeze(-1)
                predictions[name] = pred
                
            elif out_type == 'node_scalar':
                # Regression per node
                pred = self.node_regressor(node_features).squeeze(-1)
                predictions[name] = pred
                
            elif out_type == 'node_pointer':
                # Pointer prediction: softmax attention over nodes
                query = self.pointer_query(node_features)  # (batch, num_nodes, hidden_dim)
                key = self.pointer_key(node_features)  # (batch, num_nodes, hidden_dim)
                
                # Each node predicts which other node it points to
                attn = torch.einsum('bnh,bmh->bnm', query, key)
                attn = attn / (self.hidden_dim ** 0.5)
                pred = F.softmax(attn, dim=-1)
                predictions[name] = pred
                
            elif out_type in ['edge_mask', 'edge_scalar']:
                # Edge-level prediction
                pred = self.edge_classifier(edge_features).squeeze(-1)
                predictions[name] = pred
                
            elif out_type == 'array_scalar':
                # Array output (treated as node-level)
                pred = self.node_regressor(node_features).squeeze(-1)
                predictions[name] = pred
        
        return predictions


class NARModel(nn.Module):
    """
    Full Neural Algorithmic Reasoning model.
    
    Combines encoder, processor, and decoder with support for
    hint supervision and interpretability hooks.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        processor_type: str = "mpnn",
        use_gating: bool = True,
        layer_norm: bool = True,
        encode_hints: bool = True,
        decode_hints: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.encode_hints = encode_hints
        self.decode_hints = decode_hints
        
        # Encoder
        self.encoder = Encoder(hidden_dim=hidden_dim)
        
        # Processor
        if processor_type == "mpnn":
            self.processor = Processor(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                use_gating=use_gating,
                layer_norm=layer_norm,
            )
        elif processor_type == "transformer":
            self.processor = TransformerProcessor(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown processor type: {processor_type}")
        
        # Decoder
        self.decoder = Decoder(hidden_dim=hidden_dim)
        
        # Hint encoder/decoder (optional)
        if encode_hints:
            self.hint_encoder = nn.ModuleDict({
                'node_mask': nn.Linear(1, hidden_dim),
                'node_scalar': nn.Linear(1, hidden_dim),
                'node_pointer': nn.Linear(256, hidden_dim),  # One-hot encoded
                'node_categorical': nn.Linear(8, hidden_dim),  # Max 8 categories
            })
        
        if decode_hints:
            self.hint_decoder = Decoder(hidden_dim=hidden_dim)
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        hints: Optional[Dict[str, torch.Tensor]] = None,
        outputs: Optional[Dict[str, torch.Tensor]] = None,
        output_types: Dict[str, str] = None,
        hint_types: Dict[str, str] = None,
        num_steps: int = 1,
        return_activations: bool = False,
    ) -> NAROutput:
        """
        Forward pass through the NAR model.
        
        Args:
            inputs: Algorithm inputs
            hints: Ground truth hints (for training)
            outputs: Ground truth outputs (for loss computation)
            output_types: Types of each output
            hint_types: Types of each hint
            num_steps: Number of processing steps
            return_activations: Whether to collect activations
            
        Returns:
            NAROutput with predictions, losses, and activations
        """
        # Determine number of nodes
        for key, val in inputs.items():
            if 'adjacency' in key:
                num_nodes = val.shape[1]
                break
        else:
            num_nodes = next(iter(inputs.values())).shape[1]
        
        batch_size = next(iter(inputs.values())).shape[0]
        device = next(iter(inputs.values())).device
        
        # Default output types if not provided
        if output_types is None:
            output_types = {'reach': 'node_mask', 'predecessor': 'node_pointer'}
        
        if hint_types is None:
            hint_types = {'reach': 'node_mask', 'predecessor': 'node_pointer'}
        
        # Get adjacency for processor
        adjacency = inputs.get('adjacency', torch.ones(batch_size, num_nodes, num_nodes, device=device))
        
        # Encode inputs
        node_encodings, edge_encodings = self.encoder(inputs, num_nodes)
        
        # Process
        processed_nodes, processed_edges, activations = self.processor(
            node_encodings,
            edge_encodings,
            adjacency,
            num_steps=num_steps,
            return_all_activations=return_activations,
        )
        
        # Decode outputs
        predictions = self.decoder(processed_nodes, processed_edges, output_types)
        
        # Decode hints at each step (if enabled)
        hint_predictions = None
        if self.decode_hints and return_activations and 'node_features' in activations:
            hint_predictions = {name: [] for name in hint_types}
            for step_nodes in activations['node_features']:
                step_preds = self.hint_decoder(
                    step_nodes, processed_edges, hint_types
                )
                for name in hint_types:
                    hint_predictions[name].append(step_preds.get(name))
        
        # Compute losses
        output_loss = torch.tensor(0.0, device=device)
        hint_loss = torch.tensor(0.0, device=device)
        
        
        
        if outputs is not None:
            for name, pred in predictions.items():
                if name in outputs:
                    target = outputs[name]
                    out_type = output_types.get(name, 'node_mask')
                    
                    if out_type in ['node_mask', 'edge_mask']:
                        # Binary cross-entropy
                        loss = F.binary_cross_entropy_with_logits(
                            pred[:, :target.shape[-1]],
                            target.float()
                        )                        
                        
                    elif out_type == 'node_pointer':
                        # Cross-entropy over pointer targets
                        if target.dim() == 2:
                            # One-hot or probability target
                            loss = F.cross_entropy(
                                pred.view(-1, pred.shape[-1]),
                                target.argmax(-1).view(-1)
                            )
                        else:
                            loss = F.cross_entropy(
                                pred.view(-1, pred.shape[-1]),
                                target.view(-1).long()
                            )
                    else:
                        # MSE for scalars
                        loss = F.mse_loss(pred[:, :target.shape[-1]], target)
                    
                    output_loss = output_loss + loss
        
        
        if hints is not None and hint_predictions is not None:
            for name, preds in hint_predictions.items():
                if name in hints:
                    target = hints[name]  # (batch, num_steps, ...)
                    hint_type = hint_types.get(name, 'node_mask')
                    
                    for step, pred in enumerate(preds):
                        if step < target.shape[1]:
                            step_target = target[:, step]
                            
                            if hint_type in ['node_mask']:
                                loss = F.binary_cross_entropy_with_logits(
                                    pred[:, :step_target.shape[-1]],
                                    step_target.float()
                                )
                            elif hint_type == 'node_pointer':
                                if step_target.dim() == 2:
                                    loss = F.cross_entropy(
                                        pred.view(-1, pred.shape[-1]),
                                        step_target.argmax(-1).view(-1)
                                    )
                                else:
                                    loss = F.cross_entropy(
                                        pred.view(-1, pred.shape[-1]),
                                        step_target.view(-1).long()
                                    )
                            else:
                                loss = F.mse_loss(pred[:, :step_target.shape[-1]], step_target)
                            
                            hint_loss = hint_loss + loss
        
        total_loss = output_loss + 0.5 * hint_loss
        
        return NAROutput(
            predictions=predictions,
            hint_predictions=hint_predictions,
            output_loss=output_loss,
            hint_loss=hint_loss if hints is not None else None,
            total_loss=total_loss,
            activations=activations if return_activations else {},
        )
    
    def get_named_components(self) -> Dict[str, nn.Module]:
        """
        Get named components for circuit discovery.
        
        Returns dict mapping component names to modules.
        """
        components = {
            'encoder': self.encoder,
            'decoder': self.decoder,
        }
        
        # Add processor layers
        if hasattr(self.processor, 'layers'):
            for i, layer in enumerate(self.processor.layers):
                components[f'processor.layer_{i}'] = layer
                components[f'processor.layer_{i}.attention'] = layer.node_attention
                components[f'processor.layer_{i}.edge_mlp'] = layer.edge_mlp
                components[f'processor.layer_{i}.node_mlp'] = layer.node_mlp
        
        return components
