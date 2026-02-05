"""
Message Passing Neural Network Processor for Neural Algorithmic Reasoning.

This implements the core processor architecture that learns to execute algorithms
step by step, following the encode-process-decode paradigm from CLRS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from einops import rearrange, repeat
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional gating."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_gating: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_gating = use_gating
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # QKV projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Optional gating mechanism
        if use_gating:
            self.gate = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        edge_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query: (batch, seq_len, hidden_dim)
            key: (batch, seq_len, hidden_dim)
            value: (batch, seq_len, hidden_dim)
            mask: Optional attention mask
            edge_features: Optional edge features to bias attention
            
        Returns:
            output: (batch, seq_len, hidden_dim)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = query.shape
        
        # Project to Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Compute attention scores
        attn_scores = torch.einsum('bhnd,bhmd->bhnm', q, k) / self.scale
        
        # Add edge features as attention bias if provided
        if edge_features is not None:
            # edge_features: (batch, seq_len, seq_len, hidden_dim)
            edge_bias = rearrange(edge_features, 'b n m d -> b 1 n m d').mean(dim=-1)
            attn_scores = attn_scores + edge_bias
        
        # Apply mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        
        # Softmax with safe handling of -inf (replace with large negative number to prevent NaN)
        attn_scores = attn_scores.masked_fill(torch.isinf(attn_scores) & (attn_scores < 0), -1e9)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.einsum('bhnm,bhmd->bhnd', attn_weights, v)
        output = rearrange(output, 'b h n d -> b n (h d)')
        
        # Output projection
        output = self.out_proj(output)
        
        # Apply gating
        if self.use_gating:
            gate = torch.sigmoid(self.gate(query))
            output = gate * output
        
        return output, attn_weights


class MessagePassingLayer(nn.Module):
    """
    Single message passing layer with node and edge updates.
    
    This is the core computational unit that mimics algorithm execution steps.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_gating: bool = True,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Node-level attention (self-attention over nodes)
        self.node_attention = MultiHeadAttention(
            hidden_dim, num_heads, dropout, use_gating
        )
        
        # Edge update MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # Aggregation attention (for message aggregation)
        self.msg_attention = nn.Linear(hidden_dim, 1)
        
        # Layer normalization
        self.use_layer_norm = layer_norm
        if layer_norm:
            self.node_ln1 = nn.LayerNorm(hidden_dim)
            self.node_ln2 = nn.LayerNorm(hidden_dim)
            self.edge_ln = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        adjacency: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of message passing layer.
        
        Args:
            node_features: (batch, num_nodes, hidden_dim)
            edge_features: (batch, num_nodes, num_nodes, hidden_dim)
            adjacency: (batch, num_nodes, num_nodes) binary adjacency matrix
            mask: Optional node mask
            
        Returns:
            updated_nodes: (batch, num_nodes, hidden_dim)
            updated_edges: (batch, num_nodes, num_nodes, hidden_dim)
            activations: Dict of intermediate activations for interpretability
        """
        batch_size, num_nodes, _ = node_features.shape
        activations = {}
        
        # Create attention mask from adjacency
        attn_mask = adjacency.bool() if mask is None else adjacency.bool() & mask.unsqueeze(-1)
        
        # Fix: Add self-loops for disconnected nodes to prevent NaN from softmax on all-inf rows
        num_connected = attn_mask.sum(dim=-1)
        if (num_connected == 0).any():
            attn_mask = attn_mask | torch.eye(num_nodes, device=attn_mask.device, dtype=torch.bool).unsqueeze(0)
        
        # === Node Self-Attention ===
        residual = node_features
        if self.use_layer_norm:
            node_features = self.node_ln1(node_features)
        
        attn_out, attn_weights = self.node_attention(
            node_features, node_features, node_features,
            mask=attn_mask, edge_features=edge_features
        )
        
        activations['attention_weights'] = attn_weights
        activations['attention_output'] = attn_out
        
        node_features = residual + self.dropout(attn_out)
        
        # === Edge Update ===
        # Concatenate source and target node features with current edge features
        src_features = node_features.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        tgt_features = node_features.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        
        edge_input = torch.cat([src_features, tgt_features, edge_features], dim=-1)
        edge_update = self.edge_mlp(edge_input)
        activations['edge_mlp_input'] = edge_input
        activations['edge_mlp_output'] = edge_update
        
        if self.use_layer_norm:
            edge_update = self.edge_ln(edge_update)
        
        # Mask edges by adjacency
        edge_features = edge_features + edge_update * adjacency.unsqueeze(-1)
        
        # === Message Aggregation ===
        # Compute messages from neighbors
        messages = edge_features * adjacency.unsqueeze(-1)  # (batch, num_nodes, num_nodes, hidden_dim)
        
        # Attention-weighted aggregation
        msg_weights = self.msg_attention(messages).squeeze(-1)  # (batch, num_nodes, num_nodes)
        msg_weights = msg_weights.masked_fill(~adjacency.bool(), float('-inf'))
        
        # Fix: Handle positions with no neighbors by using uniform distribution
        num_inf_msg = (msg_weights == float('-inf')).sum(dim=-1)
        if (num_inf_msg == msg_weights.shape[-1]).any():
            no_neighbors = (num_inf_msg == msg_weights.shape[-1])
            msg_weights[no_neighbors] = 0  # Becomes uniform after softmax
        
        # Safe softmax: replace -inf with large negative number
        msg_weights = msg_weights.masked_fill(torch.isinf(msg_weights) & (msg_weights < 0), -1e9)
        msg_weights = F.softmax(msg_weights, dim=-1)
        
        msg_weights = msg_weights.masked_fill(~adjacency.bool(), 0)
        activations['message_weights'] = msg_weights
        
        aggregated = torch.einsum('bnm,bnmd->bnd', msg_weights, messages)
        activations['aggregated_messages'] = aggregated
        
        # === Node Update ===
        residual = node_features
        if self.use_layer_norm:
            node_features = self.node_ln2(node_features)
        
        node_input = torch.cat([node_features, aggregated], dim=-1)
        node_update = self.node_mlp(node_input)
        activations['node_mlp_input'] = node_input
        activations['node_mlp_output'] = node_update
        
        node_features = residual + self.dropout(node_update)
        
        return node_features, edge_features, activations


class Processor(nn.Module):
    """
    Full processor network with multiple message passing layers.
    
    This implements the iterative processing component that learns
    to execute algorithms over multiple steps.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_gating: bool = True,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Stack of message passing layers
        self.layers = nn.ModuleList([
            MessagePassingLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_gating=use_gating,
                layer_norm=layer_norm,
            )
            for _ in range(num_layers)
        ])
        
        # Layer-wise gating (allows skipping layers)
        self.layer_gates = nn.ParameterList([
            nn.Parameter(torch.ones(1))
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        adjacency: torch.Tensor,
        num_steps: int = 1,
        mask: Optional[torch.Tensor] = None,
        return_all_activations: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, List]]:
        """
        Process inputs through multiple message passing steps.
        
        Args:
            node_features: Initial node representations
            edge_features: Initial edge representations
            adjacency: Graph structure
            num_steps: Number of processing steps (for iterative algorithms)
            mask: Optional node mask
            return_all_activations: Whether to store activations from all steps
            
        Returns:
            final_nodes: Final node representations
            final_edges: Final edge representations
            all_activations: Dict of activation lists per layer/component
        """
        all_activations = {
            'node_features': [],
            'edge_features': [],
            'layer_activations': [[] for _ in range(self.num_layers)],
        }
        
        current_nodes = node_features
        current_edges = edge_features
        
        # Multiple processing steps (unrolling algorithm execution)
        for step in range(num_steps):
            step_nodes = current_nodes
            step_edges = current_edges
            
            # Process through all layers
            for layer_idx, layer in enumerate(self.layers):
                step_nodes, step_edges, layer_acts = layer(
                    step_nodes, step_edges, adjacency, mask
                )
                
                # Apply layer gating
                gate = torch.sigmoid(self.layer_gates[layer_idx])
                step_nodes = gate * step_nodes + (1 - gate) * current_nodes
                
                if return_all_activations:
                    all_activations['layer_activations'][layer_idx].append({
                        'step': step,
                        **{k: v.detach() for k, v in layer_acts.items()}
                    })
            
            current_nodes = step_nodes
            current_edges = step_edges
            
            if return_all_activations:
                all_activations['node_features'].append(current_nodes.detach())
                all_activations['edge_features'].append(current_edges.detach())
        
        return current_nodes, current_edges, all_activations


class TransformerProcessor(nn.Module):
    """
    Alternative processor using standard Transformer architecture.
    
    Useful for comparison and ablation studies.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        ff_dim: int = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        ff_dim = ff_dim or hidden_dim * 4
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Edge processing (separate from node transformer)
        self.edge_processor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        adjacency: torch.Tensor,
        num_steps: int = 1,
        mask: Optional[torch.Tensor] = None,
        return_all_activations: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Process using Transformer."""
        batch_size, num_nodes, _ = node_features.shape
        all_activations = {'node_features': [], 'edge_features': []}
        
        current_nodes = node_features
        current_edges = edge_features
        
        for step in range(num_steps):
            # Process nodes with transformer
            current_nodes = self.transformer(current_nodes)
            
            # Update edges based on updated node features
            src = current_nodes.unsqueeze(2).expand(-1, -1, num_nodes, -1)
            tgt = current_nodes.unsqueeze(1).expand(-1, num_nodes, -1, -1)
            edge_input = torch.cat([src, tgt], dim=-1)
            edge_update = self.edge_processor(edge_input)
            current_edges = current_edges + edge_update * adjacency.unsqueeze(-1)
            
            if return_all_activations:
                all_activations['node_features'].append(current_nodes.detach())
                all_activations['edge_features'].append(current_edges.detach())
        
        return current_nodes, current_edges, all_activations
