"""Tests for the Processor module (MPNN, Transformer, and subcomponents)."""

import pytest
import torch

from models.processor import (
    MultiHeadAttention,
    MessagePassingLayer,
    Processor,
    TransformerProcessor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hidden_dim():
    return 32


@pytest.fixture
def num_nodes():
    return 6


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def num_heads():
    return 4


@pytest.fixture
def graph_inputs(batch_size, num_nodes, hidden_dim):
    """Node features, edge features, and adjacency matrix."""
    node_features = torch.randn(batch_size, num_nodes, hidden_dim)
    edge_features = torch.randn(batch_size, num_nodes, num_nodes, hidden_dim)
    adjacency = (torch.rand(batch_size, num_nodes, num_nodes) > 0.5).float()
    return node_features, edge_features, adjacency


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------

class TestMultiHeadAttention:
    def test_output_shape(self, batch_size, num_nodes, hidden_dim, num_heads):
        mha = MultiHeadAttention(hidden_dim, num_heads)
        q = k = v = torch.randn(batch_size, num_nodes, hidden_dim)
        output, attn_weights = mha(q, k, v)
        assert output.shape == (batch_size, num_nodes, hidden_dim)
        assert attn_weights.shape == (batch_size, num_heads, num_nodes, num_nodes)

    def test_attention_weights_sum_to_one(self, batch_size, num_nodes, hidden_dim, num_heads):
        mha = MultiHeadAttention(hidden_dim, num_heads, use_gating=False)
        mha.eval()
        q = k = v = torch.randn(batch_size, num_nodes, hidden_dim)
        _, attn_weights = mha(q, k, v)
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_with_mask(self, batch_size, num_nodes, hidden_dim, num_heads):
        mha = MultiHeadAttention(hidden_dim, num_heads)
        mha.eval()
        q = k = v = torch.randn(batch_size, num_nodes, hidden_dim)
        mask = torch.ones(batch_size, num_nodes, num_nodes, dtype=torch.bool)
        mask[:, :, -1] = False  # mask out last node as target
        output, attn_weights = mha(q, k, v, mask=mask)
        # Masked positions should have ~0 attention weight
        assert (attn_weights[:, :, :, -1].abs() < 1e-5).all()

    def test_with_edge_features(self, batch_size, num_nodes, hidden_dim, num_heads):
        mha = MultiHeadAttention(hidden_dim, num_heads)
        q = k = v = torch.randn(batch_size, num_nodes, hidden_dim)
        edge_features = torch.randn(batch_size, num_nodes, num_nodes, hidden_dim)
        output, _ = mha(q, k, v, edge_features=edge_features)
        assert output.shape == (batch_size, num_nodes, hidden_dim)

    def test_no_gating(self, hidden_dim, num_heads):
        mha = MultiHeadAttention(hidden_dim, num_heads, use_gating=False)
        assert not hasattr(mha, 'gate') or not isinstance(mha.gate, torch.nn.Linear)
        # Should still produce valid output
        q = k = v = torch.randn(1, 4, hidden_dim)
        output, _ = mha(q, k, v)
        assert output.shape == (1, 4, hidden_dim)

    def test_hidden_dim_not_divisible_by_heads(self):
        with pytest.raises(AssertionError, match="divisible"):
            MultiHeadAttention(hidden_dim=33, num_heads=4)

    def test_gradients_flow(self, hidden_dim, num_heads):
        mha = MultiHeadAttention(hidden_dim, num_heads)
        q = k = v = torch.randn(1, 4, hidden_dim, requires_grad=True)
        output, _ = mha(q, k, v)
        output.sum().backward()
        assert q.grad is not None


# ---------------------------------------------------------------------------
# MessagePassingLayer
# ---------------------------------------------------------------------------

class TestMessagePassingLayer:
    def test_output_shapes(self, graph_inputs, hidden_dim, num_heads, batch_size, num_nodes):
        layer = MessagePassingLayer(hidden_dim, num_heads)
        node_f, edge_f, adj = graph_inputs
        out_nodes, out_edges, acts = layer(node_f, edge_f, adj)
        assert out_nodes.shape == (batch_size, num_nodes, hidden_dim)
        assert out_edges.shape == (batch_size, num_nodes, num_nodes, hidden_dim)

    def test_activations_dict_keys(self, graph_inputs, hidden_dim, num_heads):
        layer = MessagePassingLayer(hidden_dim, num_heads)
        _, _, acts = layer(*graph_inputs)
        expected_keys = {
            'attention_weights', 'attention_output',
            'edge_mlp_input', 'edge_mlp_output',
            'message_weights', 'aggregated_messages',
            'node_mlp_input', 'node_mlp_output',
        }
        assert set(acts.keys()) == expected_keys

    def test_no_layer_norm(self, graph_inputs, hidden_dim, num_heads):
        layer = MessagePassingLayer(hidden_dim, num_heads, layer_norm=False)
        out_nodes, out_edges, _ = layer(*graph_inputs)
        assert out_nodes.shape == graph_inputs[0].shape
        assert out_edges.shape == graph_inputs[1].shape

    def test_disconnected_graph_no_nan(self, batch_size, num_nodes, hidden_dim, num_heads):
        """A graph with no edges should not produce NaN outputs."""
        layer = MessagePassingLayer(hidden_dim, num_heads)
        layer.eval()
        node_f = torch.randn(batch_size, num_nodes, hidden_dim)
        edge_f = torch.randn(batch_size, num_nodes, num_nodes, hidden_dim)
        adj = torch.zeros(batch_size, num_nodes, num_nodes)  # no edges
        out_nodes, out_edges, _ = layer(node_f, edge_f, adj)
        assert not torch.isnan(out_nodes).any(), "NaN in node outputs for disconnected graph"
        assert not torch.isnan(out_edges).any(), "NaN in edge outputs for disconnected graph"

    def test_fully_connected_graph(self, batch_size, num_nodes, hidden_dim, num_heads):
        layer = MessagePassingLayer(hidden_dim, num_heads)
        node_f = torch.randn(batch_size, num_nodes, hidden_dim)
        edge_f = torch.randn(batch_size, num_nodes, num_nodes, hidden_dim)
        adj = torch.ones(batch_size, num_nodes, num_nodes)
        out_nodes, out_edges, _ = layer(node_f, edge_f, adj)
        assert not torch.isnan(out_nodes).any()

    def test_with_node_mask(self, graph_inputs, hidden_dim, num_heads, batch_size, num_nodes):
        layer = MessagePassingLayer(hidden_dim, num_heads)
        node_f, edge_f, adj = graph_inputs
        mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)
        mask[:, -1] = False  # mask out last node
        out_nodes, out_edges, _ = layer(node_f, edge_f, adj, mask=mask)
        assert out_nodes.shape == node_f.shape

    def test_gradients_flow(self, graph_inputs, hidden_dim, num_heads):
        layer = MessagePassingLayer(hidden_dim, num_heads)
        node_f, edge_f, adj = graph_inputs
        node_f = node_f.requires_grad_(True)
        out_nodes, _, _ = layer(node_f, edge_f, adj)
        out_nodes.sum().backward()
        assert node_f.grad is not None


# ---------------------------------------------------------------------------
# Processor (stacked MPNN)
# ---------------------------------------------------------------------------

class TestProcessor:
    def test_output_shapes(self, graph_inputs, hidden_dim, num_heads, batch_size, num_nodes):
        proc = Processor(hidden_dim, num_layers=2, num_heads=num_heads)
        node_f, edge_f, adj = graph_inputs
        out_nodes, out_edges, all_acts = proc(node_f, edge_f, adj, num_steps=1)
        assert out_nodes.shape == (batch_size, num_nodes, hidden_dim)
        assert out_edges.shape == (batch_size, num_nodes, num_nodes, hidden_dim)

    def test_multiple_steps(self, graph_inputs, hidden_dim, num_heads):
        proc = Processor(hidden_dim, num_layers=2, num_heads=num_heads)
        node_f, edge_f, adj = graph_inputs
        out1, _, _ = proc(node_f, edge_f, adj, num_steps=1)
        out2, _, _ = proc(node_f, edge_f, adj, num_steps=3)
        # Different number of steps should (generally) produce different outputs
        assert out1.shape == out2.shape

    def test_return_all_activations(self, graph_inputs, hidden_dim, num_heads):
        num_layers = 2
        num_steps = 3
        proc = Processor(hidden_dim, num_layers=num_layers, num_heads=num_heads)
        node_f, edge_f, adj = graph_inputs
        _, _, all_acts = proc(node_f, edge_f, adj, num_steps=num_steps, return_all_activations=True)
        assert len(all_acts['node_features']) == num_steps
        assert len(all_acts['edge_features']) == num_steps
        assert len(all_acts['layer_activations']) == num_layers
        for layer_acts in all_acts['layer_activations']:
            assert len(layer_acts) == num_steps

    def test_no_activations_by_default(self, graph_inputs, hidden_dim, num_heads):
        proc = Processor(hidden_dim, num_layers=1, num_heads=num_heads)
        node_f, edge_f, adj = graph_inputs
        _, _, all_acts = proc(node_f, edge_f, adj, num_steps=2, return_all_activations=False)
        assert len(all_acts['node_features']) == 0
        assert len(all_acts['edge_features']) == 0

    def test_layer_gating_exists(self, hidden_dim, num_heads):
        num_layers = 3
        proc = Processor(hidden_dim, num_layers=num_layers, num_heads=num_heads)
        assert len(proc.layer_gates) == num_layers
        for gate in proc.layer_gates:
            assert gate.shape == (1,)

    def test_single_layer(self, graph_inputs, hidden_dim, num_heads):
        proc = Processor(hidden_dim, num_layers=1, num_heads=num_heads)
        out_nodes, out_edges, _ = proc(*graph_inputs, num_steps=1)
        assert out_nodes.shape == graph_inputs[0].shape

    def test_with_mask(self, graph_inputs, hidden_dim, num_heads, batch_size, num_nodes):
        proc = Processor(hidden_dim, num_layers=1, num_heads=num_heads)
        node_f, edge_f, adj = graph_inputs
        mask = torch.ones(batch_size, num_nodes, dtype=torch.bool)
        out_nodes, _, _ = proc(node_f, edge_f, adj, num_steps=1, mask=mask)
        assert out_nodes.shape == node_f.shape


# ---------------------------------------------------------------------------
# TransformerProcessor
# ---------------------------------------------------------------------------

class TestTransformerProcessor:
    def test_output_shapes(self, graph_inputs, hidden_dim, num_heads, batch_size, num_nodes):
        proc = TransformerProcessor(hidden_dim, num_layers=2, num_heads=num_heads)
        node_f, edge_f, adj = graph_inputs
        out_nodes, out_edges, all_acts = proc(node_f, edge_f, adj, num_steps=1)
        assert out_nodes.shape == (batch_size, num_nodes, hidden_dim)
        assert out_edges.shape == (batch_size, num_nodes, num_nodes, hidden_dim)

    def test_multiple_steps(self, graph_inputs, hidden_dim, num_heads):
        proc = TransformerProcessor(hidden_dim, num_layers=1, num_heads=num_heads)
        out1, _, _ = proc(*graph_inputs, num_steps=1)
        out2, _, _ = proc(*graph_inputs, num_steps=2)
        assert out1.shape == out2.shape

    def test_return_all_activations(self, graph_inputs, hidden_dim, num_heads):
        num_steps = 3
        proc = TransformerProcessor(hidden_dim, num_layers=1, num_heads=num_heads)
        _, _, all_acts = proc(*graph_inputs, num_steps=num_steps, return_all_activations=True)
        assert len(all_acts['node_features']) == num_steps
        assert len(all_acts['edge_features']) == num_steps

    def test_no_activations_by_default(self, graph_inputs, hidden_dim, num_heads):
        proc = TransformerProcessor(hidden_dim, num_layers=1, num_heads=num_heads)
        _, _, all_acts = proc(*graph_inputs, num_steps=2, return_all_activations=False)
        assert len(all_acts['node_features']) == 0

    def test_custom_ff_dim(self, hidden_dim, num_heads):
        proc = TransformerProcessor(hidden_dim, num_layers=1, num_heads=num_heads, ff_dim=64)
        assert proc is not None

    def test_default_ff_dim(self, hidden_dim, num_heads):
        proc = TransformerProcessor(hidden_dim, num_layers=1, num_heads=num_heads)
        # Default ff_dim = hidden_dim * 4
        # Verify by checking the transformer layer
        layer = proc.transformer.layers[0]
        assert layer.linear1.out_features == hidden_dim * 4

    def test_edge_update_respects_adjacency(self, batch_size, num_nodes, hidden_dim, num_heads):
        """Edge updates should only apply where adjacency is 1."""
        proc = TransformerProcessor(hidden_dim, num_layers=1, num_heads=num_heads)
        proc.eval()
        node_f = torch.randn(batch_size, num_nodes, hidden_dim)
        edge_f = torch.zeros(batch_size, num_nodes, num_nodes, hidden_dim)
        adj = torch.zeros(batch_size, num_nodes, num_nodes)
        adj[:, 0, 1] = 1.0  # only one edge
        _, out_edges, _ = proc(node_f, edge_f, adj, num_steps=1)
        # Non-adjacent edges should remain zero
        assert (out_edges[:, 1, 0, :] == 0).all()
        assert (out_edges[:, 2, 3, :] == 0).all()
