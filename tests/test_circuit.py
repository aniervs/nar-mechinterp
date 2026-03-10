"""Tests for Circuit data structures and analysis utilities."""

import pytest
import json
import tempfile
from pathlib import Path

from interp.circuit import (
    Circuit,
    CircuitNode,
    CircuitEdge,
    compare_circuits,
    merge_circuits,
    find_shared_subcircuit,
    create_full_circuit_from_model,
)


# --- CircuitNode ---

class TestCircuitNode:
    def test_basic_creation(self):
        node = CircuitNode(id="n1", type="attention", layer=2)
        assert node.id == "n1"
        assert node.type == "attention"
        assert node.layer == 2
        assert node.metadata == {}

    def test_to_dict(self):
        node = CircuitNode(id="n1", type="mlp", layer=1, metadata={"foo": "bar"})
        d = node.to_dict()
        assert d == {"id": "n1", "type": "mlp", "layer": 1, "metadata": {"foo": "bar"}}


# --- CircuitEdge ---

class TestCircuitEdge:
    def test_basic_creation(self):
        edge = CircuitEdge(id="e1", source="a", target="b")
        assert edge.weight == 1.0
        assert edge.active is True

    def test_to_dict(self):
        edge = CircuitEdge(id="e1", source="a", target="b", weight=0.5, active=False)
        d = edge.to_dict()
        assert d["weight"] == 0.5
        assert d["active"] is False


# --- Circuit construction ---

class TestCircuitConstruction:
    def test_empty_circuit(self):
        c = Circuit()
        assert c.num_nodes == 0
        assert c.num_edges == 0
        assert c.name == ""

    def test_circuit_with_name(self):
        c = Circuit(name="my_circuit")
        assert c.name == "my_circuit"
        assert c.num_nodes == 0

    def test_circuit_from_lists(self):
        nodes = [CircuitNode(id="a", type="input"), CircuitNode(id="b", type="output")]
        edges = [CircuitEdge(id="e1", source="a", target="b")]
        c = Circuit(nodes=nodes, edges=edges)
        assert c.num_nodes == 2
        assert c.num_edges == 1


# --- Circuit mutation ---

class TestCircuitMutation:
    def test_add_node(self):
        c = Circuit(name="test")
        node = c.add_node("n1", "attention", layer=1, component="attn")
        assert isinstance(node, CircuitNode)
        assert c.num_nodes == 1
        assert c.get_node("n1") is node
        assert node.metadata["component"] == "attn"

    def test_add_edge(self):
        c = Circuit()
        c.add_node("a", "input")
        c.add_node("b", "output")
        edge = c.add_edge("a", "b", weight=0.7)
        assert isinstance(edge, CircuitEdge)
        assert c.num_edges == 1
        assert edge.id == "a->b"
        assert edge.weight == 0.7

    def test_add_edge_with_type(self):
        c = Circuit()
        c.add_node("a", "input")
        c.add_node("b", "output")
        edge = c.add_edge("a", "b", edge_type="residual")
        assert edge.metadata["edge_type"] == "residual"

    def test_remove_edge(self):
        c = Circuit()
        c.add_node("a", "input")
        c.add_node("b", "output")
        c.add_edge("a", "b")
        assert c.num_edges == 1
        c.remove_edge("a", "b")
        assert c.num_edges == 0

    def test_remove_edge_nonexistent(self):
        c = Circuit()
        c.add_node("a", "input")
        c.remove_edge("a", "b")  # Should not raise
        assert c.num_edges == 0

    def test_remove_node(self):
        c = Circuit()
        c.add_node("a", "input")
        c.add_node("b", "mlp", layer=1)
        c.add_node("c", "output", layer=2)
        c.add_edge("a", "b")
        c.add_edge("b", "c")
        c.add_edge("a", "c")

        c.remove_node("b")
        assert c.num_nodes == 2
        # Edges involving b should be removed
        assert c.num_edges == 1
        assert c.edges[0].source == "a" and c.edges[0].target == "c"

    def test_remove_node_nonexistent(self):
        c = Circuit()
        c.add_node("a", "input")
        c.remove_node("z")  # Should not raise
        assert c.num_nodes == 1


# --- Circuit properties ---

class TestCircuitProperties:
    @pytest.fixture
    def linear_circuit(self):
        c = Circuit(name="linear")
        c.add_node("input", "input", layer=0)
        c.add_node("h1", "mlp", layer=1)
        c.add_node("h2", "attention", layer=2)
        c.add_node("output", "output", layer=3)
        c.add_edge("input", "h1")
        c.add_edge("h1", "h2")
        c.add_edge("h2", "output")
        return c

    def test_num_layers(self, linear_circuit):
        assert linear_circuit.num_layers == 4  # layers 0,1,2,3

    def test_get_nodes_by_type(self, linear_circuit):
        mlps = linear_circuit.get_nodes_by_type("mlp")
        assert len(mlps) == 1
        assert mlps[0].id == "h1"

    def test_get_nodes_at_layer(self, linear_circuit):
        layer1 = linear_circuit.get_nodes_at_layer(1)
        assert len(layer1) == 1
        assert layer1[0].id == "h1"

    def test_get_edges_from(self, linear_circuit):
        edges = linear_circuit.get_edges_from("h1")
        assert len(edges) == 1
        assert edges[0].target == "h2"

    def test_get_edges_to(self, linear_circuit):
        edges = linear_circuit.get_edges_to("h2")
        assert len(edges) == 1
        assert edges[0].source == "h1"

    def test_is_connected(self, linear_circuit):
        assert linear_circuit.is_connected()

    def test_disconnected_circuit(self):
        c = Circuit()
        c.add_node("a", "input")
        c.add_node("b", "output")
        # No edges — disconnected
        assert not c.is_connected()

    def test_get_paths(self, linear_circuit):
        paths = linear_circuit.get_paths("input", "output")
        assert len(paths) == 1
        assert paths[0] == ["input", "h1", "h2", "output"]

    def test_compute_statistics(self, linear_circuit):
        stats = linear_circuit.compute_statistics()
        assert stats["num_nodes"] == 4
        assert stats["num_edges"] == 3
        assert stats["is_connected"] is True


# --- Serialization ---

class TestCircuitSerialization:
    @pytest.fixture
    def sample_circuit(self):
        c = Circuit(name="ser_test")
        c.add_node("a", "input", layer=0)
        c.add_node("b", "mlp", layer=1)
        c.add_edge("a", "b", weight=0.9)
        return c

    def test_to_dict(self, sample_circuit):
        d = sample_circuit.to_dict()
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1
        assert "statistics" in d

    def test_from_dict_roundtrip(self, sample_circuit):
        d = sample_circuit.to_dict()
        c2 = Circuit.from_dict(d)
        assert c2.num_nodes == sample_circuit.num_nodes
        assert c2.num_edges == sample_circuit.num_edges

    def test_save_and_load(self, sample_circuit):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "circuit.json"
            sample_circuit.save(str(path))
            assert path.exists()

            loaded = Circuit.load(str(path))
            assert loaded.num_nodes == 2
            assert loaded.num_edges == 1


# --- Circuit comparison utilities ---

class TestCircuitComparison:
    @pytest.fixture
    def circuit_pair(self):
        c1 = Circuit()
        c1.add_node("a", "input")
        c1.add_node("b", "mlp")
        c1.add_node("c", "output")
        c1.add_edge("a", "b")
        c1.add_edge("b", "c")

        c2 = Circuit()
        c2.add_node("a", "input")
        c2.add_node("b", "mlp")
        c2.add_node("d", "attention")
        c2.add_edge("a", "b")
        c2.add_edge("b", "d")
        return c1, c2

    def test_compare_identical(self):
        c = Circuit()
        c.add_node("a", "input")
        c.add_node("b", "output")
        c.add_edge("a", "b")
        result = compare_circuits(c, c)
        assert result["overall_similarity"] == 1.0

    def test_compare_different(self, circuit_pair):
        c1, c2 = circuit_pair
        result = compare_circuits(c1, c2)
        assert 0.0 < result["overall_similarity"] < 1.0
        assert "node_comparison" in result
        assert "edge_comparison" in result

    def test_merge_circuits(self, circuit_pair):
        c1, c2 = circuit_pair
        merged = merge_circuits([c1, c2])
        # Union: a, b, c, d
        assert merged.num_nodes == 4

    def test_find_shared_subcircuit(self, circuit_pair):
        c1, c2 = circuit_pair
        shared = find_shared_subcircuit([c1, c2])
        shared_ids = {n.id for n in shared.nodes}
        assert "a" in shared_ids
        assert "b" in shared_ids
        assert "c" not in shared_ids
        assert "d" not in shared_ids

    def test_find_shared_empty(self):
        shared = find_shared_subcircuit([])
        assert shared.num_nodes == 0


# --- create_full_circuit_from_model ---

class TestCreateFullCircuitFromModel:
    def test_with_simple_model(self):
        import torch.nn as nn
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        circuit = create_full_circuit_from_model(model, name="simple")
        assert circuit.name == "simple"
        # Should have input, output, plus leaf modules
        assert circuit.num_nodes >= 3
        assert circuit.num_edges > 0
