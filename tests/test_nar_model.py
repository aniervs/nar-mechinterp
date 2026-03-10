"""Tests for the NAR model (Encoder-Processor-Decoder)."""

import pytest
import warnings
import torch
import torch.nn as nn

from models.nar_model import NARModel, NAROutput, Encoder, Decoder


class TestEncoder:
    def test_forward_shape(self):
        encoder = Encoder(hidden_dim=64)
        inputs = {
            "adjacency": torch.randn(2, 8, 8),
            "node_features": torch.randn(2, 8, 8),
        }
        node_enc, edge_enc = encoder(inputs, num_nodes=8)
        assert node_enc.shape == (2, 8, 64)
        assert edge_enc.shape == (2, 8, 8, 64)


class TestNARModel:
    @pytest.fixture
    def model(self):
        return NARModel(hidden_dim=64, num_layers=2, num_heads=4, processor_type="mpnn")

    @pytest.fixture
    def sample_inputs(self):
        return {
            "adjacency": torch.randint(0, 2, (2, 8, 8)).float(),
            "node_features": torch.randn(2, 8, 8),
        }

    def test_creation(self, model):
        assert isinstance(model, nn.Module)
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

    def test_forward_with_output_types(self, model, sample_inputs):
        output = model(
            inputs=sample_inputs,
            output_types={"reach": "node_mask"},
            num_steps=2,
        )
        assert isinstance(output, NAROutput)
        assert "reach" in output.predictions
        assert output.total_loss is not None

    def test_forward_without_output_types_warns(self, model, sample_inputs):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = model(inputs=sample_inputs, num_steps=1)
            assert any("output_types not provided" in str(warning.message) for warning in w)
        assert output.predictions == {}

    def test_forward_with_outputs_computes_loss(self, model, sample_inputs):
        outputs = {"reach": torch.randint(0, 2, (2, 8)).float()}
        output = model(
            inputs=sample_inputs,
            outputs=outputs,
            output_types={"reach": "node_mask"},
            num_steps=2,
        )
        # With ground truth, loss should be > 0 (in general)
        assert output.output_loss is not None

    def test_return_activations(self, model, sample_inputs):
        output = model(
            inputs=sample_inputs,
            output_types={"reach": "node_mask"},
            num_steps=2,
            return_activations=True,
        )
        assert isinstance(output.activations, dict)

    def test_transformer_processor(self, sample_inputs):
        model = NARModel(
            hidden_dim=64, num_layers=2, num_heads=4, processor_type="transformer"
        )
        output = model(
            inputs=sample_inputs,
            output_types={"reach": "node_mask"},
            num_steps=2,
        )
        assert isinstance(output, NAROutput)

    def test_invalid_processor_type(self):
        with pytest.raises(ValueError, match="Unknown processor type"):
            NARModel(hidden_dim=64, num_layers=2, processor_type="invalid")

    def test_pointer_output(self, model, sample_inputs):
        output = model(
            inputs=sample_inputs,
            output_types={"predecessor": "node_pointer"},
            num_steps=1,
        )
        assert "predecessor" in output.predictions
        pred = output.predictions["predecessor"]
        # Pointer output should have shape (batch, num_nodes, num_nodes) for softmax
        assert pred.dim() == 3
