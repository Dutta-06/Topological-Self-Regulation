"""
Unit tests for TSR layers: TSRLinear, TSRConv2d, TSRGroupNorm.

Tests cover:
  - Forward/backward pass shapes and gradient flow
  - Gate masking effect on output
  - Activation mixing gradient flow
  - Structural modifications (prune, grow) and shape consistency
  - Downstream dimension updates (the bug fix)
  - Edge cases (single neuron, empty prune, etc.)
"""

import pytest
import torch
import torch.nn as nn

from tsr.layers.tsr_linear import TSRLinear, ACTIVATION_NAMES
from tsr.layers.tsr_conv import TSRConv2d
from tsr.layers.tsr_norm import TSRGroupNorm
from tsr.layers.tsr_lstm import TSRLSTMCell, TSRLSTM


# ═══════════════════════════════════════════════════════════════════════════════
# TSRLinear Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTSRLinearForward:
    """Test forward pass shapes and basic behavior."""

    def test_output_shape(self):
        layer = TSRLinear(16, 32)
        x = torch.randn(8, 16)
        out = layer(x)
        assert out.shape == (8, 32)

    def test_single_sample(self):
        layer = TSRLinear(10, 5)
        x = torch.randn(1, 10)
        out = layer(x)
        assert out.shape == (1, 5)

    def test_gradient_flow_through_gate(self):
        """Gates must allow gradients to flow to both weights and gate params."""
        layer = TSRLinear(8, 4)
        x = torch.randn(4, 8)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert layer.weight.grad is not None
        assert layer.gate.grad is not None
        assert layer.act_weights.grad is not None

    def test_gradient_flow_through_bias(self):
        layer = TSRLinear(8, 4, bias=True)
        x = torch.randn(4, 8)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.bias.grad is not None

    def test_no_bias(self):
        layer = TSRLinear(8, 4, bias=False)
        assert layer.bias is None
        x = torch.randn(2, 8)
        out = layer(x)
        assert out.shape == (2, 4)


class TestTSRLinearGating:
    """Test that gates actually modulate neuron output."""

    def test_closed_gates_zero_output(self):
        """Gates at -100 (sigmoid ≈ 0) should produce near-zero output."""
        layer = TSRLinear(8, 4, gate_init=-100.0)
        x = torch.randn(2, 8)
        out = layer(x)
        assert out.abs().max().item() < 1e-5

    def test_open_gates_nonzero_output(self):
        """Gates at +100 (sigmoid ≈ 1) should produce nonzero output."""
        layer = TSRLinear(8, 4, gate_init=100.0)
        x = torch.ones(2, 8)
        out = layer(x)
        assert out.abs().max().item() > 0.0

    def test_gate_values_range(self):
        layer = TSRLinear(8, 4, gate_init=3.0)
        gv = layer.gate_values()
        assert (gv >= 0.0).all() and (gv <= 1.0).all()

    def test_effective_neurons_with_mixed_gates(self):
        layer = TSRLinear(8, 4, gate_init=3.0)
        # Close half the gates
        with torch.no_grad():
            layer.gate[:2] = -10.0  # closed
            layer.gate[2:] = 10.0   # open
        assert layer.effective_neurons() == 2


class TestTSRLinearActivationMixing:
    """Test activation mixture mechanism."""

    def test_default_dominant_activation(self):
        layer = TSRLinear(8, 4, act_init="relu")
        assert layer.dominant_activation() == "relu"

    def test_tanh_init(self):
        layer = TSRLinear(8, 4, act_init="tanh")
        assert layer.dominant_activation() == "tanh"

    def test_activation_distribution_sums_to_one(self):
        layer = TSRLinear(8, 4)
        dist = layer.activation_distribution()
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-5

    def test_all_activation_names_present(self):
        layer = TSRLinear(8, 4)
        dist = layer.activation_distribution()
        for name in ACTIVATION_NAMES:
            assert name in dist


class TestTSRLinearStructural:
    """Test structural modification methods (prune/grow)."""

    def test_prune_neurons_shapes(self):
        layer = TSRLinear(16, 8)
        indices = torch.tensor([1, 3, 5])
        layer.prune_neurons(indices)

        assert layer.out_features == 5
        assert layer.weight.shape == (5, 16)
        assert layer.bias.shape == (5,)
        assert layer.gate.shape == (5,)

    def test_prune_empty_does_nothing(self):
        layer = TSRLinear(16, 8)
        layer.prune_neurons(torch.tensor([], dtype=torch.long))
        assert layer.out_features == 8

    def test_grow_neurons_shapes(self):
        layer = TSRLinear(16, 8)
        layer.grow_neurons(4)

        assert layer.out_features == 12
        assert layer.weight.shape == (12, 16)
        assert layer.bias.shape == (12,)
        assert layer.gate.shape == (12,)

    def test_grow_neurons_born_alive(self):
        """New neurons start at newborn_gate=0.0 (sigmoid≈0.5), alive but not open."""
        layer = TSRLinear(16, 8)
        layer.grow_neurons(4)

        gate_vals = layer.gate_values()
        # Original 8 neurons should be ~0.95 (gate_init=3.0)
        assert gate_vals[:8].min().item() > 0.9
        # New 4 neurons: born at logit=0.0 → sigmoid=0.5 (alive, above death threshold)
        assert abs(gate_vals[8:].max().item() - 0.5) < 0.01

    def test_grow_neurons_nonzero_weights(self):
        """New neuron weights must NOT be zero (the bug fix)."""
        layer = TSRLinear(16, 8)
        layer.grow_neurons(4, init_scale=0.001)

        new_weights = layer.weight.data[8:]  # last 4 rows
        # Should not be identically zero
        assert new_weights.abs().sum().item() > 0.0

    def test_prune_input_channels(self):
        layer = TSRLinear(16, 8)
        indices = torch.tensor([0, 5, 10])
        layer.prune_input_channels(indices)

        assert layer.in_features == 13
        assert layer.weight.shape == (8, 13)

    def test_grow_input_channels(self):
        layer = TSRLinear(16, 8)
        layer.grow_input_channels(4)

        assert layer.in_features == 20
        assert layer.weight.shape == (8, 20)

    def test_paired_prune_preserves_forward(self):
        """Pruning upstream + downstream should produce valid forward pass."""
        layer1 = TSRLinear(16, 8)
        layer2 = TSRLinear(8, 4)

        # Prune neurons 1, 3 from layer1 → must update layer2 input
        prune_idx = torch.tensor([1, 3])
        layer1.prune_neurons(prune_idx)
        layer2.prune_input_channels(prune_idx)

        x = torch.randn(2, 16)
        h = layer1(x)
        assert h.shape == (2, 6)
        out = layer2(h)
        assert out.shape == (2, 4)

    def test_paired_grow_preserves_forward(self):
        """Growing upstream + downstream should produce valid forward pass."""
        layer1 = TSRLinear(16, 8)
        layer2 = TSRLinear(8, 4)

        layer1.grow_neurons(4)
        layer2.grow_input_channels(4)

        x = torch.randn(2, 16)
        h = layer1(x)
        assert h.shape == (2, 12)
        out = layer2(h)
        assert out.shape == (2, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# TSRConv2d Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTSRConv2dForward:
    """Test Conv2d forward pass shapes and gradient flow."""

    def test_output_shape(self):
        layer = TSRConv2d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(4, 3, 32, 32)
        out = layer(x)
        assert out.shape == (4, 16, 32, 32)

    def test_output_shape_stride2(self):
        layer = TSRConv2d(3, 16, kernel_size=3, stride=2, padding=1)
        x = torch.randn(4, 3, 32, 32)
        out = layer(x)
        assert out.shape == (4, 16, 16, 16)

    def test_gradient_flow(self):
        layer = TSRConv2d(3, 8, kernel_size=3, padding=1)
        x = torch.randn(2, 3, 16, 16)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert layer.weight.grad is not None
        assert layer.gate.grad is not None
        assert layer.act_weights.grad is not None


class TestTSRConv2dGating:
    """Test channel gating."""

    def test_closed_gates_zero_output(self):
        layer = TSRConv2d(3, 8, kernel_size=3, padding=1, gate_init=-100.0)
        x = torch.randn(2, 3, 16, 16)
        out = layer(x)
        assert out.abs().max().item() < 1e-5

    def test_effective_channels(self):
        layer = TSRConv2d(3, 8, kernel_size=3, padding=1)
        with torch.no_grad():
            layer.gate[:4] = -10.0
            layer.gate[4:] = 10.0
        assert layer.effective_channels() == 4


class TestTSRConv2dStructural:
    """Test structural modifications for Conv2d."""

    def test_prune_channels_shapes(self):
        layer = TSRConv2d(3, 16, kernel_size=3, padding=1)
        layer.prune_channels(torch.tensor([0, 5, 10, 15]))

        assert layer.out_channels == 12
        assert layer.weight.shape == (12, 3, 3, 3)

    def test_grow_channels_shapes(self):
        layer = TSRConv2d(3, 8, kernel_size=3, padding=1)
        layer.grow_channels(4)

        assert layer.out_channels == 12
        assert layer.weight.shape == (12, 3, 3, 3)

    def test_grow_channels_nonzero_weights(self):
        layer = TSRConv2d(3, 8, kernel_size=3, padding=1)
        layer.grow_channels(4, init_scale=0.001)

        new_weights = layer.weight.data[8:]
        assert new_weights.abs().sum().item() > 0.0

    def test_paired_prune_conv_to_conv(self):
        """Prune output channels of layer1 → update input channels of layer2."""
        layer1 = TSRConv2d(3, 16, kernel_size=3, padding=1)
        layer2 = TSRConv2d(16, 32, kernel_size=3, padding=1)

        prune_idx = torch.tensor([0, 5, 10])
        layer1.prune_channels(prune_idx)
        layer2.prune_input_channels(prune_idx)

        x = torch.randn(2, 3, 16, 16)
        h = layer1(x)
        assert h.shape == (2, 13, 16, 16)
        out = layer2(h)
        assert out.shape == (2, 32, 16, 16)


# ═══════════════════════════════════════════════════════════════════════════════
# TSRGroupNorm Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTSRGroupNorm:
    """Test width-invariant normalization."""

    def test_forward_4d(self):
        norm = TSRGroupNorm(16)
        x = torch.randn(4, 16, 8, 8)
        out = norm(x)
        assert out.shape == (4, 16, 8, 8)

    def test_forward_2d(self):
        """Should work with linear layer output (batch, features)."""
        norm = TSRGroupNorm(16)
        x = torch.randn(4, 16)
        out = norm(x)
        assert out.shape == (4, 16)

    def test_resize_preserves_surviving_params(self):
        norm = TSRGroupNorm(16, affine=True)
        # Record original weight for first 8 channels
        original_weight = norm.norm.weight.data[:8].clone()

        norm.resize(24)
        assert norm.num_channels == 24
        # First 8 channels should preserve their affine params
        assert torch.allclose(norm.norm.weight.data[:8], original_weight)

    def test_resize_shrink(self):
        norm = TSRGroupNorm(32)
        norm.resize(16)
        assert norm.num_channels == 16

        x = torch.randn(4, 16, 8, 8)
        out = norm(x)
        assert out.shape == (4, 16, 8, 8)

    def test_small_channel_count(self):
        """Should handle very small channel counts (edge case for minimal seed)."""
        norm = TSRGroupNorm(4, target_group_size=8)
        x = torch.randn(2, 4, 8, 8)
        out = norm(x)
        assert out.shape == (2, 4, 8, 8)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration: Layer + Norm pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class TestLayerNormIntegration:
    """Test TSR layer followed by TSRGroupNorm — the standard building block."""

    def test_conv_norm_forward(self):
        conv = TSRConv2d(3, 16, kernel_size=3, padding=1)
        norm = TSRGroupNorm(16)

        x = torch.randn(4, 3, 32, 32)
        h = conv(x)
        out = norm(h)
        assert out.shape == (4, 16, 32, 32)

    def test_conv_norm_after_grow(self):
        conv = TSRConv2d(3, 16, kernel_size=3, padding=1)
        norm = TSRGroupNorm(16)

        conv.grow_channels(8)
        norm.resize(24)

        x = torch.randn(4, 3, 32, 32)
        h = conv(x)
        out = norm(h)
        assert out.shape == (4, 24, 32, 32)

    def test_conv_norm_after_prune(self):
        conv = TSRConv2d(3, 16, kernel_size=3, padding=1)
        norm = TSRGroupNorm(16)

        conv.prune_channels(torch.tensor([0, 5, 10]))
        norm.resize(13)

        x = torch.randn(4, 3, 32, 32)
        h = conv(x)
        out = norm(h)
        assert out.shape == (4, 13, 32, 32)

    def test_linear_norm_forward(self):
        linear = TSRLinear(64, 32)
        norm = TSRGroupNorm(32)

        x = torch.randn(4, 64)
        h = linear(x)
        out = norm(h)
        assert out.shape == (4, 32)


# ═══════════════════════════════════════════════════════════════════════════════
# TSRLSTM Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTSRLSTMCell:
    """Test structural modifications and shapes for TSRLSTMCell."""
    def test_forward_shape(self):
        cell = TSRLSTMCell(16, 8)
        x = torch.randn(4, 16)
        h, c = cell(x)
        assert h.shape == (4, 8)
        assert c.shape == (4, 8)

    def test_prune_neurons(self):
        cell = TSRLSTMCell(16, 8)
        cell.prune_neurons(torch.tensor([1, 3, 5]))
        assert cell.hidden_size == 5
        assert cell.weight_ih.shape == (20, 16)
        assert cell.weight_hh.shape == (20, 5)

        x = torch.randn(4, 16)
        h, c = cell(x)
        assert h.shape == (4, 5)
        assert c.shape == (4, 5)

    def test_grow_neurons(self):
        cell = TSRLSTMCell(16, 8)
        cell.grow_neurons(4)
        assert cell.hidden_size == 12
        assert cell.weight_ih.shape == (48, 16)
        assert cell.weight_hh.shape == (48, 12)

        x = torch.randn(4, 16)
        h, c = cell(x)
        assert h.shape == (4, 12)
        assert c.shape == (4, 12)

class TestTSRLSTMSequence:
    """Test TSRLSTM sequence wrapper."""
    def test_forward_batch_first(self):
        lstm = TSRLSTM(16, 8, batch_first=True)
        x = torch.randn(4, 10, 16) # batch, seq, feature
        out, (h, c) = lstm(x)
        assert out.shape == (4, 10, 8)
        assert h.shape == (4, 8)
        assert c.shape == (4, 8)


# ═══════════════════════════════════════════════════════════════════════════════
# TSRConv1d Tests
# ═══════════════════════════════════════════════════════════════════════════════

from tsr.layers.tsr_conv1d import TSRConv1d


class TestTSRConv1dForward:
    """Test TSRConv1d forward pass shapes and basic behavior."""

    def test_output_shape(self):
        layer = TSRConv1d(3, 16, kernel_size=3, padding=1)
        x = torch.randn(8, 3, 100)
        out = layer(x)
        assert out.shape == (8, 16, 100)

    def test_gradient_flow(self):
        layer = TSRConv1d(3, 8, kernel_size=3, padding=1)
        x = torch.randn(4, 3, 50)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.weight.grad is not None
        assert layer.gate.grad is not None

    def test_gate_values(self):
        layer = TSRConv1d(3, 8, kernel_size=3, padding=1)
        gates = layer.gate_values()
        assert gates.shape == (8,)
        assert (gates > 0).all()


class TestTSRConv1dStructural:
    """Test TSRConv1d structural modifications."""

    def test_prune_channels(self):
        layer = TSRConv1d(3, 16, kernel_size=3, padding=1)
        indices = torch.tensor([0, 2, 4])
        layer.prune_channels(indices)
        assert layer.out_channels == 13

    def test_grow_channels(self):
        layer = TSRConv1d(3, 16, kernel_size=3, padding=1)
        layer.grow_channels(4, step=100)
        assert layer.out_channels == 20

    def test_prune_input_channels(self):
        layer = TSRConv1d(8, 16, kernel_size=3, padding=1)
        indices = torch.tensor([0, 1])
        layer.prune_input_channels(indices)
        assert layer.in_channels == 6

    def test_grow_input_channels(self):
        layer = TSRConv1d(8, 16, kernel_size=3, padding=1)
        layer.grow_input_channels(4)
        assert layer.in_channels == 12


# ═══════════════════════════════════════════════════════════════════════════════
# TSRMLP Tests (benchmarks/tabular/models)
# ═══════════════════════════════════════════════════════════════════════════════

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmarks.tabular.models.tsr_mlp import TSRMLP


class TestTSRMLPForward:
    """Test TSRMLP forward pass."""

    def test_output_shape_classification(self):
        model = TSRMLP(num_numeric=10, cat_cardinalities=[5, 3], num_out=4, seed_hidden=[16, 16])
        x_num = torch.randn(8, 10)
        x_cat = torch.stack([torch.randint(0, 5, (8,)), torch.randint(0, 3, (8,))], dim=1)
        out = model(x_num, x_cat)
        assert out.shape == (8, 4)

    def test_output_shape_no_categoricals(self):
        model = TSRMLP(num_numeric=20, cat_cardinalities=[], num_out=1, seed_hidden=[32, 32])
        x_num = torch.randn(8, 20)
        x_cat = torch.empty(8, 0, dtype=torch.long)
        out = model(x_num, x_cat)
        assert out.shape == (8, 1)

    def test_insert_block(self):
        model = TSRMLP(num_numeric=10, cat_cardinalities=[], num_out=2, seed_hidden=[16, 16])
        assert len(model.blocks) == 2
        model.insert_block(0)
        assert len(model.blocks) == 3

    def test_topology_state(self):
        model = TSRMLP(num_numeric=10, cat_cardinalities=[], num_out=2, seed_hidden=[16, 32])
        topo = model.topology_state()
        assert topo["depth"] == 2
        assert topo["hidden_sizes"] == [16, 32]


# ═══════════════════════════════════════════════════════════════════════════════
# TSRTCN Tests (benchmarks/timeseries/models)
# ═══════════════════════════════════════════════════════════════════════════════

from benchmarks.timeseries.models.tsr_tcn import TSRTCNForecaster, TSRTCNClassifier


class TestTSRTCNForecaster:
    """Test TSRTCNForecaster forward pass."""

    def test_output_shape(self):
        model = TSRTCNForecaster(input_size=7, pred_len=96, seed_channels=32, num_levels=3)
        x = torch.randn(8, 96, 7)
        out = model(x)
        assert out.shape == (8, 96, 7)

    def test_insert_block(self):
        model = TSRTCNForecaster(input_size=7, pred_len=96, seed_channels=32, num_levels=3)
        assert len(model.encoder.blocks) == 3
        model.insert_block(1)
        assert len(model.encoder.blocks) == 4

    def test_topology_state(self):
        model = TSRTCNForecaster(input_size=7, pred_len=96, seed_channels=32, num_levels=3)
        topo = model.topology_state()
        assert topo["num_blocks"] == 3
        assert topo["seed_channels"] == 32


class TestTSRTCNClassifier:
    """Test TSRTCNClassifier forward pass."""

    def test_output_shape(self):
        model = TSRTCNClassifier(input_size=9, num_classes=6, seed_channels=32, num_levels=3)
        x = torch.randn(8, 100, 9)
        out = model(x)
        assert out.shape == (8, 6)

    def test_insert_block(self):
        model = TSRTCNClassifier(input_size=9, num_classes=6, seed_channels=32, num_levels=3)
        assert len(model.encoder.blocks) == 3
        model.insert_block(0)
        assert len(model.encoder.blocks) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
