"""
Unit tests for the TSR regulation engine: monitor, signals, actions, scheduler.

Tests cover:
  - Monitor hook registration and statistics collection
  - Death signal computation (correctly identifies dead neurons)
  - Bottleneck signal computation
  - Paired prune/grow actions (the critical downstream dimension fix)
  - Optimizer rebuild after structural changes
  - Scheduler timing and cooldown
"""

import pytest
import torch
import torch.nn as nn

from tsr.layers.tsr_linear import TSRLinear
from tsr.layers.tsr_conv import TSRConv2d
from tsr.layers.tsr_norm import TSRGroupNorm
from tsr.model import TSRNetwork
from tsr.regulation.monitor import StructuralPlasticityMonitor
from tsr.regulation.signals import (
    compute_death_signal,
    compute_bottleneck_signal,
    compute_growth_neurons,
)
from tsr.regulation.actions import (
    prune_neurons_paired,
    grow_neurons_paired,
    apply_structural_update,
)
from tsr.regulation.scheduler import StructuralUpdateScheduler
from tsr.utils import rebuild_optimizer, count_parameters


class TestMonitor:
    """Test the StructuralPlasticityMonitor."""

    def _make_model(self):
        return TSRNetwork(in_channels=3, seed_channels=[8, 8], num_classes=10)

    def test_hooks_registered(self):
        model = self._make_model()
        monitor = StructuralPlasticityMonitor(model, window=10)

        # Should have registered hooks for each TSR layer
        tsr_layers = monitor.get_tsr_layers()
        assert len(tsr_layers) > 0
        assert len(monitor.layer_stats) == len(tsr_layers)

        monitor.remove_hooks()

    def test_stats_collected_after_forward(self):
        model = self._make_model()
        monitor = StructuralPlasticityMonitor(model, window=5)

        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # All layers should have collected 1 forward pass of stats
        for name, stats in monitor.get_all_stats().items():
            assert len(stats.activation_magnitudes) == 1
            assert len(stats.gradient_magnitudes) == 1

        monitor.remove_hooks()

    def test_window_overflow(self):
        model = self._make_model()
        monitor = StructuralPlasticityMonitor(model, window=3)

        # Run 5 forward+backward passes → window should keep only last 3
        for _ in range(5):
            x = torch.randn(4, 3, 32, 32)
            out = model(x)
            loss = out.sum()
            loss.backward()

        for name, stats in monitor.get_all_stats().items():
            assert len(stats.activation_magnitudes) <= 3

        monitor.remove_hooks()

    def test_is_ready(self):
        model = self._make_model()
        monitor = StructuralPlasticityMonitor(model, window=3)

        assert not monitor.is_ready()

        for _ in range(3):
            x = torch.randn(4, 3, 32, 32)
            out = model(x)
            loss = out.sum()
            loss.backward()

        assert monitor.is_ready()
        monitor.remove_hooks()

    def test_loss_plateau_detection(self):
        model = self._make_model()
        monitor = StructuralPlasticityMonitor(model, window=100)

        # Simulate plateau: constant loss
        for _ in range(60):
            monitor.record_loss(1.5)

        assert monitor.is_loss_plateau(patience=50, threshold=0.001)

        monitor.remove_hooks()


class TestDeathSignal:
    """Test death signal computation."""

    def test_detects_dead_neurons_linear(self):
        layer = TSRLinear(16, 8, gate_init=3.0)

        # Kill 3 neurons: set gates to very negative
        with torch.no_grad():
            layer.gate[0] = -10.0
            layer.gate[3] = -10.0
            layer.gate[5] = -10.0

        # Create fake stats with low activations for dead neurons
        from tsr.regulation.monitor import LayerStats
        stats = LayerStats(window_size=10)
        for _ in range(10):
            mag = torch.ones(8) * 0.5
            mag[0] = 0.001
            mag[3] = 0.001
            mag[5] = 0.001
            stats.add_activation_stats(mag, torch.zeros(8))

        dead = compute_death_signal(stats, layer, threshold=0.01, min_neurons=4)
        assert set(dead.tolist()) == {0, 3, 5}

    def test_respects_min_neurons(self):
        layer = TSRLinear(16, 4, gate_init=-10.0)  # all gates nearly off

        from tsr.regulation.monitor import LayerStats
        stats = LayerStats(window_size=10)
        for _ in range(10):
            stats.add_activation_stats(
                torch.ones(4) * 0.001, torch.zeros(4)
            )

        # min_neurons=4 → can't prune any (would drop below 4)
        dead = compute_death_signal(stats, layer, threshold=0.01, min_neurons=4)
        assert len(dead) == 0

    def test_no_false_positives_active_neurons(self):
        layer = TSRLinear(16, 8, gate_init=3.0)

        from tsr.regulation.monitor import LayerStats
        stats = LayerStats(window_size=10)
        for _ in range(10):
            stats.add_activation_stats(
                torch.ones(8) * 0.5, torch.zeros(8)
            )

        dead = compute_death_signal(stats, layer, threshold=0.01, min_neurons=4)
        assert len(dead) == 0


class TestBottleneckSignal:
    """Test bottleneck (growth) signal computation."""

    def test_high_gradient_high_utilization(self):
        layer = TSRLinear(16, 8, gate_init=10.0)  # all neurons very active

        from tsr.regulation.monitor import LayerStats
        stats = LayerStats(window_size=10)
        for _ in range(10):
            stats.add_gradient_stats(
                torch.ones(8) * 5.0,  # high gradient
                torch.ones(8) * 0.1,  # low variance (uniform)
            )
            stats.add_activation_stats(
                torch.ones(8) * 2.0,  # high activation
                torch.ones(8) * 0.1,
            )

        score = compute_bottleneck_signal(stats, layer)
        assert score > 0.0

    def test_growth_neurons_below_threshold(self):
        n = compute_growth_neurons(
            bottleneck_score=0.5,
            current_width=16,
            bottleneck_threshold=1.5,
        )
        assert n == 0  # below threshold

    def test_growth_neurons_above_threshold(self):
        n = compute_growth_neurons(
            bottleneck_score=2.0,
            current_width=16,
            growth_rate=0.1,
            bottleneck_threshold=1.5,
        )
        assert n >= 1  # should grow


class TestPairedActions:
    """Test that paired prune/grow correctly updates downstream layers."""

    def _make_model(self):
        return TSRNetwork(in_channels=3, seed_channels=[16, 16], num_classes=10)

    def test_paired_prune_shapes(self):
        model = self._make_model()
        block0_conv = model.blocks[0].conv
        block1_conv = model.blocks[1].conv

        old_out = block0_conv.out_channels
        old_in_next = block1_conv.in_channels
        assert old_in_next == old_out  # they should match

        # Prune 3 channels from block 0
        prune_idx = torch.tensor([0, 5, 10])
        event = prune_neurons_paired(model, "blocks.0.conv", prune_idx, step=100)

        assert event is not None
        assert block0_conv.out_channels == old_out - 3
        assert block1_conv.in_channels == old_out - 3  # downstream updated!

        # Forward pass should work
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_paired_grow_shapes(self):
        model = self._make_model()
        block0_conv = model.blocks[0].conv
        block1_conv = model.blocks[1].conv

        old_out = block0_conv.out_channels
        old_in_next = block1_conv.in_channels

        event = grow_neurons_paired(model, "blocks.0.conv", n=4, step=100)

        assert event is not None
        assert block0_conv.out_channels == old_out + 4
        assert block1_conv.in_channels == old_out + 4  # downstream updated!

        # Forward pass should work
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_prune_then_grow_round_trip(self):
        model = self._make_model()

        # Prune 2 from block 0
        prune_neurons_paired(model, "blocks.0.conv", torch.tensor([0, 1]), step=100)
        # Grow 2 back
        grow_neurons_paired(model, "blocks.0.conv", n=2, step=200)

        # Forward pass should work
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)


class TestOptimizerRebuild:
    """Test optimizer rebuild preserves training stability."""

    def test_rebuild_creates_correct_groups(self):
        model = TSRNetwork(in_channels=3, seed_channels=[8, 8], num_classes=10)
        optimizer = rebuild_optimizer(
            model, torch.optim.Adam, {"lr": 0.001},
            gate_lr_multiplier=1.0, act_lr_multiplier=0.1,
        )

        # Should have 3 param groups: weights, gates, activations
        assert len(optimizer.param_groups) == 3

        group_names = [g.get("group_name") for g in optimizer.param_groups]
        assert "weights" in group_names
        assert "gates" in group_names
        assert "activations" in group_names

    def test_rebuild_after_grow(self):
        model = TSRNetwork(in_channels=3, seed_channels=[8, 8], num_classes=10)
        optimizer = rebuild_optimizer(
            model, torch.optim.Adam, {"lr": 0.001},
        )

        # Train for a few steps to build optimizer state
        x = torch.randn(4, 3, 32, 32)
        for _ in range(3):
            optimizer.zero_grad()
            out = model(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()

        # Grow and rebuild
        grow_neurons_paired(model, "blocks.0.conv", n=4, step=100)

        new_optimizer = rebuild_optimizer(
            model, torch.optim.Adam, {"lr": 0.001},
            old_optimizer=optimizer,
        )

        # Should still work
        new_optimizer.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()
        new_optimizer.step()  # no crash


class TestScheduler:
    """Test structural update scheduling."""

    def test_periodic_update(self):
        scheduler = StructuralUpdateScheduler(update_interval=100)

        assert scheduler.should_update(0)  # allow first update
        scheduler.record_update(0, [])

        assert not scheduler.should_update(50)
        assert scheduler.should_update(100)

    def test_cooldown(self):
        scheduler = StructuralUpdateScheduler(
            update_interval=100, cooldown_steps=50
        )

        scheduler.record_update(0, ["layer_a"])

        assert not scheduler.is_layer_cooled_down("layer_a", 30)
        assert scheduler.is_layer_cooled_down("layer_a", 60)
        assert scheduler.is_layer_cooled_down("layer_b", 10)  # never modified

    def test_state_dict_round_trip(self):
        scheduler = StructuralUpdateScheduler(update_interval=100)
        scheduler.record_update(100, ["layer_a", "layer_b"])

        state = scheduler.state_dict()
        new_scheduler = StructuralUpdateScheduler(update_interval=200)
        new_scheduler.load_state_dict(state)

        assert new_scheduler.current_interval == scheduler.current_interval
        assert new_scheduler._last_global_update == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
