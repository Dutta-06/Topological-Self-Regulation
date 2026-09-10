"""Regression and invariant tests for MobileNetV2 pipeline on CIFAR/vision benchmarks."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bench.models import build_model, describe
from bench.train_static_matched import resize_model_to_widths
from tsrx.edit.edits import prune_group_index, materialize_candidate
from tsrx.graph.bundle import build_all_bundles
from tsrx.graph.generators import is_depthwise
from tsrx.graph.groups import discover_groups
from tsrx.graph.trace import trace_model
from tsrx.sense.candidates import CandidateBank
from tsrx.sense.topo import compute_uc_norms
from tsrx.sense.saliency import first_order_saliency


def test_mobilenet_model_architecture():
    """Verify MobileNetV2 uses cifar_stem (stride=1) and produces ~2.2M params."""
    m = build_model("mobilenet_v2", num_classes=10, cifar_stem=True)
    total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)

    # Standard MobileNetV2 has ~2.23M params
    assert 2_000_000 < total_params < 2_500_000, f"Expected ~2.2M params, got {total_params}"
    assert m.features[0][0].stride == (1, 1), f"Expected stem stride (1, 1), got {m.features[0][0].stride}"
    assert m.classifier[1].out_features == 10

    desc = describe(m, "mobilenet_v2", input_hw=32)
    spatial = desc["spatial"]
    assert spatial["stage1"] == (32, 32)
    assert spatial["stage2"] == (16, 16)
    assert spatial["stage3"] == (8, 8)
    assert spatial["stage4"] == (4, 4)
    assert spatial["stage5"] == (2, 2)


def test_mobilenet_graph_tracing_and_candidate_bank():
    """Verify coupling discovery, candidate bank attachment, and exact zero-port dormancy."""
    torch.manual_seed(0)
    m = build_model("mobilenet_v2", num_classes=10, cifar_stem=True)
    x = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, 10, (2,))

    m.eval()
    y_orig = m(x)

    traced = trace_model(m, (x,))
    res = discover_groups(traced)
    bundles = build_all_bundles(res, m)

    bank = CandidateBank(m, bundles, k=4)
    assert len(bank.handles) > 0
    assert bank.max_port_magnitude() == 0.0

    # Lemma 2.2: dormancy preserves exact function output
    y_cand = m(x)
    max_drift = (y_orig - y_cand).abs().max().item()
    assert max_drift < 1e-12, f"Expected near-zero drift, got {max_drift}"

    # Verify deployed param calculation
    deployed = bank.deployed_params()
    total_with_cands = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert deployed < total_with_cands

    # Run forward/backward pass and verify topological derivatives & saliencies
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)
    opt.zero_grad()
    out = m(x)
    loss = F.cross_entropy(out, y)
    loss.backward()

    for tap, h in bank.handles.items():
        u = compute_uc_norms(bank, tap)
        assert u.shape == (4,)
        sal = first_order_saliency(m, h.bundle, h.base_size)
        assert sal.shape == (h.base_size,)

    # Step optimizer and re-zero ports: verify strict dormancy
    opt.step()
    bank.zero_ports()
    assert bank.max_port_magnitude() == 0.0


def test_mobilenet_edits_and_depthwise_surgeries():
    """Verify pruning and materialization across depthwise inverted residual blocks."""
    torch.manual_seed(0)
    m = build_model("mobilenet_v2", num_classes=10, cifar_stem=True)
    x = torch.randn(2, 3, 32, 32)

    traced = trace_model(m.eval(), (x,))
    res = discover_groups(traced)
    bundles = build_all_bundles(res, m)
    bank = CandidateBank(m, bundles, k=4)
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)

    # Find a group containing a depthwise convolution
    modules = dict(m.named_modules())
    dw_tap = None
    for tap, bd in bundles.items():
        for s in bd.producer_slots:
            mod = modules[s.module_name]
            if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and is_depthwise(mod):
                dw_tap = tap
                break
        if dw_tap is not None:
            break

    assert dw_tap is not None, "Could not find a depthwise coupling group"
    bd = bundles[dw_tap]
    h = bank.handles[dw_tap]
    orig_size = h.base_size

    # Prune one channel
    prune_group_index(m, bd, 0, optimizer=opt, bank=bank)
    assert h.base_size == orig_size - 1
    assert bd.size == orig_size - 1

    # Verify forward pass succeeds after prune
    m.eval()
    out1 = m(x)
    assert out1.shape == (2, 10)

    # Materialize one candidate back
    materialize_candidate(bank, dw_tap, 0, eps=1e-3, optimizer=opt)
    assert h.base_size == orig_size
    assert bd.size == orig_size

    # Verify forward pass succeeds after materialize
    out2 = m(x)
    assert out2.shape == (2, 10)

    # Optimizer step works cleanly
    m.train()
    loss = m(x).sum()
    loss.backward()
    opt.step()


def test_mobilenet_resize_model_to_widths():
    """Verify C2 static matched control: rebuilding MobileNetV2 to custom channel widths."""
    torch.manual_seed(0)
    m = build_model("mobilenet_v2", num_classes=10, cifar_stem=True)
    x = torch.randn(2, 3, 32, 32)

    traced = trace_model(m.eval(), (x,))
    res = discover_groups(traced)
    bundles = build_all_bundles(res, m)
    bank = CandidateBank(m, bundles, k=4)

    # Scale each group width down slightly (e.g. max(8, base - 2))
    target_widths = {str(t): max(8, h.base_size - 2) for t, h in bank.handles.items()}

    m2 = build_model("mobilenet_v2", num_classes=10, cifar_stem=True)
    m2 = resize_model_to_widths(m2, target_widths, x)

    m2.eval()
    with torch.no_grad():
        out2 = m2(x)
    assert out2.shape == (2, 10)
    assert not torch.isnan(out2).any()
