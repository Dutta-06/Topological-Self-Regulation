"""Regression and invariant tests for EfficientNet-B0 pipeline on CIFAR/vision benchmarks."""

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


def test_efficientnet_model_architecture():
    """Verify EfficientNet-B0 uses cifar_stem (stride=1) and produces ~4.0M params."""
    m = build_model("efficientnet_b0", num_classes=10, cifar_stem=True)
    total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)

    assert 3_800_000 < total_params < 4_500_000, f"Expected ~4.0M params, got {total_params}"
    assert m.features[0][0].stride == (1, 1), f"Expected stem stride (1, 1), got {m.features[0][0].stride}"
    assert m.classifier[1].out_features == 10

    desc = describe(m, "efficientnet_b0", input_hw=32)
    spatial = desc["spatial"]
    assert spatial["stage1"] == (32, 32)
    assert spatial["stage2"] == (16, 16)
    assert spatial["stage3"] == (8, 8)
    assert spatial["stage4"] == (4, 4)
    assert spatial["stage5"] == (2, 2)


def test_efficientnet_graph_tracing_and_candidate_bank():
    """Verify coupling discovery, candidate bank attachment, and exact zero-port dormancy on SE blocks."""
    torch.manual_seed(0)
    m = build_model("efficientnet_b0", num_classes=10, cifar_stem=True)
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

    # Lemma 2.2: dormancy preserves exact function output across MBConv and SE layers
    y_cand = m(x)
    max_drift = (y_orig - y_cand).abs().max().item()
    assert max_drift < 1e-12, f"Expected near-zero drift, got {max_drift}"

    deployed = bank.deployed_params()
    total_with_cands = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert deployed < total_with_cands

    # Run forward/backward pass
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

    opt.step()
    bank.zero_ports()
    assert bank.max_port_magnitude() == 0.0


def test_efficientnet_edits_and_surgeries():
    """Verify pruning and materialization across MBConv blocks."""
    torch.manual_seed(0)
    m = build_model("efficientnet_b0", num_classes=10, cifar_stem=True)
    x = torch.randn(2, 3, 32, 32)

    traced = trace_model(m.eval(), (x,))
    res = discover_groups(traced)
    bundles = build_all_bundles(res, m)
    bank = CandidateBank(m, bundles, k=4)
    opt = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)

    tap = list(bank.handles.keys())[2]
    bd = bundles[tap]
    h = bank.handles[tap]
    orig_size = h.base_size

    # Prune one channel
    prune_group_index(m, bd, 0, optimizer=opt, bank=bank)
    assert h.base_size == orig_size - 1
    assert bd.size == orig_size - 1

    m.eval()
    out1 = m(x)
    assert out1.shape == (2, 10)

    # Materialize candidate
    materialize_candidate(bank, tap, 0, eps=1e-3, optimizer=opt)
    assert h.base_size == orig_size
    assert bd.size == orig_size

    out2 = m(x)
    assert out2.shape == (2, 10)


def test_efficientnet_resize_model_to_widths():
    """Verify C2 static matched control reconstruction for EfficientNet-B0."""
    torch.manual_seed(0)
    m = build_model("efficientnet_b0", num_classes=10, cifar_stem=True)
    x = torch.randn(2, 3, 32, 32)

    traced = trace_model(m.eval(), (x,))
    res = discover_groups(traced)
    bundles = build_all_bundles(res, m)
    bank = CandidateBank(m, bundles, k=4)

    target_widths = {str(t): max(8, h.base_size - 2) for t, h in bank.handles.items()}

    m2 = build_model("efficientnet_b0", num_classes=10, cifar_stem=True)
    m2 = resize_model_to_widths(m2, target_widths, x)

    m2.eval()
    with torch.no_grad():
        out2 = m2(x)
    assert out2.shape == (2, 10)
    assert not torch.isnan(out2).any()
