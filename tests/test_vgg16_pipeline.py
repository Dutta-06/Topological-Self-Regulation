"""Regression and invariant tests for the VGG-16 (BatchNorm) pipeline on CIFAR/vision benchmarks."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bench.models import build_model, describe
from bench.train_static_matched import resize_model_to_widths
from tsrx.graph.bundle import build_all_bundles
from tsrx.graph.groups import discover_groups
from tsrx.graph.trace import trace_model
from tsrx.sense.candidates import CandidateBank
from tsrx.sense.topo import compute_uc_norms
from tsrx.sense.saliency import first_order_saliency


def test_vgg16_cifar_model_architecture():
    """Verify VGG-16-BN uses AdaptiveAvgPool2d((1, 1)) and clean 512->C linear head (~14.7M params)."""
    m = build_model("vgg16_bn", num_classes=10)
    total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    
    # Standard torchvision VGG has 134M params because of 119.5M dense weights.
    # Our standardized conv-focused head has ~14.7M params:
    assert 14_000_000 < total_params < 16_000_000, f"Expected ~14.7M params, got {total_params}"
    assert isinstance(m.avgpool, nn.AdaptiveAvgPool2d)
    assert m.avgpool.output_size == (1, 1) or m.avgpool.output_size == 1
    
    # Verify spatial resolution progression across the 5 pooling layers
    desc = describe(m, "vgg16_bn", input_hw=32)
    spatial = desc["spatial"]
    assert spatial["pool1"] == (16, 16)
    assert spatial["pool2"] == (8, 8)
    assert spatial["pool3"] == (4, 4)
    assert spatial["pool4"] == (2, 2)
    assert spatial["pool5"] == (1, 1)


def test_vgg16_graph_tracing_and_candidate_bank():
    """Verify coupling discovery, 13-stage candidate bank attachment, and zero-port dormancy."""
    torch.manual_seed(0)
    m = build_model("vgg16_bn", num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = torch.randint(0, 10, (2,))

    traced = trace_model(m.eval(), (x,))
    res = discover_groups(traced)
    bundles = build_all_bundles(res, m)

    bank = CandidateBank(m, bundles, k=4)
    # VGG-16 has 13 convolutional layers (plus final linear layer bundle)
    assert len(bank.handles) == 13
    assert bank.max_port_magnitude() == 0.0

    # Test deployed param calculation
    deployed = bank.deployed_params()
    total_with_cands = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert deployed < total_with_cands

    # Run forward/backward pass and verify topological derivatives & saliencies
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
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


def test_vgg16_resize_model_to_widths():
    """Verify C2 static matched control: resizing VGG-16 to custom channel widths."""
    torch.manual_seed(0)
    m = build_model("vgg16_bn", num_classes=10)
    x = torch.randn(2, 3, 32, 32)

    traced = trace_model(m.eval(), (x,))
    res = discover_groups(traced)
    bundles = build_all_bundles(res, m)
    bank = CandidateBank(m, bundles, k=4)

    # Pick a target width dict where every group is reduced by 4 channels
    target_widths = {str(t): max(8, h.base_size - 4) for t, h in bank.handles.items()}

    m2 = build_model("vgg16_bn", num_classes=10)
    m2 = resize_model_to_widths(m2, target_widths, x)

    # Verify resized model runs forward pass and produces valid logits
    m2.eval()
    with torch.no_grad():
        out2 = m2(x)
    assert out2.shape == (2, 10)
    
    # Total params of m2 should be less than m
    p_orig = sum(p.numel() for p in m.parameters() if p.requires_grad)
    p_resized = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    assert p_resized < p_orig
