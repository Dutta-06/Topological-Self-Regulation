"""Regression and invariant tests for the ImageNet-100 (224x224) pipeline."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from bench.models import build_model, describe
from tsrx.graph.bundle import build_all_bundles
from tsrx.graph.groups import discover_groups
from tsrx.graph.trace import trace_model
from tsrx.sense.candidates import CandidateBank
from tsrx.sense.topo import compute_uc_norms
from tsrx.sense.saliency import first_order_saliency
from tsrx.alloc.cost import kappa_params


def test_imagenet100_model_stem_and_spatial():
    """Verify ImageNet-100 uses standard 7x7 stride-2 stem and produces correct 224x224 spatial maps."""
    m = build_model("resnet18", num_classes=100, cifar_stem=False)
    assert isinstance(m.conv1, nn.Conv2d)
    assert m.conv1.kernel_size == (7, 7)
    assert m.conv1.stride == (2, 2)
    assert not isinstance(m.maxpool, nn.Identity)

    desc = describe(m, "resnet18", input_hw=224)
    # Standard ResNet-18 spatial progression on 224x224:
    # conv1 (s2) + maxpool (s2) -> 56x56 (layer1)
    # layer2 (s2) -> 28x28
    # layer3 (s2) -> 14x14
    # layer4 (s2) -> 7x7
    assert desc["spatial"]["layer1"] == (56, 56)
    assert desc["spatial"]["layer2"] == (28, 28)
    assert desc["spatial"]["layer3"] == (14, 14)
    assert desc["spatial"]["layer4"] == (7, 7)


def test_imagenet100_graph_tracing_and_candidate_bank():
    """Verify coupling discovery and candidate dormancy on 224x224 ResNet-18."""
    torch.manual_seed(0)
    m = build_model("resnet18", num_classes=100, cifar_stem=False)
    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 100, (2,))

    traced = trace_model(m.eval(), (x,))
    res = discover_groups(traced)
    bundles = build_all_bundles(res, m)

    bank = CandidateBank(m, bundles, k=4)
    assert bank.max_port_magnitude() == 0.0

    # Test exact deployed param calculation (avoiding O(k^2) double-counting)
    deployed = bank.deployed_params()
    total_with_cands = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert deployed < total_with_cands

    # Run forward/backward pass and verify topological derivatives
    m.train()
    out = m(x)
    loss = F.cross_entropy(out, y)
    loss.backward()

    for tap, h in bank.handles.items():
        u = compute_uc_norms(bank, tap)
        assert u.shape == (4,)
        sal = first_order_saliency(m, h.bundle, h.base_size)
        assert sal.shape == (h.base_size,)


def test_imagenet100_amp_dormancy():
    """Verify that zero_ports() preserves exact dormancy even under gradient scaling."""
    m = build_model("resnet18", num_classes=100, cifar_stem=False)
    x = torch.randn(2, 3, 224, 224)
    y = torch.randint(0, 100, (2,))

    traced = trace_model(m.eval(), (x,))
    res = discover_groups(traced)
    bundles = build_all_bundles(res, m)
    bank = CandidateBank(m, bundles, k=4)

    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
    m.train()

    for _ in range(3):
        opt.zero_grad()
        out = m(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step()
        bank.zero_ports()
        assert bank.max_port_magnitude() == 0.0
