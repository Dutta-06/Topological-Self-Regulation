"""Regression tests for the TSR-X invariants that were found broken.

Every test here corresponds to a defect that silently invalidated a
benchmark result. Run: python -m pytest tests_tsrx/ -q
"""

import copy

import pytest
import torch
import torch.nn.functional as F

from bench.models import build_model
from tsrx.graph.bundle import build_all_bundles
from tsrx.graph.groups import discover_groups
from tsrx.graph.trace import trace_model
from tsrx.sense.candidates import CandidateBank


def _attach(k=4, num_classes=10, cifar_stem=True):
    torch.manual_seed(0)
    m = build_model("resnet18", num_classes, cifar_stem=cifar_stem)
    x = torch.randn(4, 3, 32, 32)
    traced = trace_model(m.eval(), (x[:2],))
    res = discover_groups(traced)
    bundles = build_all_bundles(res, m)
    return m, CandidateBank(m, bundles, k=k), x


def _detached_copy(model, bank):
    m2 = copy.deepcopy(model)
    b2 = copy.copy(bank)
    object.__setattr__(b2, "model", m2)
    b2.handles = {t: copy.copy(h) for t, h in bank.handles.items()}
    for _, h in b2.handles.items():
        h.bundle = copy.deepcopy(h.bundle)
    return b2.detach()


@pytest.mark.parametrize("k", [2, 4, 8])
def test_deployed_params_exact(k):
    """deployed_params() must equal the true post-detach count.

    The old subtractive form double-counted the candidate x candidate
    corner of any module that both consumes one candidate group and
    produces another (-2,352 params at k=4, growing as O(k^2)).
    """
    m, bank, _ = _attach(k=k)
    truth = sum(p.numel() for p in _detached_copy(m, bank).parameters() if p.requires_grad)
    assert bank.deployed_params() == truth


def test_candidates_dormant_at_attach():
    m, bank, x = _attach()
    m.eval()
    with torch.no_grad():
        full = m(x)
        stripped = _detached_copy(m, bank).eval()(x)
    assert torch.allclose(full, stripped, atol=1e-5)
    assert bank.max_port_magnitude() == 0.0


def test_candidates_stay_dormant_under_training():
    """THE critical invariant. Ports carry a real gradient (u_c IS that
    gradient), so any optimizer trains them off zero unless re-zeroed.
    Without zero_ports() the measured drift was 5.3 logits after ONE step
    and ~1e6 after ten — candidates become live channels, u_c stops being
    the topological derivative, and accuracy gets measured on a model
    that deployed_params() does not count.
    """
    m, bank, x = _attach()
    y = torch.randint(0, 10, (4,))
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

    for _ in range(25):
        m.train()
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()
        bank.zero_ports()

        assert bank.max_port_magnitude() == 0.0
        m.eval()
        with torch.no_grad():
            full = m(x)
            stripped = _detached_copy(m, bank).eval()(x)
        assert torch.allclose(full, stripped, atol=1e-5), "candidates leaked into the forward pass"


def test_ports_drift_without_zeroing():
    """Guard the guard: confirm the failure mode is real, so this test
    suite would actually catch a regression that removes zero_ports()."""
    m, bank, x = _attach()
    y = torch.randint(0, 10, (4,))
    m.train()
    assert bank.max_port_magnitude() == 0.0
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    for _ in range(3):
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        opt.step()          # deliberately NO zero_ports()
    assert bank.max_port_magnitude() > 0.0


def test_uc_matches_finite_difference():
    """u_c read off .grad must equal the true directional derivative."""
    torch.set_default_dtype(torch.float64)
    try:
        m, bank, x = _attach()
        m = m.double()
        x = x.double()
        y = torch.randint(0, 10, (4,))
        m.train()
        F.cross_entropy(m(x), y).backward()

        tap = next(iter(bank.handles))
        h = bank.handles[tap]
        cons = dict(m.named_modules())[h.bundle.consumer_slots[0].module_name]
        base, c = h.base_size, 0
        col = slice(base + c, base + c + 1)

        orig = cons.weight.data[:, col].clone()
        v = torch.randn_like(orig)
        v /= v.norm()

        def loss_at(t):
            with torch.no_grad():
                cons.weight.data[:, col] = orig + t * v
            out = F.cross_entropy(m(x), y).item()
            with torch.no_grad():
                cons.weight.data[:, col] = orig
            return out

        eps = 1e-6
        fd = (loss_at(eps) - loss_at(-eps)) / (2 * eps)
        pred = (cons.weight.grad[:, col] * v).sum().item()
        assert abs(fd - pred) < 1e-7, f"fd={fd} pred={pred}"
    finally:
        torch.set_default_dtype(torch.float32)


def test_cifar_stem_keeps_spatial_resolution():
    """The ImageNet stem crushes 32x32 -> 8x8, leaving 75% of ResNet-18's
    params on a 1x1 map. That caps accuracy at ~87% and manufactures
    artificial pruning slack that flatters any pruning method."""
    from bench.models import describe

    bad = describe(build_model("resnet18", 10, cifar_stem=False), "resnet18")
    good = describe(build_model("resnet18", 10, cifar_stem=True), "resnet18")
    assert bad["spatial"]["layer4"] == (1, 1)
    assert good["spatial"]["layer4"] == (4, 4)
    assert good["spatial"]["layer1"] == (32, 32)
