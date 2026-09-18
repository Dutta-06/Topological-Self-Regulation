"""Regression tests for the tsrx-time (Stage 1) claim: a Conv1d/BatchNorm1d
TCN needs ZERO engine changes — same dormancy invariant, same exact param
accounting, same cost model — as long as the tensor layout is (B, C, L)
where channels are axis 1. Mirrors tests_tsrx/test_invariants.py's pattern
for the vision engine. Run: python -m pytest tests_tsrx/ -q
"""

import copy

import pytest
import torch
import torch.nn.functional as F

from bench.ts_models import TCNForecaster
from tsrx.alloc.cost import kappa_flops, kappa_params
from tsrx.graph.bundle import build_all_bundles
from tsrx.graph.groups import discover_groups
from tsrx.graph.trace import trace_model
from tsrx.sense.candidates import CandidateBank


def _attach(k=4, n_vars=7, pred_len=24, hidden=32):
    torch.manual_seed(0)
    m = TCNForecaster(n_vars, pred_len, hidden=hidden)
    x = torch.randn(4, 48, n_vars)  # (B, L, C) — the loader's layout
    traced = trace_model(m.eval(), (x[:2],))
    res = discover_groups(traced)
    bundles = build_all_bundles(res, m)
    return m, CandidateBank(m, bundles, k=k), x, traced


def _detached_copy(model, bank):
    m2 = copy.deepcopy(model)
    b2 = copy.copy(bank)
    object.__setattr__(b2, "model", m2)
    b2.handles = {t: copy.copy(h) for t, h in bank.handles.items()}
    for _, h in b2.handles.items():
        h.bundle = copy.deepcopy(h.bundle)
    return b2.detach()


def test_tcn_groups_discovered_and_output_head_skipped():
    m, bank, x, _ = _attach()
    # 5 attachable groups: one per-block internal bottleneck (conv1->conv2,
    # 4 blocks) plus ONE shared residual-stream group joining every block's
    # conv2/downsample output via the residual add (consumed by every later
    # block's conv1 AND the head) — the same two-tap-per-region pattern as
    # ResNet's own "layerX.Y.conv1" + "layerX.downsample" coupling groups.
    # The head's own output group (no consumer) must NOT get candidates —
    # never grow the model's output width.
    assert len(bank.handles) == 5


@pytest.mark.parametrize("k", [2, 4])
def test_deployed_params_exact(k):
    m, bank, x, _ = _attach(k=k)
    truth = sum(p.numel() for p in _detached_copy(m, bank).parameters() if p.requires_grad)
    assert bank.deployed_params() == truth


def test_candidates_dormant_at_attach():
    m, bank, x, _ = _attach()
    m.eval()
    with torch.no_grad():
        full = m(x)
        stripped = _detached_copy(m, bank).eval()(x)
    assert torch.allclose(full, stripped, atol=1e-5)
    assert bank.max_port_magnitude() == 0.0


def test_candidates_stay_dormant_under_training():
    """THE critical invariant (see [[tsr-dormant-sensor-failure-mode]]):
    candidate ports carry a real gradient and must be re-zeroed after
    every optimizer step, or they silently become live channels."""
    m, bank, x, _ = _attach()
    y = torch.randn(4, 24, 7)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

    for _ in range(25):
        m.train()
        opt.zero_grad()
        F.mse_loss(m(x), y).backward()
        opt.step()
        bank.zero_ports()

        assert bank.max_port_magnitude() == 0.0
        m.eval()
        with torch.no_grad():
            full = m(x)
            stripped = _detached_copy(m, bank).eval()(x)
        assert torch.allclose(full, stripped, atol=1e-5), "candidates leaked into the forward pass"


def test_ports_drift_without_zeroing():
    m, bank, x, _ = _attach()
    y = torch.randn(4, 24, 7)
    m.train()
    assert bank.max_port_magnitude() == 0.0
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        F.mse_loss(m(x), y).backward()
        opt.step()          # deliberately NO zero_ports()
    assert bank.max_port_magnitude() > 0.0


def test_kappa_flops_sane_on_channel_axis_1():
    """cost.py's _spatial_size assumes axis 1 is channels and prices
    everything after it as spatial — correct for (B, C, L) but would be
    silently wrong on (B, L, C) (Stage 2's blocker, not exercised here)."""
    m, bank, x, traced = _attach(hidden=32)
    for tap, h in bank.handles.items():
        kp = kappa_params(h.bundle, m)
        kf = kappa_flops(h.bundle, m, traced)
        assert kp > 0
        assert kf > 0
        # A 1-channel Conv1d cost should scale with kernel/dilation and the
        # OTHER conv's channel count, not with sequence length collapsing
        # to a constant regardless of model width — cheap sanity check that
        # FLOPs actually reflect the conv's real fan-in/out.
        assert kf > kp  # every block here has kernel=3 and hidden>=32, so FLOPs > params per index
