"""Tests for the channel-independent TCN and the C3 random control.

Two things are being guarded here:

1. `tcn_ci` adds RevIN, and RevIN is a normalization layer. This project has
   twice shipped a benchmark invalidated by a "dormant" sensor that was not
   actually dormant ([[tsr-dormant-sensor-failure-mode]]), and LayerNorm is
   already known to break condition (N) because its statistics span the
   candidate channels. RevIN reduces over TIME per (batch, variate) and holds
   no parameters on the reallocated channel axis, so it should be safe — but
   "should be" is exactly the reasoning that failed before, so it is tested.

2. C3 is only a control if it actually matches the target parameter count.
"""

import copy

import pytest
import torch
import torch.nn.functional as F

from bench.c3_random import attachable_taps, build_c3_model, random_widths_at_budget
from bench.ts_models import build_ts_model
from tsrx.graph.bundle import build_all_bundles
from tsrx.graph.groups import discover_groups
from tsrx.graph.trace import trace_model
from tsrx.sense.candidates import CandidateBank


def _attach_ci(k=4, n_vars=21, pred_len=24, hidden=32, use_revin=True):
    torch.manual_seed(0)
    m = build_ts_model("tcn_ci", n_vars, pred_len, hidden=hidden, use_revin=use_revin)
    x = torch.randn(4, 48, n_vars)
    traced = trace_model(m.eval(), (x[:2],))
    bundles = build_all_bundles(discover_groups(traced), m)
    return m, CandidateBank(m, bundles, k=k), x


def _detached_copy(model, bank):
    m2 = copy.deepcopy(model)
    b2 = copy.copy(bank)
    object.__setattr__(b2, "model", m2)
    b2.handles = {t: copy.copy(h) for t, h in bank.handles.items()}
    for _, h in b2.handles.items():
        h.bundle = copy.deepcopy(h.bundle)
    return b2.detach()


def test_ci_head_no_longer_dominates():
    """The whole reason tcn_ci exists: the channel-MIXING head was 58-99%
    of parameters, leaving TSR-X 0.8-41% of the model to reallocate."""
    for pred_len, min_body_frac in [(96, 0.90), (720, 0.60)]:
        ci = build_ts_model("tcn_ci", 862, pred_len, hidden=64)
        tot = sum(p.numel() for p in ci.parameters() if p.requires_grad)
        head = sum(p.numel() for p in ci.head.parameters())
        assert (tot - head) / tot > min_body_frac

        mixing = build_ts_model("tcn", 862, pred_len, hidden=64)
        m_tot = sum(p.numel() for p in mixing.parameters() if p.requires_grad)
        m_head = sum(p.numel() for p in mixing.head.parameters())
        assert m_head / m_tot > 0.9          # the defect this replaces
        assert tot < m_tot / 10              # and it is far smaller


def test_ci_param_count_independent_of_n_vars():
    a = sum(p.numel() for p in build_ts_model("tcn_ci", 7, 96).parameters())
    b = sum(p.numel() for p in build_ts_model("tcn_ci", 862, 96).parameters())
    assert a == b


@pytest.mark.parametrize("use_revin", [True, False])
def test_ci_candidates_dormant_at_attach(use_revin):
    m, bank, x = _attach_ci(use_revin=use_revin)
    m.eval()
    with torch.no_grad():
        full = m(x)
        stripped = _detached_copy(m, bank).eval()(x)
    assert torch.allclose(full, stripped, atol=1e-5)
    assert bank.max_port_magnitude() == 0.0


def test_ci_candidates_stay_dormant_under_training_with_revin():
    """RevIN active. If RevIN coupled candidates into the real units the way
    LayerNorm does, this is where it would show up."""
    m, bank, x = _attach_ci(use_revin=True)
    y = torch.randn(4, 24, 21)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(20):
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
        assert torch.allclose(full, stripped, atol=1e-5), "candidates leaked with RevIN active"


def test_ci_ports_drift_without_zeroing():
    """Guard the guard — the failure mode must be real on tcn_ci too."""
    m, bank, x = _attach_ci()
    y = torch.randn(4, 24, 21)
    m.train()
    assert bank.max_port_magnitude() == 0.0
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        F.mse_loss(m(x), y).backward()
        opt.step()          # deliberately NO zero_ports()
    assert bank.max_port_magnitude() > 0.0


@pytest.mark.parametrize("k", [2, 4])
def test_ci_deployed_params_exact(k):
    m, bank, x = _attach_ci(k=k)
    truth = sum(p.numel() for p in _detached_copy(m, bank).parameters() if p.requires_grad)
    assert bank.deployed_params() == truth


def test_c3_matches_target_budget():
    """A C3 that misses the budget is not a matched control."""
    ex = torch.zeros(2, 96, 21)
    build_fn = lambda: build_ts_model("tcn_ci", 21, 96)
    base = sum(p.numel() for p in build_fn().parameters() if p.requires_grad)
    target = int(base * 0.85)
    for seed in range(4):
        _, widths, achieved, rel = build_c3_model(build_fn, ex, target, seed=seed)
        assert rel <= 0.01, f"seed {seed}: off by {rel*100:.2f}%"
        assert all(w >= 8 for w in widths.values())


def test_c3_respects_min_size_and_varies_by_seed():
    ex = torch.zeros(2, 96, 21)
    build_fn = lambda: build_ts_model("tcn_ci", 21, 96)
    base = sum(p.numel() for p in build_fn().parameters() if p.requires_grad)
    a = random_widths_at_budget(build_fn, ex, int(base * 0.5), min_size=16, seed=1)
    b = random_widths_at_budget(build_fn, ex, int(base * 0.5), min_size=16, seed=2)
    assert all(w >= 16 for w in a.values())
    assert a != b, "different seeds must give different random topologies"


def test_c3_skips_output_head_group():
    """The forecast head has no consumer — resizing it would change the
    task (pred_len), not the architecture. C3 must leave it alone, the same
    way CandidateBank does."""
    m = build_ts_model("tcn_ci", 21, 96)
    taps = attachable_taps(m, torch.zeros(2, 96, 21))
    _, bank, _ = _attach_ci(n_vars=21, pred_len=96, hidden=64)
    assert set(taps) == set(bank.handles)
