"""Stage 2: transformer (PatchTST) support.

The mathematical claim being tested, from the traced graph:

  * Exactly ONE coupling group touches a LayerNorm -- the d_model residual
    stream -- and it is the group we freeze anyway (largest kappa, spans
    every block, confounds all comparisons). Condition (N) is therefore not
    a blocker for FFN reallocation: the d_ff axis carries no normalization
    at all.
  * The attention q/k/v groups are unsafe for a reason invisible to the
    coupling engine: `view(B, L, h, d_head)` requires the width to stay
    divisible by the head count, and the engine models module->module edges
    with no representation of QK^T or AV. These MUST be excluded.

So `safe_taps` must return exactly the FFN groups, and dormancy must hold
bit-identically on them despite LayerNorm being present elsewhere.
"""

import copy

import pytest
import torch
import torch.nn.functional as F

from bench.patchtst import build_patchtst, ffn_taps
from tsrx.graph.bundle import build_all_bundles
from tsrx.graph.groups import discover_groups
from tsrx.graph.safety import probe_reallocatable, safe_taps, violates_norm_condition
from tsrx.graph.trace import trace_model
from tsrx.sense.candidates import CandidateBank
from tsrx.sense.saliency import ActivationStats


def _build(n_vars=7, d_model=64, d_ff=128, n_heads=4, n_blocks=3):
    return build_patchtst(n_vars, 96, 96, d_model=d_model, d_ff=d_ff,
                           n_heads=n_heads, n_blocks=n_blocks)


def _attach(k=4):
    torch.manual_seed(0)
    m = _build()
    x = torch.randn(4, 96, 7)
    bd = build_all_bundles(discover_groups(trace_model(m.eval(), (x[:2],))), m)
    taps = ffn_taps(m, bd)
    return m, CandidateBank(m, bd, k=k, only_taps=taps, skip_unsupported=True), x, taps


def _detached(model, bank):
    m2 = copy.deepcopy(model)
    b2 = copy.copy(bank)
    object.__setattr__(b2, "model", m2)
    b2.handles = {t: copy.copy(h) for t, h in bank.handles.items()}
    for _, h in b2.handles.items():
        h.bundle = copy.deepcopy(h.bundle)
    return b2.detach()


def test_patchtst_forward_shape():
    m = _build()
    assert m(torch.randn(2, 96, 7)).shape == (2, 96, 7)


def test_only_the_d_model_group_touches_layernorm():
    """The premise of Stage 2: condition (N) blocks exactly one group, and
    it is the one we freeze regardless."""
    m = _build()
    x = torch.zeros(2, 96, 7)
    bd = build_all_bundles(discover_groups(trace_model(m.eval(), (x,))), m)
    offenders = [t for t, b in bd.items() if violates_norm_condition(b, m)]
    assert len(offenders) == 1
    # ...and it is the residual stream: many producers spanning every block
    producers = {s.module_name for s in bd[offenders[0]].producer_slots}
    assert len(producers) > 3
    # the FFN groups are clean
    for t in ffn_taps(m, bd):
        assert not violates_norm_condition(bd[t], m)


def test_safety_probe_selects_exactly_the_ffn_groups():
    """Empirical probe (prune + forward) must agree with the name-based
    allowlist -- and must reject attention, which looks resizable to the
    coupling engine but crashes on the head reshape."""
    bf = _build
    x = torch.zeros(2, 96, 7)
    m = bf()
    bd = build_all_bundles(discover_groups(trace_model(m.eval(), (x,))), m)
    assert safe_taps(bf, x) == ffn_taps(m, bd)


def test_attention_groups_rejected_with_reason():
    info = probe_reallocatable(_build, torch.zeros(2, 96, 7))
    q_taps = [t for t, i in info.items()
              if any(n.endswith(".q") for n in i["producers"])]
    assert q_taps, "expected attention q groups in the graph"
    for t in q_taps:
        assert not info[t]["safe"]
        assert "(R)" in info[t]["reason"]


def test_candidates_dormant_under_training_with_layernorm_present():
    """THE invariant. LayerNorm exists in this model (on d_model); it must
    not couple the d_ff candidates into the real units."""
    m, bank, x, taps = _attach()
    assert sorted(bank.handles) == taps
    assert bank.max_port_magnitude() == 0.0
    y = torch.randn(4, 96, 7)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(15):
        m.train()
        opt.zero_grad()
        F.mse_loss(m(x), y).backward()
        opt.step()
        bank.zero_ports()
        assert bank.max_port_magnitude() == 0.0
        m.eval()
        with torch.no_grad():
            assert torch.allclose(m(x), _detached(m, bank).eval()(x), atol=1e-5)


def test_deployed_params_exact_on_transformer():
    m, bank, x, _ = _attach()
    truth = sum(p.numel() for p in _detached(m, bank).parameters() if p.requires_grad)
    assert bank.deployed_params() == truth


def test_activation_stats_uses_last_axis_for_linear():
    """(B, L, D): channels are axis 2, not axis 1. Reducing the wrong axis
    yields per-timestep statistics for the wrong index set entirely."""
    m, bank, x, taps = _attach()
    st = ActivationStats(m, bank)
    m.train()
    m(x)
    h = bank.handles[taps[0]]
    ms = st.mean_sq(taps[0])
    assert ms is not None and ms.numel() == h.base_size + h.k


def test_activation_stats_not_corrupted_by_model_copy():
    """deepcopy carries forward hooks, but function objects are not
    deep-copied -- so a copy's forward would write into THIS instance and
    silently replace E[a^2] with statistics from a stripped model."""
    m, bank, x, taps = _attach()
    st = ActivationStats(m, bank)
    m.train()
    m(x)
    before = st.mean_sq(taps[0]).shape
    stripped = _detached(m, bank)
    stripped.eval()
    with torch.no_grad():
        stripped(x)
    assert st.mean_sq(taps[0]).shape == before


def test_skip_unsupported_off_by_default_still_raises():
    """Existing callers must keep the loud failure."""
    from tsrx.sense.candidates import UnsupportedGroupError
    m = _build()
    x = torch.zeros(2, 96, 7)
    bd = build_all_bundles(discover_groups(trace_model(m.eval(), (x,))), m)
    with pytest.raises(UnsupportedGroupError):
        CandidateBank(m, bd, k=2)
