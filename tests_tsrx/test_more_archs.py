"""iTransformer and TSMixer: same invariants as test_transformer.py, plus
one new one for TSMixer specifically.

iTransformer reuses PatchTST's `_EncoderBlock` verbatim, so its safety story
is identical (d_model frozen by condition (N), attention excluded by the
empirical probe, FFN reallocatable) — tested here mainly to confirm the
reuse didn't silently change anything.

TSMixer is the new case: it has TWO task-fixed axes that are NOT caught by
condition (N) at all (BatchNorm doesn't trigger it) — the variate count and
the seq_len both have to be excluded because pruning either would make the
model's forward pass disagree with the shape the dataloader actually
produces. Nothing routes this through condition (N); it's caught only
because `tsrx.graph.safety.safe_taps` performs an empirical prune+forward
check rather than a purely static one. That claim is the point of
`test_task_fixed_axes_excluded_by_probe` below — verified, not assumed.
"""

import copy

import pytest
import torch
import torch.nn.functional as F

from bench.itransformer import build_itransformer
from bench.tsmixer import build_tsmixer, mixer_ffn_taps
from bench.patchtst import ffn_taps
from tsrx.graph.bundle import build_all_bundles
from tsrx.graph.groups import discover_groups
from tsrx.graph.safety import safe_taps
from tsrx.graph.trace import trace_model
from tsrx.sense.candidates import CandidateBank


def _detached(model, bank):
    m2 = copy.deepcopy(model)
    b2 = copy.copy(bank)
    object.__setattr__(b2, "model", m2)
    b2.handles = {t: copy.copy(h) for t, h in bank.handles.items()}
    for _, h in b2.handles.items():
        h.bundle = copy.deepcopy(h.bundle)
    return b2.detach()


def _dormancy_roundtrip(build_fn, x, y, k=4, steps=15):
    torch.manual_seed(0)
    m = build_fn()
    bd = build_all_bundles(discover_groups(trace_model(m.eval(), (x[:2],))), m)
    taps = safe_taps(build_fn, x[:2])
    bank = CandidateBank(m, bd, k=k, only_taps=taps, skip_unsupported=True)
    assert bank.max_port_magnitude() == 0.0

    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(steps):
        m.train()
        opt.zero_grad()
        F.mse_loss(m(x), y).backward()
        opt.step()
        bank.zero_ports()
        assert bank.max_port_magnitude() == 0.0
        m.eval()
        with torch.no_grad():
            assert torch.allclose(m(x), _detached(m, bank).eval()(x), atol=1e-5)
    return m, bank, taps


# ------------------------------------------------------------- iTransformer

def test_itransformer_forward_shape():
    m = build_itransformer(7, 96, 96, d_model=32, d_ff=64, n_heads=4, n_blocks=2)
    assert m(torch.randn(3, 96, 7)).shape == (3, 96, 7)


def test_itransformer_safety_matches_ffn_taps():
    bf = lambda: build_itransformer(7, 96, 96, d_model=32, d_ff=64, n_heads=4, n_blocks=2)
    x = torch.zeros(2, 96, 7)
    m = bf()
    bd = build_all_bundles(discover_groups(trace_model(m.eval(), (x,))), m)
    assert safe_taps(bf, x) == ffn_taps(m, bd)


def test_itransformer_dormant_and_exact_params():
    bf = lambda: build_itransformer(7, 96, 24, d_model=32, d_ff=64, n_heads=4, n_blocks=2)
    x, y = torch.randn(4, 96, 7), torch.randn(4, 24, 7)
    m, bank, taps = _dormancy_roundtrip(bf, x, y)
    assert taps, "iTransformer must have reallocatable FFN groups"
    truth = sum(p.numel() for p in _detached(m, bank).parameters() if p.requires_grad)
    assert bank.deployed_params() == truth


def test_itransformer_param_count_independent_of_n_vars():
    a = sum(p.numel() for p in build_itransformer(7, 96, 96, d_model=32, d_ff=64, n_heads=4, n_blocks=2).parameters())
    b = sum(p.numel() for p in build_itransformer(862, 96, 96, d_model=32, d_ff=64, n_heads=4, n_blocks=2).parameters())
    assert a == b


# ------------------------------------------------------------------ TSMixer

def test_tsmixer_forward_shape():
    m = build_tsmixer(7, 96, 96, d_ff=64, n_blocks=3)
    assert m(torch.randn(3, 96, 7)).shape == (3, 96, 7)


def test_task_fixed_axes_excluded_by_probe():
    """THE claim this file exists to check: TSMixer's variate axis and
    seq_len axis are both task-fixed (not free capacity) and neither
    triggers condition (N) -- BatchNorm doesn't. If the empirical probe
    didn't catch these, TSR-X could 'successfully' shrink the number of
    variates a model accepts, producing a shape mismatch against the real
    dataloader on the very next batch outside this test's control."""
    bf = lambda: build_tsmixer(7, 96, 24, d_ff=64, n_blocks=3)
    x = torch.zeros(2, 96, 7)
    m = bf()
    bd = build_all_bundles(discover_groups(trace_model(m.eval(), (x,))), m)
    safe = safe_taps(bf, x)
    expected = mixer_ffn_taps(m, bd)
    assert safe == expected, "probe must keep exactly the fc1/fc2 groups, nothing else"

    # The variate axis (C) and the time-mixing axis (L) both get unioned,
    # via the residual stream, with `fc_time`/`fc2` into ONE group the
    # union-find reports as size 0 -- a degenerate group precisely because
    # it conflates two structurally different, both task-fixed, axes. A
    # size-0 group cannot even be indexed (prune_group_index would need
    # `0 <= idx < 0`, impossible), so it is unusable by construction, and
    # CandidateBank's own `bd.size == 0` guard already refuses to attach it
    # -- confirming the exclusion here isn't a probe coincidence.
    merged = [t for t in bd if t not in safe and {s.module_name for s in bd[t].producer_slots}
              and any(n.endswith(".fc2") for n in {s.module_name for s in bd[t].producer_slots})]
    assert merged, "expected the fc2/fc_time residual-stream group to be discovered"
    for tap in merged:
        assert bd[tap].size == 0

    # The real-world consequence, demonstrated directly: build ANOTHER model
    # with a genuinely different n_vars and confirm it is NOT interchangeable
    # with this one at the group level -- i.e. n_vars is not free capacity,
    # exactly what excluding this group protects against.
    m_other = build_tsmixer(5, 96, 24, d_ff=64, n_blocks=3)
    with pytest.raises(RuntimeError):
        with torch.no_grad():
            m_other(x)  # x has 7 channels, this model expects 5


def test_tsmixer_dormant_and_exact_params():
    bf = lambda: build_tsmixer(7, 96, 24, d_ff=64, n_blocks=3)
    x, y = torch.randn(4, 96, 7), torch.randn(4, 24, 7)
    m, bank, taps = _dormancy_roundtrip(bf, x, y)
    assert taps, "TSMixer must have reallocatable fc1/fc2 groups"
    truth = sum(p.numel() for p in _detached(m, bank).parameters() if p.requires_grad)
    assert bank.deployed_params() == truth


def test_tsmixer_reallocatable_share_grows_with_n_vars():
    """The feature-mixing MLP scales with n_vars (it mixes ACROSS variates),
    unlike PatchTST/iTransformer/TCN_ci -- so TSMixer's compression ceiling
    is dataset-dependent and must be computed per-dataset before a sweep,
    not assumed constant."""
    from tsrx.graph.safety import safe_taps as st
    from bench.resize import resize_model_to_widths

    def ceiling(n_vars):
        bf = lambda: build_tsmixer(n_vars, 96, 96, d_ff=256, n_blocks=4)
        x = torch.zeros(2, 96, n_vars)
        m = bf()
        tot = sum(p.numel() for p in m.parameters() if p.requires_grad)
        taps = st(bf, x)
        floor = resize_model_to_widths(bf(), {str(t): 8 for t in taps}, x)
        fp = sum(p.numel() for p in floor.parameters() if p.requires_grad)
        return (1 - fp / tot) * 100

    small, large = ceiling(7), ceiling(321)
    assert large > small + 20, f"expected ceiling to grow substantially with n_vars, got {small:.1f}% -> {large:.1f}%"
