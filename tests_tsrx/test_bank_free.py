"""Bank-free (k = 0) removal-only TSR.

The growth ablation found that growth does not produce the allocation when
the budget shrinks, so the dormant candidate bank, which exists only to sense
growth and costs most of TSR's training overhead, can be dropped there. With
k = 0 the bank still tracks every coupling group (removal needs the bundles),
but no tensor carries a dormant extent.
Run: python -m pytest tests_tsrx/test_bank_free.py -q
"""

import random

import torch

from bench.models import build_model
from tsrx.alloc.exchange import apply_exchange, evaluate_structural_update
from tsrx.graph.bundle import build_all_bundles
from tsrx.graph.groups import discover_groups
from tsrx.graph.trace import trace_model
from tsrx.sense.candidates import CandidateBank
from tsrx.sense.topo import compute_uc_norms

from tests_tsrx.test_exchange import _attach_and_train


def _count(m):
    return sum(p.numel() for p in m.parameters())


def test_bank_free_attach_changes_nothing():
    torch.manual_seed(0)
    m = build_model("resnet18", 10, cifar_stem=True).eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        before = m(x).clone()
    n = _count(m)
    bank = CandidateBank(m, build_all_bundles(discover_groups(trace_model(m, (x,))), m), k=0)
    with torch.no_grad():
        after = m.eval()(x)
    assert torch.equal(before, after)
    assert len(bank.handles) > 0 and all(h.k == 0 for h in bank.handles.values())
    assert _count(m) == n == bank.deployed_params()
    assert bank.max_port_magnitude() == 0.0
    for tap in bank.handles:
        assert compute_uc_norms(bank, tap).numel() == 0


def test_bank_free_removal_only_controller_prunes_to_budget():
    m, bank, win, sal, n, act, opt = _attach_and_train(k=0)
    deployed = bank.deployed_params()
    assert deployed == _count(m)
    decisions = evaluate_structural_update(
        bank=bank, windowed_signal=win, saliency_sum=sal, n_seen=n,
        deployed_params=deployed, budget_at_t=deployed - 20_000, act_stats=act,
        growth_mode="off", rng=random.Random(0))
    applied = [d for d in decisions if d.action != "none"]
    assert applied and all(d.action == "pure_prune" for d in applied)
    for d in applied:
        apply_exchange(d, bank, optimizer=opt)
    after = bank.deployed_params()
    assert after < deployed
    assert after == _count(m)          # no dormant extents anywhere, before or after the edit
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    m.train()
    opt.zero_grad()
    torch.nn.functional.cross_entropy(m(x), y).backward()
    opt.step()                          # optimizer state was re-indexed consistently
    assert torch.isfinite(m(x)).all()
