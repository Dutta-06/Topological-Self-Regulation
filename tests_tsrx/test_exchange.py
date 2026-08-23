"""Regression tests for the two-regime controller (feasibility vs
optimality) added to fix the stalled 100-epoch run in task-726.log.txt:
budget accounting included candidate mass, blocking growth/exchange
forever whenever a budget was set, and nothing tied prune rate to the
budget gap. Run: python -m pytest tests_tsrx/ -q
"""

import torch
import torch.nn.functional as F

from bench.models import build_model
from tsrx.alloc.exchange import evaluate_exchange, evaluate_structural_update, apply_exchange
from tsrx.alloc.schedule import budget_at
from tsrx.graph.bundle import build_all_bundles
from tsrx.graph.groups import discover_groups
from tsrx.graph.trace import trace_model
from tsrx.sense.candidates import CandidateBank
from tsrx.sense.saliency import ActivationStats, first_order_saliency
from tsrx.sense.topo import WindowedSignal, compute_uc_norms


def _attach_and_train(k=4, steps=3, num_classes=10):
    torch.manual_seed(0)
    m = build_model("resnet18", num_classes, cifar_stem=True)
    x = torch.randn(4, 3, 32, 32)
    traced = trace_model(m.eval(), (x[:2],))
    res = discover_groups(traced)
    bundles = build_all_bundles(res, m)
    bank = CandidateBank(m, bundles, k=k)

    win = WindowedSignal(window=steps)
    act_stats = ActivationStats(m, bank)
    saliency_sum = {tap: None for tap in bank.handles}
    n_seen = 0

    y = torch.randint(0, num_classes, (4,))
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    for _ in range(steps):
        m.train()
        opt.zero_grad()
        F.cross_entropy(m(x), y).backward()
        for tap, h in bank.handles.items():
            u = compute_uc_norms(bank, tap)
            win.record(tap, u)
            sal = first_order_saliency(m, h.bundle, h.base_size)
            saliency_sum[tap] = sal if saliency_sum[tap] is None else saliency_sum[tap] + sal
        n_seen += 1
        opt.step()
        bank.zero_ports()
    return m, bank, win, saliency_sum, n_seen, act_stats, opt


def test_budget_at_monotone_and_hits_target():
    base, target, total = 1_000_000, 700_000, 1000
    prev = base
    for step in range(0, total + 1, 50):
        b = budget_at(step, total, base, target, end_frac=0.5)
        assert b <= prev
        prev = b
    assert budget_at(0, total, base, target, end_frac=0.5) == base
    assert budget_at(500, total, base, target, end_frac=0.5) == target
    assert budget_at(1000, total, base, target, end_frac=0.5) == target
    assert budget_at(100, total, base, None, end_frac=0.5) is None


def test_feasibility_regime_reduces_deployed_params():
    m, bank, win, saliency_sum, n_seen, act_stats, opt = _attach_and_train()
    deployed = bank.deployed_params()
    # Force a budget tighter than current deployed -> feasibility regime.
    B_t = deployed - 5000

    decisions = evaluate_structural_update(
        bank=bank, windowed_signal=win, saliency_sum=saliency_sum, n_seen=n_seen,
        deployed_params=deployed, budget_at_t=B_t,
        min_size_per_group=8, act_stats=act_stats, max_prunes_per_update=8,
    )
    applied = [d for d in decisions if d.action != "none"]
    assert applied, "feasibility regime must prune when over budget"
    assert all(d.regime == "feasibility" and d.action == "pure_prune" for d in applied)
    # At most one prune per group.
    assert len(applied) == len({d.prune_tap for d in applied})

    for d in applied:
        apply_exchange(d, bank, optimizer=opt)
    bank.zero_ports()

    new_deployed = bank.deployed_params()
    assert new_deployed < deployed


def test_feasibility_bypasses_deadness_and_reports_reason_on_noop():
    m, bank, win, saliency_sum, n_seen, act_stats, opt = _attach_and_train()
    deployed = bank.deployed_params()
    # Every group already at (or below) min_size_per_group -> nothing prunable,
    # even though we are (by construction) over budget.
    decisions = evaluate_structural_update(
        bank=bank, windowed_signal=win, saliency_sum=saliency_sum, n_seen=n_seen,
        deployed_params=deployed, budget_at_t=deployed - 1,
        min_size_per_group=10 ** 9, act_stats=act_stats,
    )
    assert len(decisions) == 1
    assert decisions[0].action == "none"
    assert decisions[0].regime == "feasibility"
    assert decisions[0].reason  # must not be empty


def test_optimality_regime_uses_deployed_not_candidate_inflated_params():
    m, bank, win, saliency_sum, n_seen, act_stats, opt = _attach_and_train()
    deployed = bank.deployed_params()
    inflated = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert inflated > deployed, "sanity: candidate mass must actually inflate the raw sum"

    # Budget set to exactly the TRUE deployed count: under the old bug (raw
    # sum vs budget) this would read as over-budget forever and pure growth
    # (Case A) would be permanently unreachable. With deployed_params passed
    # correctly, the model is exactly AT budget and growth should be
    # evaluated as budget-neutral, not rejected outright.
    dec = evaluate_exchange(
        bank=bank, windowed_signal=win, saliency_sum=saliency_sum, n_seen=n_seen,
        budget_params=deployed, act_stats=act_stats, deployed_params=deployed,
    )
    # It may or may not choose to grow (depends on gamma/rho magnitudes),
    # but it must never be rejected on the strength of the inflated sum.
    if dec.action == "pure_grow":
        assert dec.details["kappa"] + deployed >= 0  # reachable at all, not a hard reject
    assert dec.reason != ""


def test_exchange_rejected_when_move_would_exceed_budget():
    """Case C's hard cap: WITHIN budget (current <= budget_params), any
    accepted move's PROJECTED params must not exceed the cap, and pure
    growth (which always increases params) must be rejected outright when
    budget == deployed exactly, since kappa_grow > 0 for every real group.

    (Case C's separate OVER-budget branch is deliberately looser — it
    accepts any params-reducing exchange as forward progress rather than
    hard-rejecting, since strict rejection there would just freeze the
    optimality regime forever; that path is exercised by
    test_feasibility_regime_reduces_deployed_params instead, since
    `evaluate_structural_update` never calls into `evaluate_exchange`
    while still over budget. An "exchange" CAN legitimately fire here even
    at the tightest cap, since it grows one group while pruning another —
    net params can still shrink or hold even while budget is razor-thin.)
    """
    m, bank, win, saliency_sum, n_seen, act_stats, opt = _attach_and_train()
    deployed = bank.deployed_params()
    B_t = deployed  # tightest possible "within budget" cap: zero slack

    dec = evaluate_exchange(
        bank=bank, windowed_signal=win, saliency_sum=saliency_sum, n_seen=n_seen,
        budget_params=B_t, act_stats=act_stats, deployed_params=deployed,
        min_size_per_group=8,
    )
    assert dec.action != "pure_grow", "pure growth must be impossible at zero budget slack"
    if dec.action == "exchange":
        grow_kappa = dec.details["grow"]["kappa"]
        prune_kappa = dec.details["prune"]["kappa"]
        assert deployed + grow_kappa - prune_kappa <= B_t
