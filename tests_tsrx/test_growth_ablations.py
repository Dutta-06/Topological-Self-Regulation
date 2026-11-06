"""Growth ablations and exchange edit order (rebuttal experiments).

- prune-first (released) same-group exchanges open the newborn's port in a
  random direction, because pruning replaces the consumer weight and drops
  its .grad; grow-first keeps the measured direction and otherwise produces
  the identical network.
- growth_mode "off" / "random" / "cost" change only whether / where growth
  lands, and "grad" is the released controller unchanged.
Run: python -m pytest tests_tsrx/test_growth_ablations.py -q
"""

import random

import pytest
import torch

from tsrx.alloc.exchange import ExchangeDecision, apply_exchange, evaluate_exchange
from tsrx.edit.edits import materialize_candidate

from tests_tsrx.test_exchange import _attach_and_train


def _same_group_tap(bank, min_size=8):
    """A group whose real width exceeds the floor (so it can be pruned) and
    whose consumer is a plain module we can inspect."""
    for tap, h in sorted(bank.handles.items()):
        if h.base_size > min_size and h.bundle.consumer_slots:
            return tap
    raise AssertionError("no prunable group with a consumer")


def _modules(m):
    return dict(m.named_modules())


def _same_group_exchange(tap, prune_idx=0, cand_idx=0):
    return ExchangeDecision(action="exchange", regime="optimality", reason="test",
                            grow_tap=tap, grow_cand_idx=cand_idx,
                            prune_tap=tap, prune_idx=prune_idx)


def test_prune_first_same_group_exchange_opens_random_port():
    m, bank, win, sal, n, act, opt = _attach_and_train()
    tap = _same_group_tap(bank)
    info = apply_exchange(_same_group_exchange(tap), bank, optimizer=opt, grow_first=False)
    assert info["order"] == "prune_first"
    assert set(info["port_init"].values()) == {"random_nograd"}, info


def test_grow_first_same_group_exchange_uses_measured_gradient():
    m, bank, win, sal, n, act, opt = _attach_and_train()
    tap = _same_group_tap(bank)
    h = bank.handles[tap]
    base = h.base_size
    mods = _modules(m)
    # Snapshot the candidate's measured port gradient per consumer, before the edit.
    expected = {}
    for slot in h.bundle.consumer_slots:
        if slot.module_name in expected:
            continue
        mult = slot.multiplicity
        g = mods[slot.module_name].weight.grad[:, base * mult:(base + 1) * mult].clone()
        expected[slot.module_name] = (-1e-3 * g / g.norm().clamp_min(1e-12), mult)

    info = apply_exchange(_same_group_exchange(tap, prune_idx=0, cand_idx=0), bank,
                          optimizer=opt, grow_first=True)
    assert info["order"] == "grow_first"
    assert set(info["port_init"].values()) == {"grad"}, info

    # Net width unchanged; newborn is the last real unit; its port is the
    # normalised measured gradient (rows unaffected: the pruned index lives on
    # the consumer's input axis, which this block is not part of).
    assert bank.handles[tap].base_size == base
    newborn = base - 1
    mods = _modules(m)
    for name, (v_star, mult) in expected.items():
        port = mods[name].weight.data[:, newborn * mult:(newborn + 1) * mult]
        assert torch.allclose(port, v_star, atol=1e-7), name


def test_grow_first_and_prune_first_yield_same_network_except_port():
    """Same decision, two edit orders: every tensor identical except the
    newborn's consumer port block (random vs measured)."""
    nets = {}
    port_cols = {}
    for order in ("prune_first", "grow_first"):
        torch.manual_seed(123)  # identical candidate refills in both runs
        m, bank, win, sal, n, act, opt = _attach_and_train()
        tap = _same_group_tap(bank)
        torch.manual_seed(7)
        apply_exchange(_same_group_exchange(tap, prune_idx=2, cand_idx=1), bank,
                       optimizer=opt, grow_first=(order == "grow_first"))
        bank.zero_ports()
        nets[order] = {k: v.detach().clone() for k, v in m.state_dict().items()}
        h = bank.handles[tap]
        newborn = h.base_size - 1
        port_cols[order] = {(s.module_name + ".weight"): (newborn * s.multiplicity, (newborn + 1) * s.multiplicity)
                            for s in h.bundle.consumer_slots}
        # Silence of every remaining candidate and a working forward pass.
        assert bank.max_port_magnitude() == 0.0
        assert torch.isfinite(m(torch.randn(2, 3, 32, 32))).all()

    a, b = nets["prune_first"], nets["grow_first"]
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].shape == b[k].shape, k
        if k in port_cols["grow_first"]:
            lo, hi = port_cols["grow_first"][k]
            mask = torch.ones(a[k].shape[1], dtype=torch.bool)
            mask[lo:hi] = False
            assert torch.equal(a[k][:, mask], b[k][:, mask]), k
            assert not torch.equal(a[k][:, ~mask], b[k][:, ~mask]), k  # random vs measured
        elif a[k].is_floating_point():
            assert torch.equal(a[k], b[k]), k


def _decide(mode, seed=0, budget=None):
    m, bank, win, sal, n, act, opt = _attach_and_train()
    dec = evaluate_exchange(bank=bank, windowed_signal=win, saliency_sum=sal, n_seen=n,
                            budget_params=budget, act_stats=act,
                            deployed_params=bank.deployed_params(),
                            growth_mode=mode, rng=random.Random(seed))
    return dec, bank


def test_grad_mode_is_the_released_controller():
    m, bank, win, sal, n, act, opt = _attach_and_train()
    kw = dict(bank=bank, windowed_signal=win, saliency_sum=sal, n_seen=n, budget_params=None,
              act_stats=act, deployed_params=bank.deployed_params())
    d0 = evaluate_exchange(**kw)
    d1 = evaluate_exchange(**kw, growth_mode="grad", rng=random.Random(99))
    assert (d0.action, d0.grow_tap, d0.grow_cand_idx, d0.prune_tap, d0.prune_idx) == \
           (d1.action, d1.grow_tap, d1.grow_cand_idx, d1.prune_tap, d1.prune_idx)


def test_growth_off_never_grows():
    grad_dec, _ = _decide("grad")
    assert grad_dec.action == "pure_grow"  # unlimited slack: the released rule grows
    off_dec, _ = _decide("off")
    assert off_dec.action not in ("pure_grow", "exchange")
    assert off_dec.grow_tap is None
    if off_dec.action == "none":
        assert "growth_off" in off_dec.reason


def test_cost_mode_grows_the_cheapest_group():
    from tsrx.alloc.cost import kappa_params
    dec, bank = _decide("cost")
    assert dec.action == "pure_grow"
    kappas = {t: kappa_params(h.bundle, bank.model) for t, h in bank.handles.items()}
    assert dec.details["kappa"] == min(k for k in kappas.values() if k > 0)
    assert dec.details["growth_mode"] == "cost"
    assert "grad_choice_tap" in dec.details


def test_random_mode_is_reproducible_and_varies_with_seed():
    taps = set()
    for seed in range(12):
        d1, _ = _decide("random", seed=seed)
        d2, _ = _decide("random", seed=seed)
        assert d1.action == "pure_grow" and d1.grow_tap == d2.grow_tap
        taps.add(d1.grow_tap)
    assert len(taps) >= 3, taps


def test_substituted_exchange_respects_budget():
    """At zero slack an exchange may fire; whichever group the ablation picks
    must keep the projected count within the cap."""
    for mode in ("random", "cost"):
        for seed in range(6):
            m, bank, win, sal, n, act, opt = _attach_and_train()
            deployed = bank.deployed_params()
            dec = evaluate_exchange(bank=bank, windowed_signal=win, saliency_sum=sal, n_seen=n,
                                    budget_params=deployed, act_stats=act, deployed_params=deployed,
                                    growth_mode=mode, rng=random.Random(seed), min_size_per_group=8)
            assert dec.action != "pure_grow"
            if dec.action == "exchange":
                assert deployed + dec.details["grow"]["kappa"] - dec.details["prune"]["kappa"] <= deployed


def test_unknown_growth_mode_rejected():
    m, bank, win, sal, n, act, opt = _attach_and_train()
    with pytest.raises(ValueError):
        evaluate_exchange(bank=bank, windowed_signal=win, saliency_sum=sal, n_seen=n,
                          act_stats=act, deployed_params=bank.deployed_params(), growth_mode="nope")


def test_nonfinite_gradient_falls_back_without_nan():
    m, bank, win, sal, n, act, opt = _attach_and_train()
    tap = _same_group_tap(bank)
    mods = _modules(m)
    for s in bank.handles[tap].bundle.consumer_slots:
        mods[s.module_name].weight.grad.fill_(float("inf"))
    info = materialize_candidate(bank, tap=tap, cand_idx=0, eps=1e-3, optimizer=opt)
    assert set(info.values()) == {"random_nonfinite"}
    assert all(torch.isfinite(p).all() for p in m.parameters())
