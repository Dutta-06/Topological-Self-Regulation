"""The per-step sensing fast path must not change a single number.

Two changes are covered:
  * compute_uc_norms / first_order_saliency with a cached module map
    (instead of rebuilding `dict(model.named_modules())` every call);
  * WindowedSignal keeping records on their device and copying to the host
    only when the mean is read (instead of a .cpu() per record).
Both must be bit-identical to the original path, so decisions cannot move.
"""

from collections import deque

import pytest
import torch
import torch.nn.functional as F

from bench.models import build_model
from tsrx.graph.bundle import build_all_bundles
from tsrx.graph.groups import discover_groups
from tsrx.graph.trace import trace_model
from tsrx.sense.candidates import CandidateBank
from tsrx.sense.saliency import first_order_saliency
from tsrx.sense.topo import WindowedSignal, compute_uc_norms

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _bank_with_grads(device):
    torch.manual_seed(0)
    m = build_model("resnet18", 10, cifar_stem=True).to(device)
    x = torch.randn(4, 3, 32, 32, device=device)
    bank = CandidateBank(m, build_all_bundles(discover_groups(trace_model(m.eval(), (x[:2],))), m), k=4).to(device)
    m.train()
    # give every candidate port a nonzero value so u_c is nontrivial downstream
    F.cross_entropy(m(x), torch.randint(0, 10, (4,), device=device)).backward()
    return m, bank


@pytest.mark.parametrize("device", DEVICES)
def test_cached_module_map_is_bit_identical(device):
    m, bank = _bank_with_grads(device)
    mods = dict(m.named_modules())
    for tap, h in bank.handles.items():
        assert torch.equal(compute_uc_norms(bank, tap), compute_uc_norms(bank, tap, modules=mods))
        assert torch.equal(first_order_saliency(m, h.bundle, h.base_size),
                           first_order_saliency(m, h.bundle, h.base_size, modules=mods))


@pytest.mark.parametrize("device", DEVICES)
def test_window_mean_matches_host_copy_per_record(device):
    torch.manual_seed(1)
    records = [torch.rand(4, device=device) * 10 ** torch.randint(-3, 3, ()).item() for _ in range(250)]
    win = WindowedSignal(window=100)
    old = deque(maxlen=100)  # the original implementation: host copy on arrival
    for r in records:
        win.record(7, r)
        old.append(r.detach().cpu())
        assert torch.equal(win.mean(7), torch.stack(list(old)).mean(dim=0))
    assert win.mean(7).device.type == "cpu"
    assert win.is_ready(7) and win.best(7) == float(torch.stack(list(old)).mean(dim=0).max())
