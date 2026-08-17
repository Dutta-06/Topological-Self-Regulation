"""Unit tests for TSR-X Exchange Operator (Definition 4.5 & Theorem 4.6)."""
import torch
import torchvision

from tsrx.graph.trace import trace_model
from tsrx.graph.groups import discover_groups
from tsrx.graph.bundle import build_all_bundles
from tsrx.sense.candidates import CandidateBank
from tsrx.sense.topo import WindowedSignal, compute_uc_norms
from tsrx.sense.saliency import first_order_saliency
from tsrx.alloc.exchange import evaluate_exchange, apply_exchange


def test_exchange_operator_step():
    """Verify evaluating and executing an exchange step on ResNet-18."""
    torch.manual_seed(42)
    model = torchvision.models.resnet18(num_classes=10)

    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])

    traced = trace_model(model, (x[:2],))
    res = discover_groups(traced)
    bundles = build_all_bundles(res, model)
    bank = CandidateBank(model, bundles, k=4)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    win = WindowedSignal(window=5)
    saliency_sum = {tap: None for tap in bank.handles}

    # Simulate 5 steps of training to populate signals
    for _ in range(5):
        optimizer.zero_grad()
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()

        for tap, h in bank.handles.items():
            u = compute_uc_norms(bank, tap)
            win.record(tap, u)
            sal = first_order_saliency(model, h.bundle, h.base_size)
            saliency_sum[tap] = sal if saliency_sum[tap] is None else saliency_sum[tap] + sal

    # Evaluate exchange
    orig_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dec = evaluate_exchange(
        bank, win, saliency_sum, n_seen=5, delta=1e-9, prune_tolerance=1e-9, min_size_per_group=4
    )

    # If an exchange or pure operation is proposed, apply it and check network validity
    if dec.action != "none":
        apply_exchange(dec, bank, optimizer=optimizer)
        new_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Check forward/backward still works
        optimizer.zero_grad()
        out2 = model(x)
        loss2 = torch.nn.functional.cross_entropy(out2, y)
        loss2.backward()
        optimizer.step()

        assert loss2.item() > 0.0
