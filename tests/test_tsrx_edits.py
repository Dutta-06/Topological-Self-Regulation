"""Unit tests for TSR-X structural edits on ResNet-18 (Theorem 5.5 & Lemma 2.2)."""
import torch
import torchvision

from tsrx.graph.trace import trace_model
from tsrx.graph.groups import discover_groups
from tsrx.graph.bundle import build_all_bundles
from tsrx.sense.candidates import CandidateBank
from tsrx.edit.edits import prune_group_index, materialize_candidate


def test_resnet18_candidate_attachment_and_prune():
    """Verify CandidateBank attaches to ResNet-18 and preserves output."""
    torch.manual_seed(42)
    model = torchvision.models.resnet18(num_classes=10)
    model.eval()

    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out_orig = model(x)

    traced = trace_model(model, (x[:2],))
    res = discover_groups(traced)
    bundles = build_all_bundles(res, model)

    bank = CandidateBank(model, bundles, k=4)

    # Output with zeroed port candidates should be identical (Lemma 2.2)
    with torch.no_grad():
        out_cand = model(x)
    assert torch.allclose(out_orig, out_cand, atol=1e-5), "Candidate attachment must be function-preserving!"

    # Prune one channel from a free group (e.g. tap 1: layer1.0.conv1)
    tap1 = 1
    bd1 = bank.handles[tap1].bundle
    orig_size = bd1.size
    
    # Zero out channel 0 weights before pruning to test exact function preservation
    # (Corollary 4.4: safe removal via decay)
    modules = dict(model.named_modules())
    for s in bd1.consumer_slots:
        modules[s.module_name].weight.data[:, 0] = 0.0

    with torch.no_grad():
        out_before_prune = model(x)

    prune_group_index(model, bd1, idx=0, bank=bank)
    assert bd1.size == orig_size - 1
    assert bank.handles[tap1].base_size == orig_size - 1

    with torch.no_grad():
        out_after_prune = model(x)

    assert torch.allclose(out_before_prune, out_after_prune, atol=1e-5), "Pruning a zeroed channel must be function-preserving!"


def test_resnet18_materialize_and_forward_backward():
    """Verify candidate materialization and gradient flow on ResNet-18."""
    torch.manual_seed(42)
    model = torchvision.models.resnet18(num_classes=10)

    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])

    traced = trace_model(model, (x[:2],))
    res = discover_groups(traced)
    bundles = build_all_bundles(res, model)
    bank = CandidateBank(model, bundles, k=4)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Step 1: Forward + Backward
    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()

    # Step 2: Materialize candidate 0 at tap 1 (layer1.0.conv1)
    tap1 = 1
    bd1 = bank.handles[tap1].bundle
    old_size = bd1.size
    materialize_candidate(bank, tap=tap1, cand_idx=0, eps=1e-3, optimizer=optimizer)

    assert bd1.size == old_size + 1
    assert bank.handles[tap1].base_size == old_size + 1

    # Step 3: Run another forward-backward to confirm no tensor shape mismatch
    optimizer.zero_grad()
    out2 = model(x)
    loss2 = torch.nn.functional.cross_entropy(out2, y)
    loss2.backward()
    optimizer.step()

    assert loss2.item() > 0.0
