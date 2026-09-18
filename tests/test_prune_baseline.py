"""The pruning baseline edits the same plastic set as TSR-X, lands on its target
count exactly enough, and leaves every shape legal."""

import torch

from bench.models import build_model
from bench.prune_baseline import (
    deployed_params, editable_bundles, importance_bnscale, importance_l1,
    make_importance_taylor, prune_to_target,
)
from tsrx.graph.bundle import build_all_bundles
from tsrx.graph.groups import discover_groups
from tsrx.graph.trace import trace_model


def _resnet_cifar():
    torch.manual_seed(0)
    return build_model("resnet18", 100, cifar_stem=True), torch.zeros(2, 3, 32, 32)


def test_editable_set_matches_tsrx_eligibility():
    model, example = _resnet_cifar()
    bundles = editable_bundles(model, example)
    traced = trace_model(model.eval(), (example,))
    all_bundles = build_all_bundles(discover_groups(traced), model)
    expected = {t for t, b in all_bundles.items() if b.size and b.producer_slots and b.consumer_slots}
    assert set(bundles) == expected
    assert len(bundles) == 12                      # ResNet-18: 12 editable groups, fc excluded


def test_prune_reaches_target_and_forward_runs():
    model, example = _resnet_cifar()
    bundles = editable_bundles(model, example)
    start = deployed_params(model)
    target = int(0.85 * start)
    events = prune_to_target(model, bundles, target, importance_l1, min_width=8)
    after = deployed_params(model)
    assert after <= target
    assert target - after < 0.01 * target          # never far under the target
    assert len(events) > 0
    assert all(b.size >= 8 for b in bundles.values())
    with torch.no_grad():
        out = model.eval()(example)
    assert out.shape == (2, 100)
    # the classifier's task axis and the input are untouched
    assert model.fc.out_features == 100 and model.conv1.in_channels == 3


def test_importances_have_group_shapes():
    model, example = _resnet_cifar()
    bundles = editable_bundles(model, example)
    for fn in (importance_l1, importance_bnscale):
        scores = fn(model, bundles)
        assert all(scores[t].shape == (b.size,) for t, b in bundles.items())
        assert all(torch.isfinite(scores[t]).all() for t in bundles)


def test_taylor_importance_reads_saliency():
    model, example = _resnet_cifar()
    bundles = editable_bundles(model, example)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 100, (8,))
    loader = [(x, y)] * 2
    fn = make_importance_taylor(loader, "cpu", None, n_batches=2)
    scores = fn(model, bundles)
    assert all(scores[t].shape == (b.size,) for t, b in bundles.items())
    assert any(scores[t].sum() > 0 for t in bundles)
