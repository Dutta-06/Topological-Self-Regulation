"""The DepGraph arm prunes exactly TSR-X's plastic set: the library's groups
coincide with the coupling groups on every backbone, the target is reached
within one channel's cost, the width floor holds and every shape stays legal."""

import pytest
import torch

from bench.models import build_model
from tsrx.edit.pruning import deployed_params, editable_bundles

tp = pytest.importorskip("torch_pruning")

from tsrx.edit.depgraph import (  # noqa: E402
    depgraph_group_report, ignored_layers, prune_depgraph_to_target,
)

ARCHS = ["resnet18", "vgg16_bn", "mobilenet_v2", "efficientnet_b0"]
GROUPS = {"resnet18": 12, "vgg16_bn": 13, "mobilenet_v2": 25, "efficientnet_b0": 40}


def _cifar(arch):
    torch.manual_seed(0)
    return build_model(arch, 10, cifar_stem=True), torch.zeros(2, 3, 32, 32)


@pytest.mark.parametrize("arch", ARCHS)
def test_depgraph_groups_coincide_with_coupling_groups(arch):
    model, example = _cifar(arch)
    bundles = editable_bundles(model, example)
    assert len(bundles) == GROUPS[arch]
    report = depgraph_group_report(model, example, bundles)
    assert report["matched"] == len(bundles)
    assert report["mismatched"] == [] and report["missing"] == []
    # the classifier's output axis is outside the plastic set on every backbone
    last_linear = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
    assert last_linear in ignored_layers(model, bundles)


@pytest.mark.parametrize("arch", ["resnet18", "mobilenet_v2"])
def test_depgraph_reaches_target_on_the_lattice(arch):
    model, example = _cifar(arch)
    bundles = editable_bundles(model, example)
    before = {t: b.size for t, b in bundles.items()}
    start = deployed_params(model)
    target = int(0.85 * start)
    events, pruned = prune_depgraph_to_target(model, example, bundles, target, min_width=8)
    after = deployed_params(model)
    assert after <= target
    assert target - after < 0.001 * target                # trimmed to within one channel's cost
    assert events and all(e["removed"] > 0 for e in events)
    assert set(pruned) == set(bundles)                    # re-traces to the same coupling groups
    assert all(b.size >= 8 or b.size == before[t] for t, b in pruned.items())   # floor, or untouched
    assert sum(before[t] - b.size for t, b in pruned.items()) == sum(e["removed"] for e in events)
    with torch.no_grad():
        out = model.eval()(example)
    assert out.shape == (2, 10)
    last_linear = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
    assert last_linear.out_features == 10


def test_depgraph_widths_rebuild_as_a_fresh_architecture():
    """scratch mode: the pruned width vector is a legal TSR-X width vector."""
    from bench.train_static_matched import resize_model_to_widths
    model, example = _cifar("resnet18")
    bundles = editable_bundles(model, example)
    target = int(0.9 * deployed_params(model))
    _, pruned = prune_depgraph_to_target(model, example, bundles, target)
    widths = {str(t): b.size for t, b in pruned.items()}
    fresh = resize_model_to_widths(build_model("resnet18", 10, cifar_stem=True), widths, example)
    assert deployed_params(fresh) == deployed_params(model)
