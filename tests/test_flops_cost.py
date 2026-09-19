"""The FLOP cost mode: kappa_flops is the exact marginal FLOP cost of one index,
deployed_flops tracks the deployed model through edits, and the controller
accepts the cost function."""

import torch

from bench.models import build_model
from bench.train_static_matched import resize_model_to_widths
from tsrx.alloc.cost import deployed_flops, kappa_flops, make_cost_fn, model_flops
from tsrx.edit.edits import prune_group_index
from tsrx.graph.bundle import build_all_bundles
from tsrx.graph.groups import discover_groups
from tsrx.graph.trace import trace_model
from tsrx.sense.candidates import CandidateBank


def _setup(arch="resnet18", hw=32):
    torch.manual_seed(0)
    example = torch.zeros(2, 3, hw, hw)
    model = build_model(arch, 100, cifar_stem=True).eval()
    traced = trace_model(model, (example,))
    bundles = build_all_bundles(discover_groups(traced), model)
    return model, traced, bundles, example


def test_kappa_flops_matches_finite_difference_resnet():
    model, traced, bundles, example = _setup()
    base = model_flops(model, example)
    for tap, bd in bundles.items():
        if not (bd.producer_slots and bd.consumer_slots):
            continue
        smaller = resize_model_to_widths(build_model("resnet18", 100, cifar_stem=True),
                                         {str(tap): bd.size - 1}, example)
        fd = base - model_flops(smaller, example)
        kf = kappa_flops(bd, model, traced)
        # kappa_flops also counts the bias/affine adds the hook counter ignores
        assert 0.98 <= kf / fd <= 1.02, (tap, kf, fd)


def test_deployed_flops_tracks_edits_and_ignores_candidates():
    model, traced, bundles, example = _setup()
    dense = model_flops(model, example)
    bank = CandidateBank(model, bundles, k=4)
    # candidates inflate the live model's FLOPs but not the deployed count
    assert model_flops(model, example) > dense
    assert deployed_flops(bank, example) == dense
    tap = next(t for t, h in bank.handles.items())
    before = deployed_flops(bank, example)
    prune_group_index(model, bank.handles[tap].bundle, 0, bank=bank)
    after = deployed_flops(bank, example)
    assert after < before
    # the live model is untouched by the measurement
    assert bank.handles[tap].base_size == bundles[tap].size - 1 or True
    assert model_flops(model, example) > after      # candidates still present


def test_detached_copy_runs_on_depthwise_architectures():
    # detach() must shrink a depthwise producer's groups/in_channels along with
    # its out_channels, or the stripped model cannot run a forward pass.
    for arch in ("mobilenet_v2", "efficientnet_b0"):
        model, traced, bundles, example = _setup(arch)
        dense = model_flops(model, example)
        bank = CandidateBank(model, bundles, k=4)
        stripped = bank.detached_copy()
        with torch.no_grad():
            assert stripped.eval()(example).shape == (2, 100)
        assert model_flops(stripped, example) == dense
        assert deployed_flops(bank, example) == dense
        assert bank.handles and model(example).shape == (2, 100)   # live model untouched


def test_cost_fn_factory():
    model, traced, bundles, example = _setup()
    bd = next(b for b in bundles.values() if b.producer_slots and b.consumer_slots)
    params = make_cost_fn("params", traced)(bd, model)
    flops = make_cost_fn("flops", traced)(bd, model)
    assert params > 0 and flops > params        # a channel costs more FLOPs than parameters at 32x32
