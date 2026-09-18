"""Forecasting pruning baseline: same plastic set as the harness's safety probe,
target reached, shapes legal, on the tsrx-time model zoo. Skipped where that
branch's modules are absent."""

import pytest
import torch
import torch.nn.functional as F

from tsrx.edit.pruning import (
    deployed_params, editable_bundles, importance_l1, make_importance_taylor, prune_to_target,
)

ts_models = pytest.importorskip("bench.ts_models")
c3_random = pytest.importorskip("bench.c3_random")


def _cell(arch, n_vars=7, seq_len=96, pred_len=24):
    torch.manual_seed(0)
    kw = dict(hidden=32, seq_len=seq_len, d_model=32, d_ff=64, n_heads=4, n_blocks=2)
    build_fn = lambda: ts_models.build_ts_model(arch, n_vars, pred_len, **kw)  # noqa: E731
    example = torch.zeros(2, seq_len, n_vars)
    return build_fn, example


@pytest.mark.parametrize("arch", ["tcn", "tcn_ci", "patchtst", "itransformer", "tsmixer"])
def test_plastic_set_is_the_probed_safe_set(arch):
    build_fn, example = _cell(arch)
    model = build_fn()
    allowed = c3_random.attachable_taps(build_fn(), example, build_fn=build_fn)
    bundles = editable_bundles(model, example, allowed_taps=allowed)
    assert set(bundles) <= set(allowed)
    assert bundles, f"{arch}: nothing resizable"


@pytest.mark.parametrize("arch", ["tcn_ci", "patchtst", "tsmixer"])
def test_prune_reaches_target_and_forward_runs(arch):
    build_fn, example = _cell(arch)
    model = build_fn()
    allowed = c3_random.attachable_taps(build_fn(), example, build_fn=build_fn)
    bundles = editable_bundles(model, example, allowed_taps=allowed)
    start = deployed_params(model)
    # Halfway between the reference and the width floor: a small model's
    # plastic groups may hold less than 15% of its parameters, so a fixed
    # ratio can be unreachable here even though every real C2 target is
    # reachable by construction (TSR-X reached it).
    from bench.resize import resize_model_to_widths
    at_floor = resize_model_to_widths(build_fn(), {str(t): 8 for t in bundles}, example)
    floor = deployed_params(at_floor)
    assert floor < start
    target = floor + (start - floor) // 2
    prune_to_target(model, bundles, target, importance_l1, min_width=8)
    after = deployed_params(model)
    assert after <= target
    assert all(b.size >= 8 for b in bundles.values())
    with torch.no_grad():
        out = model.eval()(example)
    assert out.shape == (2, 24, 7)                # horizon and variates untouched


def test_taylor_importance_on_tcn():
    build_fn, example = _cell("tcn_ci")
    model = build_fn()
    bundles = editable_bundles(model, example)
    x = torch.randn(4, 96, 7)
    y = torch.randn(4, 24, 7)
    fn = make_importance_taylor(lambda: iter([(x, y)] * 2),
                                lambda m, b: F.mse_loss(m(b[0]), b[1]), n_batches=2)
    scores = fn(model, bundles)
    assert all(scores[t].shape == (b.size,) for t, b in bundles.items())
    assert any(scores[t].sum() > 0 for t in bundles)
