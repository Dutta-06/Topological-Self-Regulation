# FLOP-budgeted discovery (branch `flops-budget`)

The released controller enforces its budget in deployed parameters, which on
several vision cells moved capacity toward early, spatially wide layers and
*raised* forward FLOPs (up to +47.5% for VGG-16-BN on Tiny-ImageNet). This
branch lets the same controller enforce the budget in forward FLOPs instead.

## What changed

- `tsrx/alloc/cost.py`: `model_flops(model, example)` (exact 2·MAC count over
  conv/linear, the repo's reporting convention), `deployed_flops(bank, example)`
  (counted on a candidate-free copy, the FLOP analogue of `deployed_params`),
  and `make_cost_fn(mode, traced)` returning `kappa_params` or `kappa_flops`
  bound to the traced shapes.
- `tsrx/sense/candidates.py`: `CandidateBank.detached_copy()`.
- `tsrx/alloc/exchange.py`: every proposal density takes the cost function as a
  parameter (`cost_fn`, default `kappa_params`); the released behaviour is
  unchanged.
- `bench/train_tsrx.py`: `--cost {params,flops}`. In FLOP mode the baseline,
  the annealed ceiling and the deployed count at every structural update are
  FLOPs; decision records carry `cost_mode`, `deployed_*` in the budget unit
  and `deployed_params_*` alongside; checkpoints carry `baseline_flops` and
  `flops`.
- `tests/test_flops_cost.py`: `kappa_flops` equals the finite-difference FLOP
  change of one channel to within 2% on every ResNet-18 group; `deployed_flops`
  ignores candidates and tracks edits; the factory returns both costs.

`kappa_flops` was checked against finite differences on EfficientNet-B0 as
well: 30 of 36 groups within 2%, the rest 3–8% high because it also counts
the BatchNorm and bias adds the reporting counter ignores.

## Running

```bash
python -m pytest tests/test_flops_cost.py -q

# EfficientNet-B0 on both CIFARs, discovery + C2, concurrently; resumable
python scripts/run_flops_variant.py --num-workers 4
python scripts/eval_flops_variant.py
```

One cell by hand:

```bash
python -m bench.train_tsrx --arch efficientnet_b0 --dataset cifar10 --cost flops \
    --budget-ratio 0.85 --epochs 100 --seed 42 --out results/tsrx_flops/efficientnet_b0_cifar10_flops.pt
python -m bench.train_static_matched --tsrx-checkpoint results/tsrx_flops/efficientnet_b0_cifar10_flops.pt \
    --epochs 100 --seed 42 --out results/static_matched_flops/efficientnet_b0_cifar10_flops.pt
```

The dense reference is shared with the parameter-budget campaign; only the
discovery run and its C2 retrain are new. The case-study extractor replays
`deployed_before/after` against exact parameter counts and therefore does not
apply to FLOP-mode logs as is; those logs carry `deployed_params_*` for a
future replay in parameters.

## What it answers

At the same 85% ratio, now in FLOPs: does the controller find an allocation
that keeps accuracy while *reducing* compute, and what does that allocation
look like against the parameter-budgeted one? The comparison row is the
archived parameter-budget cell (Ref, C2, TSR) beside the FLOP-budget cell
(C2, TSR) with both ΔP and ΔF reported for each.
