# Structured-pruning baselines

The paper compares the TSR-X allocation with a dense reference (Ref), the
discovered width vector retrained from scratch (C2), and, in forecasting, a
random allocation of the same size (C3). None of those is the obvious way to
reach a smaller model. This branch adds it: prune the trained reference to the
same deployed count and see whether the allocation TSR-X found is any better
than the one pruning finds.

## What is being tested

**Vision (fixed 85% budget).** At C2's exact deployed count, on the same
plastic set, is the TSR-X allocation as good as pruning's? Outcomes:

- pruned ~ C2: the accuracy result is not distinctive; TSR-X's case rests on the
  mechanism (one training run discovers the shape, no dense pre-training) and
  on compute, which must then be reported.
- C2 > pruned: the reallocation does something pruning cannot. The sharp
  version is growth: pruning only removes, while both case-study runs put
  capacity *into* an early cheap site (layer1.1, conv1.1).
- C2 < pruned: a magnitude-ranked allocation beats the task-directed one; the
  parity-with-dense result survives, the allocation story does not.

**Forecasting (swept budget).** Does TSR-X reach a tighter knee than pruning
under the same 3% tolerance? Compare the two frontiers over the same plastic
groups. This is the suite where the paper's headline (attainable compression)
lives, so it matters more there.

## Why it is built on the TSR-X engine, not torch-pruning

A fair baseline needs the same action space. The core, `tsrx/edit/pruning.py`,
removes channels with `tsrx.edit.edits.prune_group_index`, the same by-index
edit the controller uses, over exactly the coupling groups `CandidateBank` attaches to
(producer + consumer, quantum 1, no task-fixed axis). Residual trunks,
depthwise pairs and SE branches are therefore edited as coupled groups, the
parameter count is the real tensor extent, and the width floor (8) is TSR-X's.
The only thing the baseline cannot do is grow, which is the point.

Criteria, ranked globally after per-group mean normalisation:

| criterion | importance | reference |
|---|---|---|
| `l1` | L1 norm of the channel's producer weights | Li et al., 2017 |
| `bnscale` | mean \|gamma\| over the group's BatchNorm scales | Liu et al., 2017 (network slimming) |
| `taylor` | \|activation x gradient\| via `tsrx.sense.saliency.first_order_saliency`, i.e. the controller's own removal statistic | Molchanov et al., 2017 |

Pruning is greedy and iterative: importance is recomputed after every round
(at most 5% of the remaining channels), and removal stops when the deployed
count is at or below the target. The last channel can undershoot by at most
its own cost; the runner refuses a result more than 1% under the target.

Two modes, both under the shared training recipe:

- `finetune` (default): keep the reference weights, fine-tune 40 epochs at
  lr 0.01 with cosine decay. The practical pruning pipeline.
- `scratch`: rebuild the pruned width vector with fresh initialisation and
  train 100 epochs at lr 0.1, exactly as C2 is trained. This is the
  apples-to-apples comparison of *allocations* (Liu et al., 2019), since C2
  never sees the reference weights either.

## Running

Vision (`bench/prune_baseline.py`):

```bash
# one cell
python -m bench.prune_baseline --arch resnet18 --dataset cifar100 --criterion l1 \
    --mode finetune --num-workers 4 --out results/pruned/resnet18_cifar100_l1_finetune.pt

# every vision cell that has a C2 arm; resumable
python scripts/run_pruning_baselines.py --criteria l1 bnscale --modes finetune
python scripts/run_pruning_baselines.py --criteria l1 --modes scratch --cells resnet18/cifar100

# tabulate against Ref / C2 / TSR
python scripts/eval_pruning_baselines.py
```

The reference is chosen by inspecting the checkpoint (ResNet's stem must match
the cell; aborted checkpoints are refused), and the target is read from the
matching `results/static_matched/*_two_regime.pt`. Each output carries the
reference's and the pruned-before-training top-1, the removed-channel log
(`*.events.json`), and `discovered_widths` in the same form as a TSR-X
checkpoint, so `bench/train_static_matched.py` can rebuild it.

Forecasting (`bench/prune_baseline_ts.py`, written against the `tsrx-time`
harness: `bench.ts_models`, `bench.resize`, `data.ltsf`, `bench.c3_random`):

```bash
# one cell: the C2 checkpoint names the cell and fixes the target count
python -m bench.prune_baseline_ts --match results/ts_static_matched/tcn_ci_weather_h96.pt \
    --criterion l1 --mode finetune --out results/ts_pruned/tcn_ci_weather_h96_l1_finetune.pt

# every archived C2 (every backbone x dataset x horizon x budget); resumable
python scripts/run_pruning_baselines_ts.py --criteria l1 taylor --modes finetune
python scripts/run_pruning_baselines_ts.py --only weather_h96 --modes scratch

# tabulate against Ref / C2 / TSR-X / C3
python scripts/eval_pruning_baselines_ts.py
```

The forecasting driver takes its plastic set from `bench.c3_random.attachable_taps`
with the safety probe, i.e. the same allowlist C3 and the discovery run used,
so LayerNorm-spanned residual widths and attention projections are never
pruned. It inherits the discovery run's lr, weight decay and batch size (a
PatchTST fine-tuned at the TCN's lr is not a matched control), fine-tunes 20
epochs by default, and in `scratch` mode trains 50 epochs exactly as C2 does.
`bnscale` applies to the TCN only; other backbones fall back to `l1`, so use
`l1` + `taylor` there. The knee comparison is then C2's test MSE against the
pruned model's at the same count, and the frontier comparison is the same over
every budget the campaign archived.

The forecasting driver and its test only import when the `tsrx-time` modules
are present: run them on that branch or on a merge of the two. The vision
driver, the core and their tests run here.

## Cost

Per criterion, prune + fine-tune, from the measured 100-epoch static runs:

| | CIFAR-10 | CIFAR-100 | Tiny-IN | ImageNet-100 |
|---|---|---|---|---|
| RTX 4060 laptop | ~1 h / cell | ~1 h / cell | 1–3 h / cell | 3–20 h / cell |
| Blackwell 6000 | ~12 min | ~12 min | ~10–25 min | 0.5–2.5 h |

About 48 h per criterion on the laptop for the 13 complete cells, ~6.5 h on
Blackwell; `scratch` mode costs 2.5x. Forecasting is small by comparison
(roughly 19 h per criterion on the laptop for the 12 knee cells with 50-epoch
fine-tunes, dominated by PatchTST on Electricity/Traffic; ~3 h on Blackwell;
the 20-epoch default is 0.4 of that). The three ImageNet-100 cells without a C2
arm (VGG-16-BN, MobileNetV2, EfficientNet-B0) must have their Ref/TSR/C2 arms
trained first; the runner skips them until then.

## Forecasting: what differs from vision

- The count is matched per C2 checkpoint, not per cell: the campaign archives
  one C2 per budget, so the pruned frontier is built by matching each one.
- Two criteria (`l1`, `taylor`); no BatchNorm scale on the transformer/MLP
  backbones.
- The reference is the cell's `results/ts_reference/<tag>.pt`, located from the
  TSR-X checkpoint the C2 records.
- Metric is test MSE at the best-validation epoch, saved alongside MAE, as in
  the other arms, so `analysis/collect_results.py` conventions carry over.

## Paper

Vision: one added column per criterion in `tab:vision` (or a sibling table in
App. vision-results), "C2 − pruned" with the sign convention already used.
Forecasting: the pruned knee beside the TSR-X knee in `tab:timeseries`, and the
pruned frontier in App. ts-results. The Discussion paragraph naming pruning as
the required missing control is then replaced by the result, whichever way it
goes.
