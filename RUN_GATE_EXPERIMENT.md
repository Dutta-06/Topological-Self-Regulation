# Gate Experiment v3 — Run Instructions

**Do not reuse `gate_v2/`** — the architecture changes (GAP bridge, connection plasticity, normalization fix) change what the numbers mean.

## TL;DR — one command

```bash
bash run_experiments.sh
```

Runs everything — `tsr_phantom`, `tsr_heuristic`, `vgg_tiny`, `vgg_small`, `static_final`, `vgg16`, `vgg19`, `resnet_sanity` — 3 seeds, 100 epochs each, into `gate_v3_full/`. `vgg16`/`vgg19` are ~15-20M params each, so expect this to take several hours on a single GPU. Use `bash run_experiments.sh --smoke` first (1 seed, 5 epochs, a few minutes) to confirm the pipeline runs clean on your machine before committing to the full run. Add `--with-pareto` to also run the λ-FLOPs budget experiment afterward into `gate_v3_pareto/`.

Everything below is reference detail for what each variant means and how to read the output — the single command above is all you need to kick it off.

---

## What changed from v2

| Change | Why | Where |
|---|---|---|
| Global Average Pool (1×1) replaces 4×4 pool | 4×4 pool caused each new conv channel to expand the classifier by 16 inputs — ballooning params and slowing training as the network grew | `tsr/model.py` |
| Bridge factor dynamic (not hardcoded ×16) | Follows the pool size automatically | `tsr/regulation/actions.py` |
| Scale-invariant phantom signal | Raw `|∂L/∂phantom_gate|` false-stops at convergence; now normalized by `mean |∂L/∂real_gate|` so signal is "is this candidate more useful than the average existing neuron?" | `tsr/regulation/phantom.py` |
| `compute_model_flops` cached between structural steps | Was recomputed every training step — O(model_size) per step, 5-10× slowdown as network grew | `scripts/gate_experiment.py` |
| **Connection-level plasticity** | TSR now grows/prunes gated skip connections between conv blocks (same phantom sensor + born-alive + sparsity machinery applied to edges). This is what allows TSR to break the plain-VGG 90% ceiling. | `tsr/layers/gated_connection.py`, `tsr/model.py`, `tsr/regulation/actions.py`, `tsr/regulation/phantom.py` |
| One-candidate-per-destination + per-destination signal normalization | An earlier free-form generator (all src→dst pairs, single global gradient scale) piled every candidate onto the last block (`*→11`), since raw gradients are largest near the loss. Now exactly one candidate per destination (`src = dst - 2`, a standard residual unit) with per-destination normalization — output fan-in is structurally impossible. | `tsr/regulation/phantom.py` (`ConnectionPhantomManager`) |
| Skip connections grow/prune in place instead of being deleted | Channel growth on an endpoint block used to delete any touching skip connection outright, forcing rediscovery from scratch and bypassing `newborn_protect_steps`. Now the projection resizes in place (with Identity→Conv2d promotion when a mismatch first appears), preserving learned weights, gate value, and birth_step. | `tsr/layers/gated_connection.py`, `tsr/regulation/actions.py` |
| `static_final` reconstructs discovered skip connections + matched head | `capture_topology()` previously didn't serialize skip connections at all (a dead `model.topology_state()` method did, but nothing read it) — so the CORE control (`tsr_phantom` vs `static_final`) was comparing a net with skips against one without. Now `final.json`'s `topology_state` includes `skip_connections` + `pool_positions`, and `FixedVGG` rebuilds them exactly, with the same GAP head as TSR. | `tsr/topology.py`, `baselines/fixed_arch.py`, `scripts/gate_experiment.py` |
| `_make_gn` normalization bug fixed | Every static baseline (`vgg_tiny`, `vgg_small`, `static_final`, `vgg16`, `vgg19`, `resnet_sanity`) was using near-InstanceNorm (1 channel/group for any channel count ≤32) instead of proper GroupNorm, while TSR itself correctly used ~8 channels/group — a hidden confound favoring TSR in every prior baseline comparison. Now matches `TSRGroupNorm` exactly. | `baselines/fixed_arch.py` |
| `resnet_sanity` baseline added | ResNet-20-style (GroupNorm, GAP head) competitor — confirms whether real residual shortcuts clear the plain-VGG ceiling under this identical recipe, independent of TSR's own discovery process. | `baselines/fixed_arch.py`, `scripts/gate_experiment.py` |

---

## Setup

```bash
git clone <repo-url>
cd Topological-Self-Regulation

uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Verify:

```bash
python -m pytest tests/ -q
# Expected: 63 passed
```

Download CIFAR-10 (once):

```bash
python -c "
from data.cifar import get_cifar10_loaders
train, val = get_cifar10_loaders('./data', batch_size=128, num_workers=0)
print(f'Train: {len(train.dataset):,}  Val: {len(val.dataset):,}')
"
```

---

## Experiment 1 — Ceiling-Break (100 epochs, no budget pressure)

**Goal:** prove TSR + connection plasticity can break 90% on CIFAR-10, where the plain-VGG architecture class (no skip connections) tops out at ~90%.

**Config:** `flops_price: 0.0` (no λ penalty — TSR grows freely). This is the default.

### Pass criteria

1. `tsr_phantom` final val acc > 90%
2. TSR discovers ≥1 surviving skip connection (visible in `final.json → topology_state → skip_connections`)
3. `tsr_phantom > static_final` gap still holds — plasticity is the cause, not just the skip topology

If TSR breaks 90% but `static_final` (trained from scratch on TSR's discovered architecture including skips) is within 1pp: the skip *shape* explains the gain, not plasticity. Flag this before claiming.

### Quick smoke first (5 min)

```bash
python scripts/gate_experiment.py \
    --seeds 42 \
    --epochs 5 \
    --variants tsr_phantom \
    --data-root ./data \
    --results-dir gate_smoke_v3 \
    --device auto
```

Should complete cleanly. Look for `skips=N` in the topology summary and `events` count > 0.

### Full run

```bash
python scripts/gate_experiment.py \
    --seeds 42 123 456 \
    --epochs 100 \
    --data-root ./data \
    --results-dir gate_v3_ceiling \
    --device auto
```

This trains all variants: `tsr_phantom`, `tsr_heuristic`, `vgg_tiny`, `vgg_small`, `static_final`.

`static_final` runs last (needs `tsr_phantom/seed42/final.json`). The script handles this automatically — it reads TSR's discovered architecture (including skip connections) and trains a static copy from scratch.

**Estimated wall time:** 6–10h on RTX Titan at 100 epochs, 3 seeds, all variants.

### Split across two machines

**Machine 1 (Titan) — TSR variants:**
```bash
python scripts/gate_experiment.py \
    --variants tsr_phantom tsr_heuristic \
    --seeds 42 123 456 \
    --epochs 100 \
    --data-root ./data \
    --results-dir gate_v3_ceiling \
    --device cuda
```

**Machine 2 (ADA 2000) — fixed-arch baselines:**
```bash
python scripts/gate_experiment.py \
    --variants vgg_tiny vgg_small \
    --seeds 42 123 456 \
    --epochs 100 \
    --data-root ./data \
    --results-dir gate_v3_ceiling \
    --device cuda
```

**After both finish — run `static_final` on either machine** (copy `gate_v3_ceiling/tsr_phantom/` from Machine 1 first):
```bash
python scripts/gate_experiment.py \
    --variants static_final \
    --seeds 42 123 456 \
    --epochs 100 \
    --data-root ./data \
    --results-dir gate_v3_ceiling \
    --device cuda
```

---

## Experiment 2 — Pareto (100 epochs, with λ-FLOPs budget pressure)

**Goal:** prove TSR achieves higher accuracy at a given inference-FLOPs budget than any fixed architecture at the same budget.

Without λ, TSR grows freely and may end up larger than necessary. With `flops_price > 0`, the λ-FLOPs penalty opposes growth, so TSR only adds capacity when the accuracy gain is worth the compute cost. This makes TSR Pareto-efficient.

**Config:** set `flops_price` in `configs/default.yaml` (or pass via CLI once that's wired). Start with `flops_price: 1e-8` and tune.

> **Note:** The λ sweep has not been calibrated yet. Run Experiment 1 first. Once Experiment 1 passes, pick a λ that keeps TSR's final inference FLOPs in a comparable range to `vgg_tiny`/`vgg_small`, then run Experiment 2.

**Comparison for the paper:** TSR-at-X-MFLOPs vs best-static-at-X-MFLOPs on a Pareto curve. TSR should dominate the curve (higher accuracy at every FLOPs point).

```bash
# Example — tune flops_price in default.yaml first
python scripts/gate_experiment.py \
    --seeds 42 123 456 \
    --epochs 100 \
    --data-root ./data \
    --results-dir gate_v3_pareto \
    --device auto
```

---

## Output files

Each run writes to `<results-dir>/<variant>/seed<N>/`:

| File | Contents |
|---|---|
| `metrics.jsonl` | One JSON per epoch: `val_accuracy`, `params`, `inference_flops`, `cumulative_train_flops`, `num_structural_events`, `topology_summary` |
| `events.jsonl` | One JSON per structural event: `action` (`prune`/`grow`/`insert_layer`/`grow_connection`/`prune_connection`), layer/edge, sizes |
| `topology.jsonl` | Full topology snapshot at each structural change, including `skip_connections` list |
| `final.json` | Summary: `best_val_accuracy`, `params`, `inference_flops`, `num_structural_events`, `topology_state` (includes skip connections for `static_final` reconstruction) |
| `checkpoint_NNNNNNN.pt` | 2 most recent only — older deleted automatically |

After all variants: `<results-dir>/summary.json` — mean ± std per variant.

---

## Reading the results

### Console at end of run

```
======================================================================
VARIANT                 VAL_ACC     ±STD   INF_MFLOPS     PARAMS
----------------------------------------------------------------------
tsr_phantom              0.XXX    0.00XX        X.XXX       XXXX
tsr_heuristic            0.XXX    0.00XX        X.XXX       XXXX
vgg_tiny                 0.XXX    0.00XX        X.XXX      19106
vgg_small                0.XXX    0.00XX        X.XXX      74426
static_final             0.XXX    0.00XX        X.XXX       XXXX
======================================================================

============================================================
ACCEPTANCE GATE
============================================================
  [PASS/FAIL] tsr_phantom vs static_final  [CORE]:   X.XXXX vs X.XXXX
  [PASS/FAIL] tsr_phantom vs tsr_heuristic [SENSOR]: X.XXXX vs X.XXXX
  [PASS/FAIL] tsr_phantom vs vgg_tiny      [PARETO]: X.XXXX vs X.XXXX
============================================================
```

**The CORE check is what matters.** `tsr_phantom > static_final` at matched params/topology proves structural plasticity, not shape discovery.

### Healthy v3 indicators in `final.json`

```json
{
  "best_val_accuracy": 0.90+,
  "params": ...,
  "topology_state": {
    "skip_connections": [
      {"src": 2, "dst": 5, "gate_value": 0.87, ...},
      ...
    ]
  }
}
```

At least one `skip_connection` with `gate_value > 0.5` should be present (TSR discovered and kept a skip). If `skip_connections` is empty, connection plasticity did not fire — check `events.jsonl` for `grow_connection` events.

---

## Key config values

All in `configs/default.yaml`:

| Key | Default | Meaning |
|---|---|---|
| `regulation.phantom_threshold` | `1.0` | Relative growth threshold: `phantom_signal / real_gate_grad_scale`. Ratio > 1 = candidate more useful than average existing neuron. |
| `regulation.connection_plasticity_enabled` | `true` | Enable/disable skip connection growth |
| `regulation.connection_threshold` | `1.0` | Same relative scale as `phantom_threshold`, for skip edges |
| `regulation.max_skip_span` | `4` | Max block distance for candidate skip edges (e.g. 4 = skips up to 4 blocks apart) |
| `regulation.newborn_gate_init` | `0.0` | Gate logit at birth for neurons and connections (sigmoid=0.5, alive) |
| `regulation.newborn_protect_steps` | `400` | Steps before a grown neuron/connection can be pruned |
| `regulation.gate_sparsity_coeff` | `0.0001` | L1 pressure on all gates (neurons + connections) — what makes pruning happen |
| `regulation.flops_price` | `0.0` | λ-FLOPs penalty coefficient. `0.0` = Experiment 1 (free growth). `> 0` = Experiment 2 (budget pressure). |
| `regulation.layer_insertion_threshold` | `2.0` | Gradient-norm ratio above which a new block is inserted |
| `regulation.max_blocks` | `24` | Hard cap on network depth |
| `regulation.update_interval` | `200` | Training steps between structural update checks |

---

## What to send back

Share the entire results directory. The `.pt` checkpoint files are gitignored; everything else is plain text.

Minimum needed to compute gate results:
```
<results-dir>/summary.json
<results-dir>/*/seed*/final.json
```

Full `metrics.jsonl`, `events.jsonl`, `topology.jsonl` needed for connection-discovery plots and topology evolution.
