# Gate Experiment v2 — Run Instructions

This is the second smoke test for the TSR rewrite. **Do not reuse the `gate/` directory from v1** — the architectural fixes change what the numbers mean.

## What changed from v1

The v1 smoke test revealed a bug: newborn neurons were born with gate logit `−5.0` (sigmoid≈0.007), which is *below* the death threshold of 0.01. The phantom sensor correctly measured that capacity was needed and materialized a neuron — but it was born already dead, and the sparsity penalty ensured it never woke up. The result was 220 events of grow→fail→prune with params pinned deterministically at the seed value across all 3 seeds (std=0.0).

**Three fixes are in:**

| Fix | What changed | Where |
|---|---|---|
| E1: Born-alive | New neurons start at gate logit `0.0` (sigmoid=0.5, above death threshold) | `tsr/layers/tsr_conv.py`, `tsr/layers/tsr_linear.py` |
| E2: Newborn protection | `neuron_birth_step` buffer tracks each neuron's birth step; death signal skips neurons younger than `newborn_protect_steps=400` steps | `tsr/regulation/signals.py`, both layer files |
| Gate fix | Acceptance gate now compares TSR vs `static_final` (matched params) as the core test, not TSR vs `vgg_small` (12× params) | `scripts/gate_experiment.py` |

**What to look for in v2 vs v1:**

| Signal | v1 (broken) | v2 (expected) |
|---|---|---|
| `params` std across seeds | 0.0 (pinned) | > 0 (actual growth varies) |
| Final params vs seed params | ≈ equal (no net growth) | > seed (growth sticks) |
| `num_structural_events` | 220 (thrashing) | Fewer, fewer reversals |
| TSR vs static_final gap | +10.3pp (curriculum only) | ≥10pp, now also from genuine capacity growth |

---

## 1. Setup

```bash
git clone <repo-url>
cd Topological-Self-Regulation

uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest tests/ -q
```

All 63 tests should pass.

---

## 2. Download CIFAR-10

Run once:

```bash
python -c "
from data.cifar import get_cifar10_loaders
train, val = get_cifar10_loaders('./data', batch_size=128, num_workers=0)
print(f'Train: {len(train.dataset):,}  Val: {len(val.dataset):,}')
"
```

Expected:
```
Files already downloaded and verified
Train: 50,000  Val: 10,000
```

---

## 3. Run the Gate Experiment

**Use `gate_v2/` as the results dir** to keep v1 results intact for comparison.

### Option A — Single machine, full run

```bash
python scripts/gate_experiment.py \
    --seeds 42 123 456 \
    --epochs 50 \
    --data-root ./data \
    --results-dir gate_v2 \
    --device auto
```

Trains all five variants sequentially:
1. `tsr_phantom` — TSR with phantom-sensor growth (the rewrite + v2 fixes)
2. `tsr_heuristic` — TSR with heuristic bottleneck score
3. `vgg_tiny` — FixedVGG `[8, 8, 16]`
4. `vgg_small` — FixedVGG `[16, 16, 32]`
5. `static_final` — FixedVGG matching TSR's discovered shape, trained from scratch

`static_final` runs last (needs `tsr_phantom/seed42/final.json`). The script handles this automatically.

**Estimated wall time:** ~3–5h on ADA 2000 / RTX Titan at 50 epochs, 3 seeds.

### Option B — Split across two machines

**Machine 1 (Titan) — TSR variants:**

```bash
python scripts/gate_experiment.py \
    --variants tsr_phantom tsr_heuristic \
    --seeds 42 123 456 \
    --epochs 50 \
    --data-root ./data \
    --results-dir gate_v2 \
    --device cuda
```

**Machine 2 (ADA 2000) — fixed-arch baselines (run in parallel):**

```bash
python scripts/gate_experiment.py \
    --variants vgg_tiny vgg_small \
    --seeds 42 123 456 \
    --epochs 50 \
    --data-root ./data \
    --results-dir gate_v2 \
    --device cuda
```

**After both finish — run `static_final` on either machine** (needs `tsr_phantom/seed42/final.json` from Machine 1):

```bash
python scripts/gate_experiment.py \
    --variants static_final \
    --seeds 42 123 456 \
    --epochs 50 \
    --data-root ./data \
    --results-dir gate_v2 \
    --device cuda
```

Then copy all `gate_v2/` directories to one machine before running the summary step.

### Option C — Quick smoke test (5 min, 1 seed, 5 epochs)

```bash
python scripts/gate_experiment.py \
    --seeds 42 \
    --epochs 5 \
    --data-root ./data \
    --results-dir gate_smoke_v2 \
    --device auto
```

Use this to verify the pipeline end-to-end. At 5 epochs params will not show much growth but the pipeline should run clean and newborns should survive.

---

## 4. Resuming

Safe to interrupt and resume — any run with a `final.json` is skipped automatically. Re-run the same command.

---

## 5. Output Files

Each run writes to `gate_v2/<variant>/seed<N>/`:

| File | Contents |
|---|---|
| `metrics.jsonl` | One JSON line per epoch: `step`, `epoch`, `val_accuracy`, `val_loss`, `params`, `effective_params`, `inference_flops`, `cumulative_train_flops`, `num_structural_events`, `topology_summary` |
| `events.jsonl` | One JSON line per structural event (TSR only): `step`, `layer_name`, `action` (`prune`/`grow`/`insert_layer`), `old_size`, `new_size` |
| `topology.jsonl` | Full topology snapshot at each structural change (layer widths, gate stats, dominant activation) |
| `final.json` | Summary at run end: `best_val_accuracy`, `params`, `inference_flops`, `total_train_flops`, `num_structural_events`, `final_topology`, `topology_state` |
| `checkpoint_NNNNNNN.pt` | Only the **2 most recent** per run — older ones deleted automatically |

After all variants and seeds:

| File | Contents |
|---|---|
| `gate_v2/summary.json` | Mean ± std per variant across seeds |

---

## 6. Reading the Results

### Console output

```
======================================================================
VARIANT              VAL_ACC        ±STD   INF_MFLOPS     PARAMS
----------------------------------------------------------------------
tsr_phantom           0.XXX      0.00XX        X.XXX       XXXX
tsr_heuristic         0.XXX      0.00XX        X.XXX       XXXX
vgg_tiny              0.XXX      0.00XX        X.XXX      19106
vgg_small             0.XXX      0.00XX        X.XXX      74426
static_final          0.XXX      0.00XX        X.XXX       XXXX
======================================================================

============================================================
ACCEPTANCE GATE
============================================================
  [PASS/FAIL] tsr_phantom vs static_final  [CORE]:   X.XXXX vs X.XXXX  (delta=+X.XXXX)
  [PASS/FAIL] tsr_phantom vs tsr_heuristic [SENSOR]: X.XXXX vs X.XXXX  (delta=+X.XXXX)
  [PASS/FAIL] tsr_phantom vs vgg_tiny      [PARETO]: X.XXXX vs X.XXXX  (delta=+X.XXXX)
============================================================
  OVERALL: PASSED / FAILED
============================================================
```

**The CORE check is the one that matters.** TSR vs static_final at matched params — this proves structural plasticity, not just shape discovery.

### Healthy growth indicators in `final.json`

In v2, you should see:
```json
"params": 7800,
"num_structural_events": 40,
"final_topology": "TSR[conv=12/12→14/14, ...]"
```

In v1, params was always 6136 with std=0.0 and topology stuck at 8/9→8/9 (no net growth).

### Generate paper-ready outputs

```bash
# After all variants complete
python analysis/ablation_table.py --summary gate_v2/summary.json

# Topology evolution plots for a TSR run
python analysis/topology_plot.py --run-dir gate_v2/tsr_phantom/seed42
```

---

## 7. Key Config Values

All in `configs/default.yaml`:

| Key | Default | Meaning |
|---|---|---|
| `regulation.newborn_gate_init` | `0.0` | Gate logit for newly grown neurons (sigmoid=0.5, alive) |
| `regulation.newborn_protect_steps` | `400` | Steps before a grown neuron can be pruned (2 update cycles) |
| `regulation.phantom_threshold` | `0.05` | Min phantom gate-gradient to trigger growth |
| `regulation.gate_sparsity_coeff` | `0.0001` | L1 pressure on open gates |
| `regulation.update_interval` | `200` | Steps between structural update checks |
| `regulation.growth_signal_mode` | `phantom` | `phantom` or `heuristic` |
| `training.batch_size` | `128` | |
| `training.learning_rate` | `0.001` | |

---

## 8. What to Send Back

After the run, share the entire `gate_v2/` directory. The `.pt` checkpoint files are gitignored; everything else (`.jsonl`, `.json`, `.csv`, `.tex`) is plain text.

Minimum needed to compute the gate result:
```
gate_v2/summary.json
gate_v2/*/seed*/final.json
```

Full `metrics.jsonl`, `events.jsonl`, `topology.jsonl` needed for topology plots.
