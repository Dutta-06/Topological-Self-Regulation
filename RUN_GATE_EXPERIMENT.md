# Gate Experiment — Run Instructions

This is the acceptance gate for the TSR rewrite. It trains five model variants on CIFAR-10 under identical conditions and checks whether **TSR (phantom sensor mode)** beats the pre-rewrite TSR baseline and the fixed-architecture baselines it previously claimed to beat.

**Hardware required:** Any CUDA GPU with ≥8GB VRAM. Tested target: RTX 2000 Ada (16GB). The models are small (CIFAR-10, 32×32 input) — peak memory is well under 2GB even with phantom probes enabled.

**Estimated wall time:** ~3–5 hours for 3 seeds × 5 variants at 50 epochs. Scale linearly with `--epochs`.

---

## 1. Setup

Clone the repo and set up the environment. The project uses `uv` for dependency management.

```bash
git clone <repo-url>
cd Topological-Self-Regulation

# Create venv and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Verify the install:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pytest tests/ -q
```

All 63 tests should pass. If `torch.cuda.is_available()` returns `False`, add `--device cpu` to all commands below (slow but functional for a smoke test).

---

## 2. Download CIFAR-10

Run this once. It downloads ~170MB to `./data/` and exits.

```bash
python -c "
from data.cifar import get_cifar10_loaders
train, val = get_cifar10_loaders('./data', batch_size=128, num_workers=0)
print(f'Train: {len(train.dataset):,}  Val: {len(val.dataset):,}')
"
```

Expected output:
```
Files already downloaded and verified
Train: 50,000  Val: 10,000
```

---

## 3. Run the Gate Experiment

### Option A — Run everything on one machine

```bash
python scripts/gate_experiment.py \
    --seeds 42 123 456 \
    --epochs 50 \
    --data-root ./data \
    --results-dir results/gate \
    --device auto
```

This trains all five variants sequentially:
1. `tsr_phantom` — TSR with measured phantom-sensor growth signal (the rewrite)
2. `tsr_heuristic` — TSR with original bottleneck-score growth signal (pre-rewrite)
3. `vgg_tiny` — FixedVGG `[8, 8, 16]`
4. `vgg_small` — FixedVGG `[16, 16, 32]`
5. `static_final` — FixedVGG matching TSR's discovered shape, trained from scratch

`static_final` automatically reads the channel configuration from `tsr_phantom/seed42/final.json`, so it must run after the phantom run completes. The script handles this ordering automatically.

### Option B — Split across two machines (recommended)

**Machine 1 (Titan / higher VRAM) — TSR variants:**

```bash
python scripts/gate_experiment.py \
    --variants tsr_phantom tsr_heuristic \
    --seeds 42 123 456 \
    --epochs 50 \
    --data-root ./data \
    --results-dir results/gate \
    --device cuda
```

**Machine 2 (ADA 2000 / any GPU) — fixed-arch baselines:**

```bash
python scripts/gate_experiment.py \
    --variants vgg_tiny vgg_small \
    --seeds 42 123 456 \
    --epochs 50 \
    --data-root ./data \
    --results-dir results/gate \
    --device cuda
```

**After both finish — run `static_final` on either machine** (needs `tsr_phantom/seed42/final.json` from Machine 1 to exist first):

```bash
python scripts/gate_experiment.py \
    --variants static_final \
    --seeds 42 123 456 \
    --epochs 50 \
    --data-root ./data \
    --results-dir results/gate \
    --device cuda
```

Then copy all `results/gate/` directories to one machine before running the summary step (Section 5).

### Option C — Quick smoke test (5 minutes, 1 seed, 5 epochs)

```bash
python scripts/gate_experiment.py \
    --seeds 42 \
    --epochs 5 \
    --data-root ./data \
    --results-dir results/gate_smoke \
    --device auto
```

Use this to verify the pipeline end-to-end before committing to a full run.

---

## 4. Resuming a Partial Run

The script is **safe to interrupt and resume**. Any run with a `final.json` already written is skipped automatically. Just re-run the same command — it picks up where it left off.

```bash
# Same command as before — skips completed runs
python scripts/gate_experiment.py --seeds 42 123 456 --epochs 50
```

---

## 5. Output Files

Each run writes to `results/gate/<variant>/seed<N>/`:

| File | Contents |
|---|---|
| `metrics.jsonl` | One JSON line per epoch: `step`, `epoch`, `val_accuracy`, `val_loss`, `params`, `inference_flops`, `cumulative_train_flops`, `num_structural_events`, `topology_summary` |
| `events.jsonl` | One JSON line per structural event (TSR variants only): `step`, `layer_name`, `action` (`prune`/`grow`/`insert_layer`), `old_size`, `new_size` |
| `topology.jsonl` | Full topology snapshot at each structural change, plus start and end: per-layer width, gate mean/min/max, dominant activation |
| `final.json` | Summary scalars at run end: `best_val_accuracy`, `params`, `inference_flops`, `total_train_flops`, `num_structural_events`, `final_topology`, full `topology_state` |
| `checkpoint_NNNNNNN.pt` | Model + optimizer checkpoint. Only the **2 most recent** are kept per run — older ones are deleted automatically to avoid filling disk. |

After all variants and seeds complete, the script writes:

| File | Contents |
|---|---|
| `results/gate/summary.json` | Mean ± std for every metric across seeds, per variant |

---

## 6. Reading the Results

### Console output (printed automatically at the end)

A results table:

```
======================================================================
VARIANT              VAL_ACC        ±STD   INF_MFLOPS     PARAMS
----------------------------------------------------------------------
tsr_phantom           0.8234      0.0041        1.234       6821
tsr_heuristic         0.7891      0.0058        1.187       5436
vgg_tiny              0.7712      0.0033        0.891       4201
vgg_small             0.8019      0.0044        2.341      12800
static_final          0.8101      0.0039        1.234       6821
======================================================================
```

Followed by the gate check:

```
============================================================
ACCEPTANCE GATE
============================================================
  [PASS] tsr_phantom vs tsr_heuristic: 0.8234 vs 0.7891  (delta=+0.0343)
  [PASS] tsr_phantom vs vgg_tiny:      0.8234 vs 0.7712  (delta=+0.0522)
  [PASS] tsr_phantom vs vgg_small:     0.8234 vs 0.8019  (delta=+0.0215)
============================================================
  OVERALL: PASSED
============================================================
```

The script **exits with code 0** if the gate passes, **code 1** if it fails.

### Generate paper-ready outputs

Run these after the experiment completes (on whichever machine has all `results/gate/` data):

```bash
# Ablation table: prints ASCII, writes CSV + LaTeX + Pareto plot PNG
python analysis/ablation_table.py --summary results/gate/summary.json

# Topology evolution plots for a TSR run (width, gates, activations, event timeline)
python analysis/topology_plot.py --run-dir results/gate/tsr_phantom/seed42
```

Outputs written to `results/gate/`:

| File | Description |
|---|---|
| `ablation_table.csv` | Machine-readable results table |
| `ablation_table.tex` | LaTeX `booktabs` table, paste directly into the paper |
| `pareto_frontier.png` | Accuracy vs. inference MFLOPs scatter plot |
| `tsr_phantom/seed42/plots/width_evolution.png` | Per-layer channel count over training steps |
| `tsr_phantom/seed42/plots/gate_evolution.png` | Mean gate value per layer over time |
| `tsr_phantom/seed42/plots/activation_evolution.png` | Dominant activation heatmap |
| `tsr_phantom/seed42/plots/events_timeline.png` | Structural events overlaid on val accuracy curve |

---

## 7. Key Config Values

All defaults are in `configs/default.yaml`. The most relevant for this experiment:

| Key | Default | Meaning |
|---|---|---|
| `regulation.growth_signal_mode` | `phantom` | `phantom` = rewrite; `heuristic` = pre-rewrite |
| `regulation.phantom_threshold` | `0.05` | Min phantom gate-gradient to trigger growth |
| `regulation.gate_sparsity_coeff` | `0.0001` | L1 pressure on open gates (makes pruning fire) |
| `regulation.update_interval` | `200` | Steps between structural update checks |
| `regulation.bottleneck_threshold` | `0.1` | Heuristic-mode growth threshold |
| `training.batch_size` | `128` | |
| `training.learning_rate` | `0.001` | |
| `training.max_epochs` | `200` | Overridden by `--epochs` flag |

Override any config value via the CLI flags (`--epochs`, `--batch-size`) or by editing `configs/default.yaml` directly.

---

## 8. What to Send Back

After the run, please share the entire `results/gate/` directory. The `.pt` checkpoint files are gitignored (large binaries); everything else — `.jsonl`, `.json`, `.csv`, `.tex` — is plain text and can be committed or zipped and shared directly.

The minimum needed to compute the gate result is:
```
results/gate/summary.json
results/gate/*/seed*/final.json
```

The full `metrics.jsonl`, `events.jsonl`, and `topology.jsonl` files are needed for the topology plots and learning curve figures.
