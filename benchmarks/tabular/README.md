# Tabular Benchmarks (Table 1)

Baseline suite for the paper's Table 1 — tabular/MLP competitors. **TSR
itself is intentionally out of scope here** (no residual connections, no
phantom growth) — this is purely the competitor infrastructure: a plain MLP,
ResNet-MLP, FT-Transformer, TabNet, and SAINT, run under matched
config/training conditions across 5 datasets.

| Task | Datasets |
|---|---|
| Regression | California Housing (8 features, 20,640 rows) |
| Classification | Adult Income (14 features, 48,842 rows), Forest Cover Type (54 features, 581,012 rows, 7 classes), HIGGS (28 features, subsampled from 11M rows, binary), Fashion-MNIST (784 flattened-pixel features, 70,000 rows, 10 classes) |

Covertype, HIGGS, and Fashion-MNIST are the three datasets large/hard enough
to need real capacity without padding — Covertype's ResNet-MLP/FT-Transformer
configs land in the same 1-2M param range the "Revisiting Deep Learning
Models for Tabular Data" paper reports for this dataset, HIGGS's famously
non-linear decision boundary is why every deep-tabular-vs-GBDT scaling study
includes it, and Fashion-MNIST (784 features, 60K rows) is the single most
reviewer-familiar dataset in ML — chosen over plain digit MNIST because every
baseline here would saturate digit MNIST near ~97-99%, making the accuracy
column uninformative for comparing architectures; Fashion-MNIST was
purpose-built by Zalando Research as a harder, non-saturated drop-in
replacement while remaining exactly as recognizable. Adult and California
Housing are small by design — no capacity sweep needed there.

**TabPFN-v2 is not included.** It's a fixed pretrained prior-fitted
transformer, not trained per-dataset, so it can't be sized into a
param-matched Pareto sweep, and its public checkpoint's row/feature-count
comfort zone (thousands of rows) doesn't fit Covertype/HIGGS at these scales
anyway.

## Protocol

**MLP is a single fixed-size reference, deliberately not swept** — the "plain
architecture, no tricks" baseline (see each config's `models.mlp.presets`,
always exactly one preset named `m`). ResNet-MLP/FT-Transformer/TabNet sweep
2 size presets (`l`/`xl`) on Covertype/HIGGS/Fashion-MNIST — an
accuracy-vs-params Pareto point pair instead of one arbitrary size — and use
a single fixed preset (`m`) on Adult/California Housing, where a sweep
wouldn't add information (see Param counts below). SAINT gets `l`/`xl` on
Covertype/HIGGS too, but a single `m` preset on Fashion-MNIST — at 784
features its intersample attention cost hits an architectural floor (see
fidelity note) where a second, smaller preset isn't possible.

**Categorical features**: only Adult has genuine categorical columns (8 of
them); every other dataset is fully numeric (Covertype's soil-type/wilderness
columns ship pre-one-hot from UCI, so they're treated as numeric 0/1
features, not re-encoded as categoricals). MLP/ResNet-MLP/TabNet embed
categoricals and concatenate with numeric features into one flat vector;
FT-Transformer/SAINT tokenize every feature (numeric via a per-feature linear
map, categorical via embedding) into a `(B, F, d_token)` sequence.

**Parallelism**: (model, preset) combinations train **concurrently on the
same GPU** (`--max-parallel`, default 4) — each is a separate process (spawn
context, required for CUDA), same pattern as `benchmarks/timeseries`. Safe to
kill and resume: completed combinations are skipped before a process is even
spawned for them.

**Single seed (42) by default** — no variance estimate, results are point
estimates. Pass `--seeds 42 123 456` explicitly if/when a variance estimate
is needed for the final paper numbers.

## Setup

```bash
source .venv/bin/activate
uv pip install -e ".[dev]"   # sklearn + pandas already in requirements.txt
```

## Datasets

All five auto-download on first run:

- **Adult Income**: official UCI train/test files (~4MB total).
- **California Housing**: via `sklearn.datasets.fetch_california_housing` (~400KB).
- **Forest Cover Type**: via `sklearn.datasets.fetch_covtype` (~75MB).
- **HIGGS**: UCI's `HIGGS.csv.gz` (~2.6GB) — the one genuinely slow download
  in this suite; only the last 500,000 rows (test set) and a configurable
  training-row prefix (`data.num_train_rows`, default 1,000,000) are ever
  parsed, not the full 11M rows — see `data/higgs.py`'s docstring for why a
  prefix is a valid subsample here (no ordering key in the file).
- **Fashion-MNIST**: via `torchvision.datasets.FashionMNIST` (~30MB).

## Run

### One command, everything

```bash
bash run_tabular_baselines.sh              # full sweep, 1 seed
bash run_tabular_baselines.sh --smoke      # 3 epochs, mlp/m only, adult only — fast pipeline check
bash run_tabular_baselines.sh --datasets higgs covertype   # narrow the dataset list
bash run_tabular_baselines.sh --presets l       # narrow the size sweep further
bash run_tabular_baselines.sh --max-parallel 8     # more concurrent processes on the GPU
bash run_tabular_baselines.sh --models mlp resnet_mlp   # override the default model list
```

Or run each dataset individually:

```bash
python benchmarks/tabular/run_tabular.py --dataset higgs \
    --config benchmarks/tabular/configs/higgs.yaml \
    --results-dir benchmarks/tabular/results/higgs
```

Flags: `--models`, `--presets` (default: all presets defined for each model in
the config), `--seeds`, `--epochs`, `--max-parallel` (default 4).

### Quick smoke test (any command above)

Add `--models mlp --presets m --epochs 3` to run just the fixed baseline for
a few epochs — confirms the pipeline runs end-to-end before committing to the
full sweep. **Don't point this at your real `--results-dir`** — the runner
skips any `(model, preset, seed)` combination whose `final.json` already
exists, so a quick smoke result left in the real results directory will be
silently treated as "done" and never re-run at full epochs.

## Output files

Same convention as `benchmarks/timeseries`:

- `results/<dataset>/<model>/<preset>/seed<N>/{metrics.jsonl, final.json}`
- `results/<dataset>/summary.json` — mean ± std per (model, preset), sorted by
  param count ascending (Pareto-ready)

## What's in here

| File | Contents |
|---|---|
| `data/adult.py` | UCI Adult loader (official train/test split, 6 numeric + 8 categorical, "?" kept as its own category) |
| `data/california_housing.py` | California Housing loader (8 numeric, regression, target standardized for training + unstandardized for scoring) |
| `data/covertype.py` | Forest Cover Type loader (54 numeric — 10 continuous + 44 pre-one-hot indicators — 7-class, fixed-seed 80/10/10 split) |
| `data/higgs.py` | HIGGS loader (28 numeric, last-500K-rows test convention, configurable training-row prefix) |
| `data/fashion_mnist.py` | Fashion-MNIST loader (784 flattened-pixel numeric features, official 60K/10K split) |
| `data/common.py` | Shared `TabularDataset`, `CategoryEncoder` (train-vocab + unknown-bucket label encoding), `standardize` |
| `models/mlp.py` | Plain MLP — Linear/BatchNorm/ReLU/Dropout stack |
| `models/resnet_mlp.py` | ResNet-MLP (Gorishniy et al. 2021) — residual Linear blocks |
| `models/ft_transformer.py` | FT-Transformer (Gorishniy et al. 2021) — per-feature tokenizer + CLS + Transformer encoder |
| `models/tabnet.py` | TabNet (Arik & Pfister 2019) — sequential sparse-attention decision steps; see docstring for the simplification vs. the paper |
| `models/saint.py` | SAINT (Somepalli et al. 2021) — feature self-attention + intersample (across-batch) attention |
| `models/common.py` | Shared `CatEmbedding` (flat categorical embedding) and `FeatureTokenizer` (per-feature tokenizer for the two Transformer models) |
| `run_tabular.py` | Trains/evals all 5 models, all presets, on a dataset across seeds (classification + regression) |
| `configs/*.yaml` | Per-dataset training + per-model size presets, TSR-style sectioned YAML |

## Notes on fidelity

- **Mixed precision (AMP)**: `run_tabular.py` uses `torch.amp.autocast` +
  `GradScaler` automatically whenever `device.type == "cuda"` (a no-op on
  CPU), same as the timeseries runners.
- **TabNet**: implements the paper's core mechanism — sparsemax-projected
  attention masks (implemented from scratch, Martins & Astudillo 2016),
  prior-scale feature-reuse penalty (`gamma`), and decision-output
  aggregation across steps — but each decision step's "feature transformer"
  uses fully independent GLU-block weights, rather than the paper's 2 blocks
  shared across all steps + 2 step-specific ones. This drops a
  parameter-sharing efficiency trick but keeps every mechanism that defines
  TabNet as an architecture family; `n_d`/`n_a`/`n_steps` still control param
  count the same way. Known from the literature (and part of why it's
  included as a baseline, not a scaling target) that TabNet's capacity
  doesn't scale as cleanly as ResNet-MLP/FT-Transformer's — the `xl` preset
  shows diminishing returns relative to its param count, which is expected.
- **SAINT**: implements both attention passes from the paper — feature
  self-attention (like FT-Transformer) and intersample attention (reshape
  each sample's tokens into a flat vector, then attend *across the batch*,
  implemented via a `(1, B, F*d_token)` reshape so `nn.MultiheadAttention`
  treats the batch dimension as its sequence). The intersample sublayer
  **skips the FFN** that would normally follow attention in a transformer
  block — its embed dim (`F*d_token`) is already large before any FFN
  widening, and a `d_ffn_factor`-scaled FFN there caused param count to blow
  up quadratically with feature count for no accuracy benefit (the
  feature-attention sublayer already provides FFN capacity). Even without it,
  SAINT is inherently costlier per `d_token` than FT-Transformer once feature
  count grows, which is why its presets use a noticeably smaller `d_token`
  than the other Transformer baseline (see param table below) — a real
  architectural characteristic, not a bug. One caveat inherent to the
  architecture itself (not specific to this implementation): because
  intersample attention mixes information across samples in the same
  minibatch, a given test sample's prediction is technically dependent on
  which other samples share its eval batch — expected SAINT behavior, not a
  correctness issue, since eval batches are built deterministically
  (`shuffle=False`). On Fashion-MNIST (784 features), intersample attention's
  `(F*d_token)^2` cost hits its floor: `d_token=1` (the minimum possible)
  already costs ~2.5M params for one intersample layer, so there's no smaller
  setting to pair it with for an `l`/`xl` sweep — SAINT gets a single `m`
  preset on that dataset instead (see configs/fashion_mnist.yaml). This is a
  real, known scaling limitation of SAINT's intersample-attention design at
  high feature counts, not specific to this implementation.
- **FT-Transformer / SAINT tokenizer**: numeric tokenization follows the
  FT-Transformer paper's convention — a per-feature learned linear map
  (`x_i * w_i + b_i`), not a single shared linear layer — so every feature
  gets its own scale/shift into token space before attention mixes anything.
- **Size presets**: `l`/`xl` tiers (and the single `m` tier for small
  datasets) are chosen per-architecture using each model's natural scaling
  knob — they are **not** hand-tuned to produce identical param counts across
  model types. The Pareto curve (accuracy vs. actual param count) is the
  comparison that matters, not nominal tier labels.

## Param counts (per dataset)

Computed directly from each config (no training needed — pure architecture
size).

| Dataset | MLP (m) | ResNet-MLP (l/xl) | FT-Transformer (l/xl) | TabNet (l/xl) | SAINT |
|---|---|---|---|---|---|
| Adult (single `m`) | 9.9K | 39.0K | 29.2K | 80.1K | 119K (single `m`) |
| California Housing (single `m`) | 5.1K | 34.2K | 26.0K | 40.8K | 44.2K (single `m`) |
| **HIGGS** | **980K** | **1.02M / 2.12M** | **1.01M / 2.24M** | **910K / 2.28M** | **1.74M / 3.28M (l/xl)** |
| **Covertype** | **991K** | **1.02M / 2.12M** | **1.02M / 2.25M** | **1.01M / 2.45M** | **1.56M / 2.94M (l/xl)** |
| **Fashion-MNIST** | **773K** | **747K / 1.78M** | **633K / 1.23M** | **658K / 1.22M** | **2.47M (single `m`, floor — see fidelity note)** |

SAINT runs somewhat higher than the other three swept models at nominally
comparable settings on HIGGS/Covertype — expected, given intersample
attention's parameter cost (see fidelity note above), not a config error.
