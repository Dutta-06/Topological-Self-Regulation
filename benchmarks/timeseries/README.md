# Time-Series Benchmarks (Table 3)

Baseline suite for the paper's Table 3 — forecasting and classification
competitors. **TSR itself is intentionally out of scope here** (no residual
connections, no phantom growth) — this is purely the competitor infrastructure:
LSTM, GRU, TCN, PatchTST, and a Mamba-style selective-SSM, run under matched
config/training conditions across 5 datasets.

| Task | Datasets |
|---|---|
| Forecasting | ETTh1, ETTh2, UCI Electricity Load (ECL, 321 clients), Weather (MPI Jena, 21 channels) |
| Classification | UCI HAR, UCR/UEA archive (subset) |

Electricity uses **321 client channels** (not a small subset) — the standard
ECL benchmark scale, selected by earliest-install-date (see
`data/electricity.py`'s docstring for why this rule, not the original
papers' unpublished exact client list). At this scale, `l`→`xl` presets span
**~2.1M → ~4.4M params** at the shortest horizon alone, and up to ~15M+ at
longer horizons (see the 5-20M table below) — genuinely large models, because
321 channels actually need it, not because anything was padded. Weather
completes the standard 4-dataset LTSF-forecasting subset we cover (of the
canonical 8: ETTh1/ETTh2/ETTm1/ETTm2/Electricity/Traffic/Weather/Exchange —
we don't yet have ETTm1/ETTm2/Traffic/Exchange).

**Traffic (862-sensor PEMS) and the PEMS-SF classification dataset are not
included.** The continuous "Traffic" forecasting benchmark used by Informer/
Autoformer/PatchTST is distributed by those papers' own repos, not from a
clean institutional URL we could verify here — the UCI-hosted "PEMS-SF" at
this same name is actually a *different*, day-of-week *classification*
dataset (963-dim, UEA archive format), and even that failed to load through
`aeon` in repeated attempts in this environment (timed out both times,
likely a slow/unreachable upstream mirror for this specific dataset). It's
reachable via `aeon.datasets.load_classification('PEMS-SF')` in principle —
worth retrying on a machine with better bandwidth to `aeon`'s data mirror.

## Protocol

**Forecasting** follows the standard multivariate long-horizon convention
(Informer/Autoformer/PatchTST): given a `seq_len`-step window of ALL channels,
predict the next `pred_len` steps of ALL channels; report MSE and MAE over the
full `(pred_len, channels)` target. `pred_len` sweeps the standard ETT horizon
set `{96, 192, 336, 720}` — **not** a single scalar/single-channel target, so
numbers are comparable to published results at the same operating point.

**Every model runs at 2 size presets** (`l`/`xl`, see each config's
`models.<name>.presets`) rather than one arbitrary size. This turns each
(dataset, horizon) cell into an **accuracy-vs-params Pareto point pair**
instead of a single point per model — the right comparison for an efficiency
claim, and a way to avoid "you beat a smaller/undersized baseline" objections.
(Configs originally also had `s`/`m` presets; dropped to keep the sweep
tractable — `l`/`xl` are used everywhere now.)

**Parallelism**: both runners train multiple (model, preset[, horizon])
combinations **concurrently on the same GPU** (`--max-parallel`, default 4)
rather than sequentially — each combination is a separate process (spawn
context, required for CUDA), and since none of these models alone saturate a
modern GPU, this gives real speedup rather than just time-slicing through a
lock. Seeds are *not* parallelized (kept sequential within each process) —
parallelizing at the (model, preset, horizon) level was simpler and covers
the actual bottleneck (many small/medium training runs), so seed-level
parallelism wasn't worth the added complexity. Safe to kill and resume:
completed combinations are skipped before a process is even spawned for them.

## Setup

```bash
source .venv/bin/activate
uv pip install -e ".[dev]"   # pulls in aeon + pandas (added for this suite)
```

## Download datasets (run once, separately — large, can take a while)

```bash
mkdir -p data

# UCI HAR (~60MB)
curl -L -o data/uci_har.zip https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
unzip -o data/uci_har.zip -d data/

# UCI Electricity Load Diagrams (~250MB)
curl -L -o data/LD2011_2014.zip https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip
unzip -o data/LD2011_2014.zip -d data/
```

ETTh1/ETTh2, Weather, and UCR/UEA datasets auto-download on first run
(small/fast; ETT is a ~2.5MB CSV per series, Weather is two ~1.3MB half-year
zips from MPI Jena, UCR/UEA subset datasets are similarly small).

## Run

### One command, everything

```bash
bash run_timeseries_baselines.sh              # full sweep: 5 models x 2 presets x 4 horizons x 3 seeds
bash run_timeseries_baselines.sh --smoke      # 1 seed/horizon, 3 epochs, lstm/l only — fast pipeline check
bash run_timeseries_baselines.sh --forecast-only   # etth1 + etth2 + electricity + weather only
bash run_timeseries_baselines.sh --classify-only   # har + ucr_uea only
bash run_timeseries_baselines.sh --presets l       # narrow the size sweep further
bash run_timeseries_baselines.sh --pred-lens 96 192   # narrow the horizon sweep (forecasting only)
bash run_timeseries_baselines.sh --max-parallel 8     # more concurrent processes on the GPU
```

**This is a much bigger sweep than a single-point run** — per forecasting
dataset: 5 models × 2 presets × 4 horizons × 3 seeds = 120 training runs, but
most of that is (model, preset, horizon) combinations training **concurrently
on the GPU**, not one-at-a-time — see Parallelism above. Narrow with
`--presets`/`--pred-lens`/`--models` if you need an even faster pass, or raise
`--max-parallel` if you have GPU headroom to spare.

Or run each dataset individually:

### Forecasting (ETTh1 / ETTh2 / Electricity / Weather)

```bash
python benchmarks/timeseries/run_forecasting.py --dataset etth1 \
    --config benchmarks/timeseries/configs/etth1.yaml \
    --results-dir benchmarks/timeseries/results/etth1
```

Flags: `--models`, `--presets` (default: both `l xl`), `--pred-lens`
(default: config's `data.pred_lens`, i.e. `96 192 336 720`), `--seeds`,
`--epochs`, `--max-parallel` (default 4 — concurrent (model, preset, horizon)
processes on the GPU).

### Classification (HAR / UCR-UEA)

```bash
python benchmarks/timeseries/run_classification.py --dataset ucr_uea \
    --config benchmarks/timeseries/configs/ucr_uea.yaml \
    --results-dir benchmarks/timeseries/results/ucr_uea
```

`ucr_uea` loops over every dataset name in `configs/ucr_uea.yaml`'s
`data.datasets` list (default: `ECG200`, `FordA`, `BasicMotions` — a small
univariate + multivariate + size-varied subset). Extend that list to widen
archive coverage, or override per-run with `--ucr-datasets NAME1 NAME2`.

### Quick smoke test (any command above)

Add `--models lstm --presets l --epochs 3` (plus `--pred-lens 96` for
forecasting) to run just one model at one size for a few epochs — confirms
the pipeline runs end-to-end in under a minute before committing to the full
sweep. **Don't point this at your real `--results-dir`** — the runner skips
any `(model, preset, [horizon,] seed)` combination whose `final.json` already
exists, so a quick smoke result left in the real results directory will be
silently treated as "done" and never re-run at full epochs.

## Output files

Same convention as `scripts/gate_experiment.py`:

- Forecasting: `results/<model>/<preset>/h<horizon>/seed<N>/{metrics.jsonl, final.json}`
- Classification: `results/<dataset>/<model>/<preset>/seed<N>/{metrics.jsonl, final.json}`
- `results/summary.json` — mean ± std per (model, preset[, horizon]), sorted by
  param count ascending (Pareto-ready)
  (`ucr_uea` additionally writes one `summary.json` per dataset under
  `results/<dataset_name>/`, plus a combined `results/summary_all.json`)

## What's in here

| File | Contents |
|---|---|
| `data/etth.py` | ETTh1 + ETTh2 loaders (7-channel, multivariate multi-horizon target, canonical 12/4/4-month split) |
| `data/electricity.py` | UCI ECL loader (15-min → hourly resample, 321 earliest-installed clients, multivariate multi-horizon target) |
| `data/weather.py` | MPI Jena Weather loader (21 channels, 10-min resolution, full 2020 year, 70/10/20 split) |
| `data/har.py` | UCI HAR loader (9-channel inertial signals, 6 activity classes) |
| `data/ucr_uea.py` | UCR/UEA archive loader (via `aeon`, any dataset by name) |
| `models/rnn.py` | LSTM / GRU, shared encoder + forecast/classify heads |
| `models/tcn.py` | TCN (Bai et al. 2018) — causal dilated convs, weight-normed, residual blocks |
| `models/patchtst.py` | PatchTST (Nie et al. 2023) — channel-independent patch Transformer |
| `models/mamba.py` | Mamba-style selective SSM (Gu & Dao 2023), pure PyTorch sequential scan — see docstring for what's simplified vs. the paper's hardware kernel |
| `run_forecasting.py` | Trains/evals all 5 models, all presets, all horizons on a forecasting dataset across seeds |
| `run_classification.py` | Same, for classification datasets |
| `configs/*.yaml` | Per-dataset training + per-model size presets, TSR-style sectioned YAML |

## Notes on fidelity

- **Mixed precision (AMP)**: both runners use `torch.amp.autocast` +
  `GradScaler` automatically whenever `device.type == "cuda"` (a no-op on
  CPU). Numerically verified to reproduce identical MSE/MAE/accuracy to the
  pre-AMP fp32 runs on ETTh1 and ECG200. Speeds up LSTM/GRU/TCN/PatchTST on
  any CUDA GPU, and specifically engages Tensor Cores on GPUs that have them
  (Titan RTX, L4, and other Turing/Ampere/Ada-class cards and newer) — this
  is on top of, not instead of, the raw hardware speedup from a faster GPU.
  Does not meaningfully help Mamba, whose bottleneck is sequential kernel-launch
  overhead (see below), not per-step compute precision.
- **Mamba**: implements the paper's selective-SSM recurrence exactly (input-
  dependent A/B/C/delta, zero-order-hold discretization) but as a sequential
  Python loop over time rather than the paper's hardware-aware parallel scan
  (`mamba_ssm` CUDA kernel). Correct, not fast — fine at the sequence lengths
  used here (~100-500 steps) *and* moderate channel counts (ETT/Weather/HAR/
  UCR-UEA). **Confirmed too slow for Electricity's 321 channels** in this
  environment (a 2-epoch, 2-preset smoke test didn't finish in 10+ minutes —
  killed rather than let run indefinitely); LSTM/GRU/TCN/PatchTST all handle
  321 channels fine. If Mamba-on-Electricity numbers are needed for the
  paper, either run it on a GPU with more headroom and patience, or swap in
  the real `mamba_ssm` CUDA kernel for that one baseline.
- **PatchTST forecasting**: per-channel flatten head (`Linear(num_patches*d_model,
  pred_len)`), faithful to the paper — channel-independence preserved (every
  channel patched/encoded by the same shared Transformer, each predicts its
  own future). This head's param count scales with `num_patches` (hence with
  `seq_len`), matching the paper's own reported configs — expected, not a bug.
- **PatchTST classification**: pools over the patch axis (mean) *before* the
  class head, so the head's size depends only on `d_model` — not on
  `num_patches`/`seq_len`. This keeps PatchTST's classification param count
  comparable to the other (seq-len-invariant) baselines across datasets of
  different sequence length (e.g. FordA `seq_len=500` vs ECG200 `seq_len=96`);
  a flatten-based head would make PatchTST alone balloon with no counterpart
  in the other baselines.
- **TCN**: standard reference architecture (Bai et al.), not a novel variant.
- **Size presets**: `l`/`xl` tiers are chosen per-architecture using each
  model's natural scaling knob (`hidden_size` for RNNs, `hidden_channels` for
  TCN, `d_model` for PatchTST/Mamba) — they are **not** hand-tuned to produce
  identical param counts across model types at each tier. That's intentional:
  the Pareto curve (accuracy vs. actual param count) is the comparison that
  matters, not nominal tier labels.

## Param counts (l / xl, per dataset)

Computed directly from each config (no training needed — pure architecture
size). Forecasting counts are at `pred_len=96` (PatchTST's forecast head
scales with horizon; see fidelity note above).

| Dataset | LSTM (l/xl) | GRU (l/xl) | TCN (l/xl) | PatchTST (l/xl) | Mamba (l/xl) |
|---|---|---|---|---|---|
| ETTh1 / ETTh2 (7ch) | 95.6K / 289K | 82.7K / 238K | 133K / 437K | 219K / 734K | 147K / 452K |
| **Electricity (321ch)** | **2.14M / 4.34M** | **2.10M / 4.25M** | **2.17M / 4.49M** | 219K / 734K | 2.13M / 4.38M* |
| Weather (21ch) | 187K / 469K | 173K / 417K | 224K / 617K | 219K / 734K | 235K / 628K |
| HAR (9ch, 6cls) | 52.9K / 204K | 39.8K / 153K | 89.8K / 352K | 157K / 616K | 103K / 367K |
| ECG200 (1ch, 2cls) | 50.6K / 199K | 38.0K / 150K | 87.5K / 347K | 156K / 615K | 103K / 365K |
| FordA (1ch, 2cls) | 50.6K / 199K | 38.0K / 150K | 87.5K / 347K | 159K / 622K | 103K / 365K |
| BasicMotions (6ch, 4cls) | 52.0K / 202K | 39.0K / 152K | 88.9K / 350K | 156K / 615K | 103K / 366K |

\* Mamba-on-Electricity params are correct but the model is currently
untrained here — see the scaling caveat above.

PatchTST's forecasting param count is constant across ETTh1/ETTh2/Electricity/
Weather at a given preset+horizon because its per-channel head depends on
`(num_patches, pred_len)`, not on channel count (channel-independence) —
channel count only changes classification PatchTST slightly, via the
positional embedding (see fidelity note above).

### The 5-20M range: Electricity at longer horizons (no new code needed)

LSTM/GRU/TCN/Mamba's forecast head is `Linear(hidden, pred_len * channels)` —
a flat projection scaling with **horizon × channel count**, not just hidden
size. At Electricity's 321 channels, the horizons we already sweep by default
(`pred_lens: [96, 192, 336, 720]`) push several presets straight into 5-20M:

| Horizon | Preset | LSTM | GRU | TCN | Mamba |
|---|---|---|---|---|---|
| 192 | xl | 8.31M | 8.22M | 8.46M | 8.36M |
| 336 | l | 7.14M | 7.11M | 7.18M | 7.13M |
| 336 | xl | 14.28M | 14.19M | 14.42M | 14.32M |
| 720 | l | 15.16M | 15.12M | 15.19M | 15.15M |

(PatchTST stays at 134K–1.6M across these same cells — its channel-independent
head doesn't scale with channel count at all. This contrast is worth stating
explicitly in the paper: it's *why* channel-independent methods exist, not an
inconsistency in our setup.)

**No new dataset, model, or config change was needed for this** — these rows
come from the sweep already defined in `configs/electricity.yaml`. When
writing up Table 3, report at least one of the `h=336`/`h=720` rows alongside
`h=96` so the 5-20M point is visible, rather than only showing the shortest
horizon.
