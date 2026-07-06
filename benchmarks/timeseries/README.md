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
papers' unpublished exact client list). At this scale, `s`→`xl` presets span
**~535K → ~4.4M params** — genuinely large models, because 321 channels
actually need it, not because anything was padded. Weather completes the
standard 4-dataset LTSF-forecasting subset we cover (of the canonical 8:
ETTh1/ETTh2/ETTm1/ETTm2/Electricity/Traffic/Weather/Exchange — we don't yet
have ETTm1/ETTm2/Traffic/Exchange).

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

**Every model runs at 4 size presets** (`s`/`m`/`l`/`xl`, see each config's
`models.<name>.presets`) rather than one arbitrary size. This turns each
(dataset, horizon) cell into an **accuracy-vs-params Pareto curve** instead of
a single point per model — the right comparison for an efficiency claim, and
the only way to avoid "you beat a smaller/undersized baseline" objections.

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
bash run_timeseries_baselines.sh              # full sweep: 5 models x 4 presets x 4 horizons x 3 seeds
bash run_timeseries_baselines.sh --smoke      # 1 seed/preset/horizon, 3 epochs, lstm only — fast pipeline check
bash run_timeseries_baselines.sh --forecast-only   # etth1 + etth2 + electricity + weather only
bash run_timeseries_baselines.sh --classify-only   # har + ucr_uea only
bash run_timeseries_baselines.sh --presets l xl    # narrow the size sweep
bash run_timeseries_baselines.sh --pred-lens 96 192   # narrow the horizon sweep (forecasting only)
```

**This is a much bigger sweep than a single-point run** — per forecasting
dataset: 5 models × 4 presets × 4 horizons × 3 seeds = 240 training runs.
Narrow with `--presets`/`--pred-lens`/`--models` if you need a faster pass.

Or run each dataset individually:

### Forecasting (ETTh1 / ETTh2 / Electricity / Weather)

```bash
python benchmarks/timeseries/run_forecasting.py --dataset etth1 \
    --config benchmarks/timeseries/configs/etth1.yaml \
    --results-dir benchmarks/timeseries/results/etth1
```

Flags: `--models`, `--presets` (default: all of `s m l xl`), `--pred-lens`
(default: config's `data.pred_lens`, i.e. `96 192 336 720`), `--seeds`, `--epochs`.

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

Add `--models lstm --presets s --epochs 3` (plus `--pred-lens 96` for
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
- **Size presets**: `s`/`m`/`l`/`xl` tiers are chosen per-architecture using
  each model's natural scaling knob (`hidden_size` for RNNs, `hidden_channels`
  for TCN, `d_model` for PatchTST/Mamba) — they are **not** hand-tuned to
  produce identical param counts across model types at each tier. That's
  intentional: the Pareto curve (accuracy vs. actual param count) is the
  comparison that matters, not nominal tier labels.

## Param counts (s → xl range, per dataset)

Computed directly from each config (no training needed — pure architecture
size). Forecasting counts are at `pred_len=96` (PatchTST's forecast head
scales with horizon; see fidelity note above).

| Dataset | LSTM | GRU | TCN | PatchTST | Mamba |
|---|---|---|---|---|---|
| ETTh1 / ETTh2 (7ch) | 13K – 289K | 13K – 238K | 16K – 437K | 24K – 734K | 17K – 452K |
| **Electricity (321ch)** | **546K – 4.34M** | **540K – 4.25M** | **548K – 4.49M** | 24K – 734K | 535K – 4.38M* |
| Weather (21ch) | 37K – 469K | 36K – 417K | 40K – 617K | 24K – 734K | 40K – 628K |
| HAR (9ch, 6cls) | 1.8K – 204K | 1.4K – 153K | 4.7K – 352K | 7.4K – 616K | 5.8K – 367K |
| ECG200 (1ch, 2cls) | 1.3K – 199K | 0.9K – 150K | 4.1K – 347K | 7.3K – 615K | 5.6K – 365K |
| FordA (1ch, 2cls) | 1.3K – 199K | 0.9K – 150K | 4.1K – 347K | 8.1K – 622K | 5.6K – 365K |
| BasicMotions (6ch, 4cls) | 1.6K – 202K | 1.2K – 152K | 4.5K – 350K | 7.3K – 615K | 5.7K – 366K |

\* Mamba-on-Electricity params are correct but the model is currently
untrained here — see the scaling caveat above.

PatchTST's forecasting param count is constant across ETTh1/ETTh2/Electricity/
Weather at a given preset+horizon because its per-channel head depends on
`(num_patches, pred_len)`, not on channel count (channel-independence) —
channel count only changes classification PatchTST slightly, via the
positional embedding (see fidelity note above).
