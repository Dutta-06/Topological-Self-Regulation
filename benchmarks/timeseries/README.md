# Time-Series Benchmarks (Table 3)

Baseline suite for the paper's Table 3 — forecasting and classification
competitors. **TSR itself is intentionally out of scope here** (no residual
connections, no phantom growth) — this is purely the competitor infrastructure:
LSTM, GRU, TCN, PatchTST, and a Mamba-style selective-SSM, run under matched
config/training conditions across 5 datasets.

| Task | Datasets |
|---|---|
| Forecasting | ETTh1, ETTh2, UCI Electricity Load (ECL) |
| Classification | UCI HAR, UCR/UEA archive (subset) |

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

ETTh1/ETTh2 and UCR/UEA datasets auto-download on first run (small/fast; ETT
is a ~2.5MB CSV per series, UCR/UEA subset datasets are similarly small).

## Run

### One command, everything

```bash
bash run_timeseries_baselines.sh              # full: all 5 models, 3 seeds, all 5 datasets
bash run_timeseries_baselines.sh --smoke      # 1 seed, 3 epochs, lstm only — fast pipeline check
bash run_timeseries_baselines.sh --forecast-only   # etth1 + etth2 + electricity only
bash run_timeseries_baselines.sh --classify-only   # har + ucr_uea only
```

Or run each dataset individually:

### Forecasting (ETTh1 / ETTh2 / Electricity)

```bash
python benchmarks/timeseries/run_forecasting.py --dataset etth1 \
    --config benchmarks/timeseries/configs/etth1.yaml \
    --results-dir benchmarks/timeseries/results/etth1

python benchmarks/timeseries/run_forecasting.py --dataset etth2 \
    --config benchmarks/timeseries/configs/etth2.yaml \
    --results-dir benchmarks/timeseries/results/etth2

python benchmarks/timeseries/run_forecasting.py --dataset electricity \
    --config benchmarks/timeseries/configs/electricity.yaml \
    --results-dir benchmarks/timeseries/results/electricity
```

### Classification (HAR / UCR-UEA)

```bash
python benchmarks/timeseries/run_classification.py --dataset har \
    --config benchmarks/timeseries/configs/har.yaml \
    --results-dir benchmarks/timeseries/results/har

python benchmarks/timeseries/run_classification.py --dataset ucr_uea \
    --config benchmarks/timeseries/configs/ucr_uea.yaml \
    --results-dir benchmarks/timeseries/results/ucr_uea
```

`ucr_uea` loops over every dataset name in `configs/ucr_uea.yaml`'s
`data.datasets` list (default: `ECG200`, `FordA`, `BasicMotions` — a small
univariate + multivariate + size-varied subset). Extend that list to widen
archive coverage, or override per-run with `--ucr-datasets NAME1 NAME2`.

### Quick smoke test (any command above)

Add `--models lstm --epochs 3` to run just one model for a few epochs —
confirms the pipeline runs end-to-end in under a minute before committing to
the full sweep.

## Output files

Same convention as `scripts/gate_experiment.py`:

- `results/<model>/seed<N>/metrics.jsonl` — per-epoch train/val metrics
- `results/<model>/seed<N>/final.json` — best val + test metric, param count
- `results/summary.json` — mean ± std per model across seeds
  (`ucr_uea` additionally writes one `summary.json` per dataset under
  `results/<dataset_name>/`, plus a combined `results/summary_all.json`)

## What's in here

| File | Contents |
|---|---|
| `data/etth.py` | ETTh1 + ETTh2 loaders (7-channel, OT target, canonical 12/4/4-month split) |
| `data/electricity.py` | UCI ECL loader (15-min → hourly resample, configurable client count) |
| `data/har.py` | UCI HAR loader (9-channel inertial signals, 6 activity classes) |
| `data/ucr_uea.py` | UCR/UEA archive loader (via `aeon`, any dataset by name) |
| `models/rnn.py` | LSTM / GRU, shared encoder + forecast/classify heads |
| `models/tcn.py` | TCN (Bai et al. 2018) — causal dilated convs, weight-normed, residual blocks |
| `models/patchtst.py` | PatchTST (Nie et al. 2023) — channel-independent patch Transformer |
| `models/mamba.py` | Mamba-style selective SSM (Gu & Dao 2023), pure PyTorch sequential scan — see docstring for what's simplified vs. the paper's hardware kernel |
| `run_forecasting.py` | Trains/evals all 5 models on a forecasting dataset across seeds |
| `run_classification.py` | Same, for classification datasets |
| `configs/*.yaml` | Per-dataset training + per-model hyperparameters, TSR-style sectioned YAML |

## Notes on fidelity

- **Mamba**: implements the paper's selective-SSM recurrence exactly (input-
  dependent A/B/C/delta, zero-order-hold discretization) but as a sequential
  Python loop over time rather than the paper's hardware-aware parallel scan
  (`mamba_ssm` CUDA kernel). Correct, not fast — fine at the sequence lengths
  used here (~100-500 steps); would not scale to the paper's long-context
  regime without the real kernel.
- **PatchTST**: forecasting adapts to this repo's single-scalar-target
  convention (window → one future value of the last/target channel) rather
  than the paper's full multi-step multivariate output. Channel-independence
  is preserved — every channel is patched and encoded by the same shared
  Transformer; only the target channel's head output is read for the loss.
- **TCN**: standard reference architecture (Bai et al.), not a novel variant.
