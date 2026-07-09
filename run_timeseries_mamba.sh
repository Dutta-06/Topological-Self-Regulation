#!/bin/bash
# Runs ONLY the Mamba baseline, separately from run_timeseries_baselines.sh.
#
# Mamba's selective-SSM forward pass is a sequential Python loop over
# timesteps (see benchmarks/timeseries/models/mamba.py's docstring) — far
# slower than LSTM/GRU/TCN/PatchTST, confirmed to not finish a 2-epoch/
# 2-preset smoke test on Electricity's 321 channels in 10+ minutes. Keeping
# it in its own script means the fast-model sweep isn't stuck waiting on it
# to print final results, and you can run this one separately (overnight,
# lower priority, fewer epochs, whatever fits) without touching the other 4.
#
# Early stopping is ON by default here (patience 10 epochs, no val
# improvement) — unlike run_timeseries_baselines.sh, which always trains the
# other 4 models for the full fixed 100 epochs. Given how slow Mamba is,
# there's no reason to keep training past convergence; the 4 fast baselines
# don't need this since they finish the full budget quickly anyway.
#
# Same flags as run_timeseries_baselines.sh (this is the same script with
# --models fixed to mamba and no --models override):
#
# Usage:
#   bash run_timeseries_mamba.sh              # full sweep, 1 seed, early stopping (patience 10)
#   bash run_timeseries_mamba.sh --smoke      # 1 horizon, 3 epochs, l preset only
#   bash run_timeseries_mamba.sh --forecast-only
#   bash run_timeseries_mamba.sh --classify-only
#   bash run_timeseries_mamba.sh --presets l      # narrow the size sweep further
#   bash run_timeseries_mamba.sh --pred-lens 96 192  # narrow the horizon sweep
#   bash run_timeseries_mamba.sh --max-parallel 2    # concurrent (preset, horizon) processes
#   bash run_timeseries_mamba.sh --early-stopping-patience 20  # override the default patience
#   bash run_timeseries_mamba.sh --early-stopping-patience 0   # disable early stopping

set -e

SEEDS="42"
DEVICE="auto"
RESULTS_ROOT="benchmarks/timeseries/results"
MODELS_ARGS=(--models mamba)
EARLY_STOPPING_PATIENCE=10

COMMON_ARGS=()
PRED_LENS_ARGS=()

for arg in "$@"; do
    case $arg in
        --smoke)
            SEEDS="42"
            COMMON_ARGS+=(--presets l --epochs 3)
            PRED_LENS_ARGS+=(--pred-lens 96)
            ;;
        --forecast-only) CLASSIFY=false ;;
        --classify-only) FORECAST=false ;;
    esac
done
FORECAST=${FORECAST:-true}
CLASSIFY=${CLASSIFY:-true}

# Pass through --presets / --pred-lens / --max-parallel / --early-stopping-patience
# if the user supplied them explicitly (beyond --smoke, which already sets
# its own narrow defaults).
ARGS=("$@")
for i in "${!ARGS[@]}"; do
    if [[ "${ARGS[$i]}" == "--presets" ]]; then
        COMMON_ARGS+=(--presets)
        j=$((i+1))
        while [[ $j -lt ${#ARGS[@]} && "${ARGS[$j]}" != --* ]]; do
            COMMON_ARGS+=("${ARGS[$j]}"); j=$((j+1))
        done
    fi
    if [[ "${ARGS[$i]}" == "--pred-lens" ]]; then
        PRED_LENS_ARGS=(--pred-lens)
        j=$((i+1))
        while [[ $j -lt ${#ARGS[@]} && "${ARGS[$j]}" != --* ]]; do
            PRED_LENS_ARGS+=("${ARGS[$j]}"); j=$((j+1))
        done
    fi
    if [[ "${ARGS[$i]}" == "--max-parallel" ]]; then
        COMMON_ARGS+=(--max-parallel "${ARGS[$((i+1))]}")
    fi
    if [[ "${ARGS[$i]}" == "--early-stopping-patience" ]]; then
        EARLY_STOPPING_PATIENCE="${ARGS[$((i+1))]}"
    fi
done
COMMON_ARGS+=(--early-stopping-patience "$EARLY_STOPPING_PATIENCE")

source .venv/bin/activate

run_forecast() {
    local dataset=$1
    echo ""
    echo "========================================================"
    echo "MAMBA FORECASTING — $dataset"
    echo "Results -> $RESULTS_ROOT/$dataset/"
    echo "========================================================"
    python benchmarks/timeseries/run_forecasting.py \
        --dataset "$dataset" \
        --config "benchmarks/timeseries/configs/$dataset.yaml" \
        --seeds $SEEDS \
        --results-dir "$RESULTS_ROOT/$dataset" \
        --device "$DEVICE" \
        "${MODELS_ARGS[@]}" "${COMMON_ARGS[@]}" "${PRED_LENS_ARGS[@]}"
}

run_classify() {
    local dataset=$1
    echo ""
    echo "========================================================"
    echo "MAMBA CLASSIFICATION — $dataset"
    echo "Results -> $RESULTS_ROOT/$dataset/"
    echo "========================================================"
    python benchmarks/timeseries/run_classification.py \
        --dataset "$dataset" \
        --config "benchmarks/timeseries/configs/$dataset.yaml" \
        --seeds $SEEDS \
        --results-dir "$RESULTS_ROOT/$dataset" \
        --device "$DEVICE" \
        "${MODELS_ARGS[@]}" "${COMMON_ARGS[@]}"
}

if $FORECAST; then
    run_forecast etth1
    run_forecast etth2
    run_forecast electricity
    run_forecast weather
fi

if $CLASSIFY; then
    run_classify har
    run_classify ucr_uea
fi

echo ""
echo "Done. Per-dataset summaries in $RESULTS_ROOT/<dataset>/summary.json"
echo "(ucr_uea also writes $RESULTS_ROOT/ucr_uea/summary_all.json across its dataset subset)"
echo "NOTE: this summary.json/summary_all.json only reflects Mamba's own runs —"
echo "merge with the other 4 models' results (from run_timeseries_baselines.sh)"
echo "for the full Table 3 comparison."
