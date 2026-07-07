#!/bin/bash
# Runs the FULL Table 3 baseline suite: LSTM, GRU, TCN, PatchTST, Mamba on
# every time-series dataset — ETTh1, ETTh2, Electricity, Weather (forecasting),
# HAR, UCR/UEA subset (classification). One command, everything.
#
# Protocol: standard multivariate long-horizon forecasting (all channels,
# horizons {96,192,336,720}) and 2 size presets per model (l/xl) so results
# form an accuracy-vs-params Pareto curve rather than one arbitrary size per
# model. Per forecasting dataset: 5 models x 2 presets x 4 horizons x 1 seed
# = 40 runs. NOTE: single-seed by default — no variance estimate, results are
# point estimates only. Electricity uses the standard 321-client scale (not a
# small subset) and Weather has 21 channels — both genuinely need more
# capacity than ETT/HAR, not padding.
#
# Parallelism: both runners train multiple (model, preset[, horizon])
# combinations CONCURRENTLY on the same GPU (--max-parallel, default 4) —
# separate processes, not separate seeds, since none of these models alone
# saturate a modern GPU. See benchmarks/timeseries/run_forecasting.py's
# docstring for details. Safe to re-run after killing midway: completed
# combinations are skipped before any process is even spawned for them.
#
# TSR itself is intentionally excluded (see benchmarks/timeseries/README.md).
#
# Usage:
#   bash run_timeseries_baselines.sh              # full sweep, 1 seed
#   bash run_timeseries_baselines.sh --smoke      # 1 seed/horizon, 3 epochs, lstm/l only
#   bash run_timeseries_baselines.sh --forecast-only
#   bash run_timeseries_baselines.sh --classify-only
#   bash run_timeseries_baselines.sh --presets l      # narrow the size sweep further
#   bash run_timeseries_baselines.sh --pred-lens 96 192  # narrow the horizon sweep
#   bash run_timeseries_baselines.sh --max-parallel 8    # more concurrent processes on the GPU

set -e

SEEDS="42"
DEVICE="auto"
RESULTS_ROOT="benchmarks/timeseries/results"

COMMON_ARGS=()
PRED_LENS_ARGS=()

for arg in "$@"; do
    case $arg in
        --smoke)
            SEEDS="42"
            COMMON_ARGS+=(--models lstm --presets l --epochs 3)
            PRED_LENS_ARGS+=(--pred-lens 96)
            ;;
        --forecast-only) CLASSIFY=false ;;
        --classify-only) FORECAST=false ;;
    esac
done
FORECAST=${FORECAST:-true}
CLASSIFY=${CLASSIFY:-true}

# Pass through --presets / --pred-lens / --max-parallel if the user supplied
# them explicitly (beyond --smoke, which already sets its own narrow defaults).
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
done

source .venv/bin/activate

run_forecast() {
    local dataset=$1
    echo ""
    echo "========================================================"
    echo "FORECASTING — $dataset"
    echo "Results -> $RESULTS_ROOT/$dataset/"
    echo "========================================================"
    python benchmarks/timeseries/run_forecasting.py \
        --dataset "$dataset" \
        --config "benchmarks/timeseries/configs/$dataset.yaml" \
        --seeds $SEEDS \
        --results-dir "$RESULTS_ROOT/$dataset" \
        --device "$DEVICE" \
        "${COMMON_ARGS[@]}" "${PRED_LENS_ARGS[@]}"
}

run_classify() {
    local dataset=$1
    echo ""
    echo "========================================================"
    echo "CLASSIFICATION — $dataset"
    echo "Results -> $RESULTS_ROOT/$dataset/"
    echo "========================================================"
    python benchmarks/timeseries/run_classification.py \
        --dataset "$dataset" \
        --config "benchmarks/timeseries/configs/$dataset.yaml" \
        --seeds $SEEDS \
        --results-dir "$RESULTS_ROOT/$dataset" \
        --device "$DEVICE" \
        "${COMMON_ARGS[@]}"
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
