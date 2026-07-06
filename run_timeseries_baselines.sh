#!/bin/bash
# Runs the FULL Table 3 baseline suite: LSTM, GRU, TCN, PatchTST, Mamba on
# every time-series dataset — ETTh1, ETTh2, Electricity, Weather (forecasting),
# HAR, UCR/UEA subset (classification). One command, everything.
#
# Protocol: standard multivariate long-horizon forecasting (all channels,
# horizons {96,192,336,720}) and 4 size presets per model (s/m/l/xl) so
# results form an accuracy-vs-params Pareto curve rather than one arbitrary
# size per model. This is a MUCH bigger sweep than a single-point comparison —
# per forecasting dataset: 5 models x 4 presets x 4 horizons x 3 seeds = 240 runs.
# Electricity uses the standard 321-client scale (not a small subset) and
# Weather has 21 channels — both genuinely need more capacity than ETT/HAR,
# not padding.
#
# TSR itself is intentionally excluded (see benchmarks/timeseries/README.md).
#
# Usage:
#   bash run_timeseries_baselines.sh              # full sweep, 3 seeds each
#   bash run_timeseries_baselines.sh --smoke      # 1 seed/preset/horizon, 3 epochs, lstm only
#   bash run_timeseries_baselines.sh --forecast-only
#   bash run_timeseries_baselines.sh --classify-only
#   bash run_timeseries_baselines.sh --presets l xl   # narrow the size sweep
#   bash run_timeseries_baselines.sh --pred-lens 96 192  # narrow the horizon sweep

set -e

SEEDS="42 123 456"
DEVICE="auto"
RESULTS_ROOT="benchmarks/timeseries/results"

COMMON_ARGS=()
PRED_LENS_ARGS=()

for arg in "$@"; do
    case $arg in
        --smoke)
            SEEDS="42"
            COMMON_ARGS+=(--models lstm --presets s --epochs 3)
            PRED_LENS_ARGS+=(--pred-lens 96)
            ;;
        --forecast-only) CLASSIFY=false ;;
        --classify-only) FORECAST=false ;;
    esac
done
FORECAST=${FORECAST:-true}
CLASSIFY=${CLASSIFY:-true}

# Pass through --presets / --pred-lens if the user supplied them explicitly
# (beyond --smoke, which already sets its own narrow defaults above).
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
