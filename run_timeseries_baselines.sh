#!/bin/bash
# Runs the FULL Table 3 baseline suite: LSTM, GRU, TCN, PatchTST, Mamba on
# every time-series dataset — ETTh1, ETTh2, Electricity (forecasting), HAR,
# UCR/UEA subset (classification). One command, everything.
#
# TSR itself is intentionally excluded (see benchmarks/timeseries/README.md).
#
# Usage:
#   bash run_timeseries_baselines.sh              # full run, 3 seeds each
#   bash run_timeseries_baselines.sh --smoke      # 1 seed, 3 epochs, 1 model (fast check)
#   bash run_timeseries_baselines.sh --forecast-only
#   bash run_timeseries_baselines.sh --classify-only

set -e

SEEDS="42 123 456"
DEVICE="auto"
RESULTS_ROOT="benchmarks/timeseries/results"

SMOKE=false
FORECAST=true
CLASSIFY=true
EXTRA_ARGS=()

for arg in "$@"; do
    case $arg in
        --smoke)          SMOKE=true; SEEDS="42"; EXTRA_ARGS+=(--models lstm --epochs 3) ;;
        --forecast-only)  CLASSIFY=false ;;
        --classify-only)  FORECAST=false ;;
    esac
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
        "${EXTRA_ARGS[@]}"
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
        "${EXTRA_ARGS[@]}"
}

if $FORECAST; then
    run_forecast etth1
    run_forecast etth2
    run_forecast electricity
fi

if $CLASSIFY; then
    run_classify har
    run_classify ucr_uea
fi

echo ""
echo "Done. Per-dataset summaries in $RESULTS_ROOT/<dataset>/summary.json"
echo "(ucr_uea also writes $RESULTS_ROOT/ucr_uea/summary_all.json across its dataset subset)"
