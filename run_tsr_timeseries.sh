#!/bin/bash
# Runs TSR (Topological Self-Regulation) on every time-series dataset:
#   Forecasting: ETTh1, ETTh2, Electricity, Weather  (horizons 96/192/336/720)
#   Classification: HAR, UCR/UEA subset
#
# One command. TSR starts from a minimal seed architecture and grows/prunes
# channels and layers during training via phantom sensors. Each dataset (and
# each horizon for forecasting) produces:
#   results/<dataset>/tsr/pred<horizon>/seed42/         — TSR with full plasticity
#   results/<dataset>/tsr_static_final/pred<horizon>/seed42/ — matched-param ablation
#
# Parallelism: multiple (dataset, horizon) pairs train CONCURRENTLY (--parallel
# N, default 3). Each pair is independent and spawns its own process. Safe to
# re-run after killing midway: completed runs are skipped.
#
# Usage:
#   bash run_tsr_timeseries.sh                             # full sweep, 1 seed
#   bash run_tsr_timeseries.sh --smoke                     # 2 epochs, etth1/h96 + har
#   bash run_tsr_timeseries.sh --parallel 6                # more concurrency
#   bash run_tsr_timeseries.sh --forecast-only             # skip classification
#   bash run_tsr_timeseries.sh --classify-only             # skip forecasting
#   bash run_tsr_timeseries.sh --horizons 96 192           # subset of horizons
#   bash run_tsr_timeseries.sh --datasets etth1 weather    # subset of datasets
#   bash run_tsr_timeseries.sh --seeds 42 123              # multi-seed

set -e

SEEDS="42"
DEVICE="auto"
PARALLEL=3
EPOCHS=""
RESULTS_ROOT="benchmarks/timeseries/results"
FORECAST_DATASETS=(etth1 etth2 electricity weather)
# run_tsr_classification.py has no internal loop over UCR/UEA sub-datasets
# (unlike the baseline runner) — "ucr_uea" is not itself a valid --dataset,
# each archive dataset name (from configs/ucr_uea.yaml's data.datasets) must
# be submitted individually. CLASSIFY_DATASETS holds the literal --dataset
# values; UCR_UEA_CONFIG maps the UCR ones back to the shared config file.
CLASSIFY_DATASETS=(har ECG200 FordA BasicMotions)
HORIZONS=(96 192 336 720)
FORECAST=true
CLASSIFY=true
EXTRA_ARGS=()

# Parse flags
for arg in "$@"; do
    case $arg in
        --smoke)
            SEEDS="42"
            EPOCHS="--epochs 2"
            FORECAST_DATASETS=(etth1)
            CLASSIFY_DATASETS=(har)
            HORIZONS=(96)
            PARALLEL=2
            ;;
        --forecast-only) CLASSIFY=false ;;
        --classify-only) FORECAST=false ;;
    esac
done

ARGS=("$@")
for i in "${!ARGS[@]}"; do
    case "${ARGS[$i]}" in
        --parallel)
            PARALLEL="${ARGS[$((i+1))]}"
            ;;
        --seeds)
            SEEDS=""
            j=$((i+1))
            while [[ $j -lt ${#ARGS[@]} && "${ARGS[$j]}" != --* ]]; do
                SEEDS="$SEEDS ${ARGS[$j]}"; j=$((j+1))
            done
            ;;
        --datasets)
            # Reset both lists; the user's list may mix forecast/classify datasets
            FORECAST_DATASETS=()
            CLASSIFY_DATASETS=()
            j=$((i+1))
            while [[ $j -lt ${#ARGS[@]} && "${ARGS[$j]}" != --* ]]; do
                d="${ARGS[$j]}"
                case $d in
                    etth1|etth2|electricity|weather) FORECAST_DATASETS+=("$d") ;;
                    har|ECG200|FordA|BasicMotions) CLASSIFY_DATASETS+=("$d") ;;
                    ucr_uea) CLASSIFY_DATASETS+=(ECG200 FordA BasicMotions) ;;
                    *) echo "Unknown dataset: $d"; exit 1 ;;
                esac
                j=$((j+1))
            done
            ;;
        --horizons)
            HORIZONS=()
            j=$((i+1))
            while [[ $j -lt ${#ARGS[@]} && "${ARGS[$j]}" != --* ]]; do
                HORIZONS+=("${ARGS[$j]}"); j=$((j+1))
            done
            ;;
        --epochs)
            EPOCHS="--epochs ${ARGS[$((i+1))]}"
            ;;
        --skip-static-final)
            EXTRA_ARGS+=("--skip-static-final")
            ;;
        --log-level)
            EXTRA_ARGS+=("--log-level" "${ARGS[$((i+1))]}")
            ;;
    esac
done

source .venv/bin/activate

mkdir -p "$RESULTS_ROOT"

# ── Log directory for parallel runs ──────────────────────────────────────────
LOG_DIR="$RESULTS_ROOT/_logs"
mkdir -p "$LOG_DIR"

echo "========================================================================"
echo "TSR TIME-SERIES BENCHMARK"
if $FORECAST; then
    echo "  Forecasting:  ${FORECAST_DATASETS[*]}"
    echo "  Horizons:     ${HORIZONS[*]}"
fi
if $CLASSIFY; then
    echo "  Classification: ${CLASSIFY_DATASETS[*]}"
fi
echo "  Seeds:        $SEEDS"
echo "  Parallel:     $PARALLEL"
echo "  Results:      $RESULTS_ROOT"
echo "========================================================================"

# ── Concurrency limiter ──────────────────────────────────────────────────────
declare -a PIDS=()
declare -a NAMES=()
RUNNING=0

wait_one() {
    # Wait for any one background job to finish
    for pid_idx in "${!PIDS[@]}"; do
        pid="${PIDS[$pid_idx]}"
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            unset 'PIDS[pid_idx]'
            unset 'NAMES[pid_idx]'
            RUNNING=$((RUNNING - 1))
            return
        fi
    done
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        wait "${PIDS[0]}" 2>/dev/null || true
        unset 'PIDS[0]'
        unset 'NAMES[0]'
        RUNNING=$((RUNNING - 1))
    fi
}

# ── Submit a job ─────────────────────────────────────────────────────────────
submit_forecast() {
    local dataset=$1
    local horizon=$2
    local tag="${dataset}/h${horizon}"
    local logfile="$LOG_DIR/tsr_ts_${dataset}_h${horizon}.log"

    echo "[START] $tag (log: $logfile)"
    python benchmarks/timeseries/run_tsr_forecasting.py \
        --dataset "$dataset" \
        --config "benchmarks/timeseries/configs/$dataset.yaml" \
        --horizons "$horizon" \
        --seeds $SEEDS \
        --results-dir "$RESULTS_ROOT/$dataset" \
        --device "$DEVICE" \
        --max-parallel 1 \
        $EPOCHS \
        "${EXTRA_ARGS[@]}" \
        > "$logfile" 2>&1 &
    PIDS+=($!)
    NAMES+=("$tag")
    RUNNING=$((RUNNING + 1))
}

submit_classify() {
    local dataset=$1
    local tag="$dataset"
    local logfile="$LOG_DIR/tsr_ts_${dataset}.log"
    local config="benchmarks/timeseries/configs/$dataset.yaml"
    local results_dir="$RESULTS_ROOT/$dataset"

    # ECG200/FordA/BasicMotions share configs/ucr_uea.yaml (no per-dataset
    # config file exists) and live under results/ucr_uea/<name>, matching
    # the baseline runner's layout.
    case $dataset in
        ECG200|FordA|BasicMotions)
            config="benchmarks/timeseries/configs/ucr_uea.yaml"
            results_dir="$RESULTS_ROOT/ucr_uea/$dataset"
            ;;
    esac

    echo "[START] $tag (log: $logfile)"
    python benchmarks/timeseries/run_tsr_classification.py \
        --dataset "$dataset" \
        --config "$config" \
        --seeds $SEEDS \
        --results-dir "$results_dir" \
        --device "$DEVICE" \
        --max-parallel 1 \
        $EPOCHS \
        "${EXTRA_ARGS[@]}" \
        > "$logfile" 2>&1 &
    PIDS+=($!)
    NAMES+=("$tag")
    RUNNING=$((RUNNING + 1))
}

# ── Submit all jobs with concurrency control ─────────────────────────────────
if $FORECAST; then
    for dataset in "${FORECAST_DATASETS[@]}"; do
        for horizon in "${HORIZONS[@]}"; do
            if [[ $RUNNING -ge $PARALLEL ]]; then
                wait_one
            fi
            submit_forecast "$dataset" "$horizon"
        done
    done
fi

if $CLASSIFY; then
    for dataset in "${CLASSIFY_DATASETS[@]}"; do
        if [[ $RUNNING -ge $PARALLEL ]]; then
            wait_one
        fi
        submit_classify "$dataset"
    done
fi

# Wait for all remaining jobs
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

# ── Collect and print results ────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "TSR TIME-SERIES SUMMARY"
echo "========================================================================"

if $FORECAST; then
    printf "%-14s %-8s %-10s %-14s %-14s" "DATASET" "HORIZON" "PARAMS" "TEST_MSE" "TEST_MAE"
    echo ""
    echo "------------------------------------------------------------------------"
    for dataset in "${FORECAST_DATASETS[@]}"; do
        for horizon in "${HORIZONS[@]}"; do
            final="$RESULTS_ROOT/$dataset/tsr/pred${horizon}/seed42/final.json"
            if [[ -f "$final" ]]; then
                read_params=$(python3 -c "
import json
d=json.load(open('$final'))
print(f\"{d['params']:,}\")" 2>/dev/null || echo "N/A")
                read_metrics=$(python3 -c "
import json
d=json.load(open('$final'))
mse=d.get('test_mse', 'N/A')
mae=d.get('test_mae', 'N/A')
print(f\"{mse:.4f} {mae:.4f}\" if isinstance(mse, float) else f'{mse} {mae}')" 2>/dev/null || echo "N/A N/A")
                mse=$(echo "$read_metrics" | cut -d' ' -f1)
                mae=$(echo "$read_metrics" | cut -d' ' -f2)
                printf "%-14s %-8s %-10s %-14s %-14s" "$dataset" "$horizon" "$read_params" "$mse" "$mae"
                echo ""
            else
                printf "%-14s %-8s %-10s %-14s %-14s" "$dataset" "$horizon" "—" "(no results)" ""
                echo ""
            fi
        done
    done
    echo ""
fi

if $CLASSIFY; then
    printf "%-14s %-10s %-14s" "DATASET" "PARAMS" "TEST_ACC"
    echo ""
    echo "------------------------------------------------------------------------"
    for dataset in "${CLASSIFY_DATASETS[@]}"; do
        results_dir="$RESULTS_ROOT/$dataset"
        case $dataset in
            ECG200|FordA|BasicMotions) results_dir="$RESULTS_ROOT/ucr_uea/$dataset" ;;
        esac
        final="$results_dir/tsr/default/seed42/final.json"
        if [[ -f "$final" ]]; then
            read_params=$(python3 -c "
import json
d=json.load(open('$final'))
print(f\"{d['params']:,}\")" 2>/dev/null || echo "N/A")
            read_metric=$(python3 -c "
import json
d=json.load(open('$final'))
v=d.get('test_acc', 'N/A')
print(f'{v:.4f}' if isinstance(v, float) else str(v))" 2>/dev/null || echo "N/A")
            printf "%-14s %-10s %-14s" "$dataset" "$read_params" "$read_metric"
            echo ""
        else
            printf "%-14s %-10s %-14s" "$dataset" "—" "(no results)"
            echo ""
        fi
    done
fi

echo "========================================================================"
echo ""
echo "Per-dataset details: $RESULTS_ROOT/<dataset>/tsr_*_summary.json"
echo ""

# ── Print any errors ─────────────────────────────────────────────────────────
ERRORS=0
for logfile in "$LOG_DIR"/tsr_ts_*.log; do
    [[ -f "$logfile" ]] || continue
    if grep -qi "error\|traceback\|FAILED" "$logfile" 2>/dev/null; then
        name=$(basename "$logfile" .log | sed 's/^tsr_ts_//')
        echo "[ERROR] $name — see $logfile"
        ERRORS=$((ERRORS + 1))
    fi
done
if [[ $ERRORS -gt 0 ]]; then
    echo ""
    echo "$ERRORS run(s) had errors. Check logs in $LOG_DIR/"
fi
