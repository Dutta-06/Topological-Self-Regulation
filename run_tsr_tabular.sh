#!/bin/bash
# Runs TSR (Topological Self-Regulation) on every tabular dataset: Adult Income,
# California Housing, Forest Cover Type, HIGGS, Fashion-MNIST. One command.
#
# TSR starts from a minimal seed architecture and grows/prunes neurons and
# layers during training via phantom sensors. Each dataset produces:
#   results/<dataset>/tsr/default/seed42/        — TSR with full plasticity
#   results/<dataset>/tsr_static_final/default/seed42/ — matched-param ablation
#
# Parallelism: multiple datasets train CONCURRENTLY (--parallel N, default 2).
# Each dataset spawns its own process; within a dataset, TSR and static_final
# run sequentially (static_final needs TSR's discovered topology). Safe to
# re-run after killing midway: completed runs are skipped.
#
# Usage:
#   bash run_tsr_tabular.sh                        # full sweep, 1 seed
#   bash run_tsr_tabular.sh --smoke                # 1 seed, 2 epochs, adult only
#   bash run_tsr_tabular.sh --parallel 4           # all 5 datasets at once
#   bash run_tsr_tabular.sh --datasets adult higgs # subset of datasets
#   bash run_tsr_tabular.sh --seeds 42 123         # multi-seed
#   bash run_tsr_tabular.sh --skip-static-final    # TSR only, no ablation

set -e

SEEDS="42"
DEVICE="auto"
PARALLEL=2
EPOCHS=""
RESULTS_ROOT="benchmarks/tabular/results"
ALL_DATASETS=(adult california_housing covertype higgs fashion_mnist)
DATASETS=("${ALL_DATASETS[@]}")
EXTRA_ARGS=()

# Parse flags
for arg in "$@"; do
    case $arg in
        --smoke)
            SEEDS="42"
            EPOCHS="--epochs 2"
            DATASETS=(adult)
            PARALLEL=1
            ;;
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
            DATASETS=()
            j=$((i+1))
            while [[ $j -lt ${#ARGS[@]} && "${ARGS[$j]}" != --* ]]; do
                DATASETS+=("${ARGS[$j]}"); j=$((j+1))
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
echo "TSR TABULAR BENCHMARK"
echo "  Datasets:  ${DATASETS[*]}"
echo "  Seeds:     $SEEDS"
echo "  Parallel:  $PARALLEL"
echo "  Results:   $RESULTS_ROOT"
echo "========================================================================"

# ── Run with concurrency limiter ─────────────────────────────────────────────
PIDS=()
NAMES=()
RUNNING=0

wait_one() {
    # Wait for any one background job to finish
    for pid_idx in "${!PIDS[@]}"; do
        pid="${PIDS[$pid_idx]}"
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" || true
            unset 'PIDS[pid_idx]'
            unset 'NAMES[pid_idx]'
            RUNNING=$((RUNNING - 1))
            return
        fi
    done
    # If all are still running, wait for the first one
    if [[ ${#PIDS[@]} -gt 0 ]]; then
        wait "${PIDS[0]}" || true
        unset 'PIDS[0]'
        unset 'NAMES[0]'
        RUNNING=$((RUNNING - 1))
    fi
}

run_dataset() {
    local dataset=$1
    local logfile="$LOG_DIR/tsr_tabular_${dataset}.log"

    echo "[START] $dataset (log: $logfile)"
    python benchmarks/tabular/run_tsr_tabular.py \
        --dataset "$dataset" \
        --config "benchmarks/tabular/configs/$dataset.yaml" \
        --seeds $SEEDS \
        --results-dir "$RESULTS_ROOT/$dataset" \
        --device "$DEVICE" \
        --max-parallel 1 \
        $EPOCHS \
        "${EXTRA_ARGS[@]}" \
        > "$logfile" 2>&1 &
    PIDS+=($!)
    NAMES+=("$dataset")
    RUNNING=$((RUNNING + 1))
}

for dataset in "${DATASETS[@]}"; do
    if [[ $RUNNING -ge $PARALLEL ]]; then
        wait_one
    fi
    run_dataset "$dataset"
done

# Wait for remaining jobs
for pid in "${PIDS[@]}"; do
    wait "$pid" || true
done

# ── Collect and print results ────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo "TSR TABULAR SUMMARY"
echo "========================================================================"
printf "%-22s %-10s %-14s" "DATASET" "PARAMS" "TEST_METRIC"
echo ""
echo "------------------------------------------------------------------------"

for dataset in "${DATASETS[@]}"; do
    # Find the TSR final.json (try default preset first, then any)
    final="$RESULTS_ROOT/$dataset/tsr/default/seed42/final.json"
    if [[ -f "$final" ]]; then
        params=$(python3 -c "import json; d=json.load(open('$final')); print(f\"{d['params']:,}\")" 2>/dev/null || echo "N/A")
        metric=$(python3 -c "import json; d=json.load(open('$final')); v=d.get('test_acc', d.get('test_mse', d.get('test_mae', 'N/A'))); print(f'{v:.4f}' if isinstance(v, float) else v)" 2>/dev/null || echo "N/A")
        printf "%-22s %-10s %-14s" "$dataset" "$params" "$metric"
        echo ""
    else
        printf "%-22s %-10s %-14s" "$dataset" "—" "(no results)"
        echo ""
    fi
done

echo "========================================================================"
echo ""
echo "Per-dataset details: $RESULTS_ROOT/<dataset>/tsr_summary.json"
echo "Static-final ablation: $RESULTS_ROOT/<dataset>/tsr_static_final/"
echo ""

# ── Print any errors ─────────────────────────────────────────────────────────
ERRORS=0
for dataset in "${DATASETS[@]}"; do
    logfile="$LOG_DIR/tsr_tabular_${dataset}.log"
    if [[ -f "$logfile" ]] && grep -qi "error\|traceback\|FAILED" "$logfile" 2>/dev/null; then
        echo "[ERROR] $dataset — see $logfile"
        ERRORS=$((ERRORS + 1))
    fi
done
if [[ $ERRORS -gt 0 ]]; then
    echo ""
    echo "$ERRORS dataset(s) had errors. Check logs in $LOG_DIR/"
fi
