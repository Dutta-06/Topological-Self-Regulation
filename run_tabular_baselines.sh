#!/bin/bash
# Runs the Table 1 tabular baseline suite — MLP, ResNet-MLP, FT-Transformer,
# TabNet, SAINT — on every tabular dataset: Adult Income, California Housing,
# Forest Cover Type, HIGGS. One command.
#
# Protocol: MLP is a single fixed-size reference (deliberately not swept —
# see benchmarks/tabular/README.md). The other 4 models sweep 2 size presets
# (l/xl) on Covertype/Higgs, which are large enough to need real capacity and
# form an accuracy-vs-params Pareto curve; on Adult/California Housing (small
# datasets, no need for a sweep) they use a single fixed preset too. Single
# seed by default — no variance estimate, results are point estimates only.
#
# Parallelism: (model, preset) combinations train CONCURRENTLY on the same
# GPU (--max-parallel, default 4) — separate processes, since none of these
# models alone saturate a modern GPU. Safe to re-run after killing midway:
# completed combinations are skipped before any process is even spawned.
#
# TSR itself is intentionally excluded (see benchmarks/tabular/README.md).
#
# Usage:
#   bash run_tabular_baselines.sh              # full sweep, 1 seed
#   bash run_tabular_baselines.sh --smoke      # 1 seed, 3 epochs, mlp only
#   bash run_tabular_baselines.sh --datasets higgs covertype  # narrow the dataset list
#   bash run_tabular_baselines.sh --presets l      # narrow the size sweep further
#   bash run_tabular_baselines.sh --max-parallel 8    # more concurrent processes on the GPU
#   bash run_tabular_baselines.sh --models mlp resnet_mlp # override the default model list

set -e

SEEDS="42"
DEVICE="auto"
RESULTS_ROOT="benchmarks/tabular/results"
ALL_DATASETS=(adult california_housing covertype higgs fashion_mnist)

COMMON_ARGS=()
MODELS_ARGS=()
DATASETS=("${ALL_DATASETS[@]}")

for arg in "$@"; do
    case $arg in
        --smoke)
            SEEDS="42"
            MODELS_ARGS=(--models mlp)
            COMMON_ARGS+=(--presets m --epochs 3)
            DATASETS=(adult)
            ;;
    esac
done

# Pass through --presets / --max-parallel / --models / --datasets if the user
# supplied them explicitly (beyond --smoke, which already sets its own narrow
# defaults above).
ARGS=("$@")
for i in "${!ARGS[@]}"; do
    if [[ "${ARGS[$i]}" == "--presets" ]]; then
        COMMON_ARGS+=(--presets)
        j=$((i+1))
        while [[ $j -lt ${#ARGS[@]} && "${ARGS[$j]}" != --* ]]; do
            COMMON_ARGS+=("${ARGS[$j]}"); j=$((j+1))
        done
    fi
    if [[ "${ARGS[$i]}" == "--max-parallel" ]]; then
        COMMON_ARGS+=(--max-parallel "${ARGS[$((i+1))]}")
    fi
    if [[ "${ARGS[$i]}" == "--models" ]]; then
        MODELS_ARGS=(--models)
        j=$((i+1))
        while [[ $j -lt ${#ARGS[@]} && "${ARGS[$j]}" != --* ]]; do
            MODELS_ARGS+=("${ARGS[$j]}"); j=$((j+1))
        done
    fi
    if [[ "${ARGS[$i]}" == "--datasets" ]]; then
        DATASETS=()
        j=$((i+1))
        while [[ $j -lt ${#ARGS[@]} && "${ARGS[$j]}" != --* ]]; do
            DATASETS+=("${ARGS[$j]}"); j=$((j+1))
        done
    fi
done

# Default: all 5 models. Only kicks in when neither --smoke nor an explicit
# --models was given.
if [[ ${#MODELS_ARGS[@]} -eq 0 ]]; then
    MODELS_ARGS=(--models mlp resnet_mlp ft_transformer tabnet saint)
fi

source .venv/bin/activate

run_tabular() {
    local dataset=$1
    echo ""
    echo "========================================================"
    echo "TABULAR — $dataset"
    echo "Results -> $RESULTS_ROOT/$dataset/"
    echo "========================================================"
    python benchmarks/tabular/run_tabular.py \
        --dataset "$dataset" \
        --config "benchmarks/tabular/configs/$dataset.yaml" \
        --seeds $SEEDS \
        --results-dir "$RESULTS_ROOT/$dataset" \
        --device "$DEVICE" \
        "${MODELS_ARGS[@]}" "${COMMON_ARGS[@]}"
}

for dataset in "${DATASETS[@]}"; do
    run_tabular "$dataset"
done

echo ""
echo "Done. Per-dataset summaries in $RESULTS_ROOT/<dataset>/summary.json"
