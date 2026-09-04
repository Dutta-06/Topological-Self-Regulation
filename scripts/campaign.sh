#!/usr/bin/env bash
# TSR-X results campaign: locate the compression knee, get error bars, prove
# the mechanism.
#
# Where this comes from: on weather/h96 we measured 70.7% parameter reduction
# at +0.9% test MSE (reference 0.4946 -> C2 0.4992) against SPAT's published
# 28.19%. But 0.3 was an arbitrary budget, not a limit -- the feasibility
# floor at --min-size 8 is 86-91% across these datasets, and nothing had
# broken at 70.7%. Stage 1 finds where it actually breaks.
#
# The second result is the mechanism: C2 (signal) vs C3 (random) at matched
# budget flips sign as capacity binds (-0.0038 at 15% reduction, +0.0035 at
# 50%, +0.0069 at 70.7%), exactly as gamma = u/kappa predicts. Stage 3 makes
# that robust to the random draw instead of one sample per point.
#
# Resumable: every step skips if its checkpoint exists, so re-running after
# an interruption costs nothing. Ordered cheapest-first so a short window
# still yields complete lower stages.
#
#   bash scripts/campaign.sh 2>&1 | tee campaign.log
#
# Env overrides: EPOCHS, DEV, PY, STAGES (e.g. STAGES="1 2"), KNEE (skip
# Stage 1's auto-detection and force the knee budget).

set -u -o pipefail

EPOCHS=${EPOCHS:-50}
END_FRAC=${END_FRAC:-0.3}   # NOT 0.5 -- the search must converge before early
                            # stopping bites, or C2/C3 train a barely-pruned
                            # architecture. That bug voided a whole campaign.
DEV=${DEV:-cuda}
PY=${PY:-python}
SEED=${SEED:-42}
STAGES=${STAGES:-"1 2 3"}
DATASETS=${DATASETS:-"weather electricity traffic"}
BUDGETS=${BUDGETS:-"0.85 0.50 0.30 0.20 0.15 0.10"}
H=${H:-96}
ARCH=${ARCH:-tcn}

mkdir -p results/ts_sweep results/ts_c3 logs

bs_for () { case "$1" in traffic) echo 8 ;; electricity) echo 16 ;; *) echo 32 ;; esac; }
has_stage () { [[ " $STAGES " == *" $1 "* ]]; }

run () {  # run <out.pt> <module+args...>
  local out="$1"; shift
  if [ -f "$out" ]; then echo "    skip (exists): $out"; return 0; fi
  local log="logs/$(basename "${out%.pt}").log"
  echo "    >>> $(date +%H:%M:%S) $out"
  if ! $PY "$@" --out "$out" > "$log" 2>&1; then
    echo "    !!! FAILED: $out"; tail -4 "$log"; return 1
  fi
}

# TSR-X -> C2 -> C3 for one (dataset, budget, seed). C2/C3 read the TSR-X
# checkpoint's final_widths, so they must follow it and are skipped if it
# failed (an infeasible budget exits non-zero by design).
triple () {
  local ds=$1 br=$2 seed=$3 c3seed=${4:-} suffix=${5:-}
  local bs; bs=$(bs_for "$ds")
  local tag="${ds}_h${H}_br${br}"
  [ "$seed" != "$SEED" ] && tag="${tag}_s${seed}"
  local T="results/ts_sweep/tsrx_${tag}.pt"

  run "$T" -m bench.train_ts_tsrx --arch "$ARCH" --dataset "$ds" --pred-len "$H" \
      --epochs "$EPOCHS" --batch-size "$bs" --budget-ratio "$br" \
      --budget-end-frac "$END_FRAC" --seed "$seed" --device "$DEV" || return 0
  [ -f "$T" ] || { echo "    (no TSR-X checkpoint for $tag -- skipping C2/C3)"; return 0; }

  run "results/ts_sweep/c2_${tag}.pt" -m bench.train_ts_static_matched \
      --tsrx-checkpoint "$T" --epochs "$EPOCHS" --batch-size "$bs" \
      --seed "$seed" --device "$DEV"

  if [ -n "$c3seed" ]; then
    run "results/ts_sweep/c3_${tag}${suffix}.pt" -m bench.train_c3_random \
        --tsrx-checkpoint "$T" --domain ts --epochs "$EPOCHS" --batch-size "$bs" \
        --seed "$seed" --c3-seed "$c3seed" --device "$DEV"
  else
    run "results/ts_sweep/c3_${tag}.pt" -m bench.train_c3_random \
        --tsrx-checkpoint "$T" --domain ts --epochs "$EPOCHS" --batch-size "$bs" \
        --seed "$seed" --device "$DEV"
  fi
}

# ------------------------------------------------------------------ Stage 1
if has_stage 1; then
  echo "=== STAGE 1: compression frontier — locate the knee ==="
  for ds in $DATASETS; do
    for br in $BUDGETS; do
      echo "  [$ds br=$br]"
      triple "$ds" "$br" "$SEED"
    done
    echo "--- frontier so far: $ds ---"
    $PY -m analysis.sweep_table --dataset "$ds" --horizon "$H" \
        --reference "results/ts_reference/${ARCH}_${ds}_h${H}.pt" 2>/dev/null || true
  done
fi

# The knee: tightest budget whose C2 test MSE stays within TOL of the
# reference. Deliberately a fixed tolerance, declared up front, rather than
# "best MSE" -- picking the argmin would be selecting on noise, and the
# Stage-1 deltas are already known to be non-monotone in compression.
detect_knee () {
  $PY - "$1" <<'EOF'
import glob, os, re, sys, torch
ds = sys.argv[1]; TOL = float(os.environ.get("TOL", "0.03"))  # 3% relative
ref = None
for p in glob.glob(f"results/ts_reference/*_{ds}_h96.pt"):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    ref = ck.get("test_mse"); break
if ref is None: print(""); raise SystemExit
best = ""
for p in sorted(glob.glob(f"results/ts_sweep/c2_{ds}_h96_br*.pt")):
    m = re.search(r"_br([0-9.]+)\.pt$", p)
    if not m: continue
    ck = torch.load(p, map_location="cpu", weights_only=False)
    t = ck.get("test_mse")
    if t is None: continue
    if (t - ref) / ref <= TOL:
        if best == "" or float(m.group(1)) < float(best):
            best = m.group(1)
print(best)
EOF
}

# ------------------------------------------------------------------ Stage 2
if has_stage 2; then
  echo "=== STAGE 2: error bars at the knee (seeds 43, 44) ==="
  for ds in $DATASETS; do
    K=${KNEE:-$(detect_knee "$ds")}
    if [ -z "$K" ]; then echo "  [$ds] no budget within tolerance — skipping"; continue; fi
    echo "  [$ds] knee budget = $K"
    for s in 43 44; do triple "$ds" "$K" "$s"; done
  done
fi

# ------------------------------------------------------------------ Stage 3
if has_stage 3; then
  echo "=== STAGE 3: mechanism robustness — extra C3 random draws at the knee ==="
  for ds in $DATASETS; do
    K=${KNEE:-$(detect_knee "$ds")}
    if [ -z "$K" ]; then echo "  [$ds] no knee — skipping"; continue; fi
    bs=$(bs_for "$ds")
    T="results/ts_sweep/tsrx_${ds}_h${H}_br${K}.pt"
    [ -f "$T" ] || continue
    for cs in 1 2 3; do
      run "results/ts_sweep/c3_${ds}_h${H}_br${K}_c${cs}.pt" -m bench.train_c3_random \
          --tsrx-checkpoint "$T" --domain ts --epochs "$EPOCHS" --batch-size "$bs" \
          --seed "$SEED" --c3-seed "$cs" --device "$DEV"
    done
  done
fi

# ------------------------------------------------------------------ report
echo
echo "=== FINAL FRONTIER ==="
for ds in $DATASETS; do
  echo
  $PY -m analysis.sweep_table --dataset "$ds" --horizon "$H" \
      --reference "results/ts_reference/${ARCH}_${ds}_h${H}.pt" || true
done | tee results/ts_frontier.md
echo
echo "DONE $(date)"
