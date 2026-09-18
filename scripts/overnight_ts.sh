#!/usr/bin/env bash
# Unattended timeseries campaign.
#
# Phase 0 resolves an OPEN QUESTION rather than assuming an answer: the
# channel-independent model (tcn_ci) was introduced because the
# channel-mixing one gave TSR-X almost nothing to reallocate (head = 58-99%
# of params), but on its first gate it scored test MSE 0.5924 against the
# mixing model's 0.4948 on weather/h96 -- i.e. WORSE. That gate was run at
# 15 epochs against a 50-epoch number, so it was not a fair comparison.
# Phase 0 re-runs all three variants at matched epochs and picks the winner
# by val MSE; every later phase uses that winner. If the mixing model wins,
# the campaign still produces a full grid -- just with the honest caveat
# that TSR-X has little room to act on that architecture.
#
# Ordering is deliberate: cheap datasets first, traffic (862 variates) last,
# so an overnight window that runs short still yields complete results for
# weather and electricity rather than a uniformly half-finished grid.
#
# Resumable: every step skips if its output checkpoint already exists.
#
#   bash scripts/overnight_ts.sh 2>&1 | tee overnight.log

set -u -o pipefail

EPOCHS=${EPOCHS:-50}
BUDGET=${BUDGET:-0.5}
END_FRAC=${END_FRAC:-0.3}     # NOT 0.5: the search must converge before
                              # early stopping bites, or C2/C3 train a
                              # barely-pruned architecture (measured: 1.4%
                              # instead of the intended 15%).
SEED=${SEED:-42}
DEV=${DEV:-cuda}
PY=${PY:-python}

mkdir -p results/ablate results/ts_reference results/ts_tsrx \
         results/ts_static_matched results/ts_c3 results/ts_sweep logs

run () {  # run <output.pt> <args...>
  local out="$1"; shift
  if [ -f "$out" ]; then echo "SKIP (exists): $out"; return 0; fi
  local log="logs/$(basename "${out%.pt}").log"
  echo ">>> $(date +%H:%M:%S)  $out"
  if ! $PY "$@" --out "$out" > "$log" 2>&1; then
    echo "!!! FAILED: $out  (see $log)"; tail -5 "$log"; return 1
  fi
}

bs_for () {  # channel-independent runs B*C sequences per batch; 862-variate
             # traffic needs a smaller batch to fit in memory.
  case "$1" in
    traffic) echo 8 ;;
    electricity) echo 16 ;;
    *) echo 32 ;;
  esac
}

# ---------------------------------------------------------------- Phase 0
echo "=== PHASE 0: architecture ablation (weather h96, matched epochs) ==="
run results/ablate/ci_revin_w96.pt   -m bench.train_ts_reference --arch tcn_ci \
    --dataset weather --pred-len 96 --epochs "$EPOCHS" --seed "$SEED" --device "$DEV"
run results/ablate/ci_norevin_w96.pt -m bench.train_ts_reference --arch tcn_ci --no-revin \
    --dataset weather --pred-len 96 --epochs "$EPOCHS" --seed "$SEED" --device "$DEV"
run results/ablate/mixing_w96.pt     -m bench.train_ts_reference --arch tcn \
    --dataset weather --pred-len 96 --epochs "$EPOCHS" --seed "$SEED" --device "$DEV"

read -r ARCH REVIN_FLAG <<< "$(
$PY - <<'EOF'
import torch
cands = [("results/ablate/ci_revin_w96.pt",   "tcn_ci", ""),
         ("results/ablate/ci_norevin_w96.pt", "tcn_ci", "--no-revin"),
         ("results/ablate/mixing_w96.pt",     "tcn",    "")]
best, best_v = None, float("inf")
for p, arch, flag in cands:
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        continue
    v = ck.get("val_mse", float("inf"))
    print(f"# {p:38} val_mse={v:.4f} test_mse={ck.get('test_mse',float('nan')):.4f}",
          file=__import__("sys").stderr)
    if v < best_v:
        best, best_v = (arch, flag), v
print(f"{best[0]} {best[1]}" if best else "tcn_ci ")
EOF
)"
echo "=== PHASE 0 RESULT: arch=$ARCH ${REVIN_FLAG:-(revin on)} ==="

# ---------------------------------------------------------------- Phase 1
# Budget sweep on the CHEAPEST dataset, to locate where allocation starts to
# bite. 0.85 is known to tie everything -- that is why 0.5 is the default
# for the grid, and why this sweep exists to check that choice.
echo "=== PHASE 1: budget sweep (weather h96) ==="
for BR in 0.85 0.5 0.3; do
  run "results/ts_sweep/tsrx_weather_h96_br${BR}.pt" -m bench.train_ts_tsrx \
      --arch "$ARCH" $REVIN_FLAG --dataset weather --pred-len 96 --epochs "$EPOCHS" \
      --budget-ratio "$BR" --budget-end-frac "$END_FRAC" --seed "$SEED" --device "$DEV"
done

# ---------------------------------------------------------------- Phase 2
echo "=== PHASE 2: full 4-arm grid @ budget $BUDGET ==="
for DS in weather electricity traffic; do          # cheapest -> heaviest
  BS=$(bs_for "$DS")
  for H in 96 720; do
    TAG="${ARCH}_${DS}_h${H}"
    T="results/ts_tsrx/${TAG}.pt"

    run "results/ts_reference/${TAG}.pt" -m bench.train_ts_reference \
        --arch "$ARCH" $REVIN_FLAG --dataset "$DS" --pred-len "$H" --epochs "$EPOCHS" \
        --batch-size "$BS" --seed "$SEED" --device "$DEV"

    run "$T" -m bench.train_ts_tsrx \
        --arch "$ARCH" $REVIN_FLAG --dataset "$DS" --pred-len "$H" --epochs "$EPOCHS" \
        --batch-size "$BS" --budget-ratio "$BUDGET" --budget-end-frac "$END_FRAC" \
        --seed "$SEED" --device "$DEV"

    # C2 and C3 both need the TSR-X checkpoint; skip them if it failed.
    if [ -f "$T" ]; then
      run "results/ts_static_matched/${TAG}.pt" -m bench.train_ts_static_matched \
          --tsrx-checkpoint "$T" --epochs "$EPOCHS" --batch-size "$BS" \
          --seed "$SEED" --device "$DEV"
      run "results/ts_c3/${TAG}.pt" -m bench.train_c3_random \
          --tsrx-checkpoint "$T" --domain ts --epochs "$EPOCHS" --batch-size "$BS" \
          --seed "$SEED" --device "$DEV"
    else
      echo "!!! no TSR-X checkpoint for $TAG -- skipping C2/C3"
    fi
  done
done

# ---------------------------------------------------------------- Phase 3
echo "=== PHASE 3: results table ==="
$PY -m analysis.collect_results --domain ts | tee results/ts_table.md
echo
echo "=== budget sweep summary ==="
$PY - <<'EOF'
import glob, os, torch
rows=[]
for p in sorted(glob.glob("results/ts_sweep/*.pt")):
    ck=torch.load(p,map_location="cpu",weights_only=False)
    base=ck.get("baseline_params",0); fin=ck.get("final_params") or ck.get("params")
    rows.append((os.path.basename(p), ck.get("final_test_mse", ck.get("test_mse")),
                 fin, (1-fin/base)*100 if base and fin else float("nan")))
for n,m,f,r in rows:
    print(f"{n:44} test_mse={m if m is None else round(m,4)}  params={f:,}  reduction={r:.1f}%")
EOF
echo "DONE $(date)"
