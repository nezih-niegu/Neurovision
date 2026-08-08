#!/usr/bin/env bash
# run_all_targets.sh — run the reconstruction benchmark once per target channel.
#
#   bash run_all_targets.sh                  # default: no MINE, all 5 windows
#   MINE_ITERS=1500 bash run_all_targets.sh  # with MINE (much slower)
#   WINDOWS="25 50 100" bash run_all_targets.sh
#   RANK_ESTIMATOR=pearson OUTDIR=results/base_pearson bash run_all_targets.sh
#   PREDICTOR=ridge OUTDIR=results/base_ridge bash run_all_targets.sh
#   SELECTIONS="top random bottom nearest" OUTDIR=results/base_nearest bash run_all_targets.sh
#   REFERENCE=csd BAND=alpha OUTDIR=results/band_alpha bash run_all_targets.sh
#
# All variables: CHANNELS WINDOWS MINE_ITERS K DURATION REFERENCE
#                RANK_ESTIMATOR PREDICTOR SELECTIONS BAND OUTDIR
#                JOBS CACHE DEVICE
#
# Parallelism lives inside each run now (JOBS workers over recordings), so run
# one configuration at a time with a high JOBS rather than several configs at
# once: the latter re-reads and re-preprocesses the same recordings N times.
#
# Each target writes results/targets/<CH>.csv etc. Already-completed targets are
# skipped, so the script is safe to interrupt and restart.
set -euo pipefail

CHANNELS=${CHANNELS:-"Fp1 AF3 F7 F3 FC1 FC5 T7 C3 CP1 CP5 P7 P3 Pz PO3 O1 Oz"}
WINDOWS=${WINDOWS:-"25 50 100 200 400"}
MINE_ITERS=${MINE_ITERS:-0}
K=${K:-6}
DURATION=${DURATION:-180}
REFERENCE=${REFERENCE:-average}
RANK_ESTIMATOR=${RANK_ESTIMATOR:-gcmi}
PREDICTOR=${PREDICTOR:-mlp}
MINE_LAMBDA=${MINE_LAMBDA:-0.1}
SELECTIONS=${SELECTIONS:-"top random bottom"}
BAND=${BAND:-broadband}
JOBS=${JOBS:-0}          # 0 = all cores but one
QUIET=${QUIET:-0}        # 1 = write logs only, no screen output
CACHE=${CACHE:-.preproc_cache}
DEVICE=${DEVICE:-auto}
OUTDIR=${OUTDIR:-results/targets}

mkdir -p "$OUTDIR"
total=$(echo "$CHANNELS" | wc -w | tr -d ' ')
i=0
ran=0
start=$(date +%s)

for ch in $CHANNELS; do
  i=$((i+1))
  out="$OUTDIR/$ch"
  if [ -f "$out.csv" ]; then
    echo "[$i/$total] $ch — already done, skipping"
    continue
  fi
  echo "[$i/$total] $ch — running..."
  uv run neurovision mine \
      --target "$ch" \
      --k "$K" \
      --windows $WINDOWS \
      --duration "$DURATION" \
      --reference "$REFERENCE" \
      --rank-estimator "$RANK_ESTIMATOR" \
      --predictor "$PREDICTOR" \
      --mine-lambda "$MINE_LAMBDA" \
      --selections $SELECTIONS \
      --band "$BAND" \
      --mine-iters "$MINE_ITERS" \
      --jobs "$JOBS" \
      --cache-dir "$CACHE" \
      --device "$DEVICE" \
      --no-traces \
      --out "$out" \
      2>&1 | tee "$out.log" | { [ "${QUIET:-0}" = 1 ] && cat > /dev/null || sed "s/^/     /"; }
  # Summarise via the project environment: bare python3 is the system
  # interpreter and has none of the project's dependencies.
  summary=$(uv run python -c "
import pandas as pd
d = pd.read_csv('$out.csv')
print('target=%s, mean top R2=%.3f' % (d.target.iloc[0], d[d.selection=='top'].r2.mean()))
" 2>/dev/null || echo "written")
  now=$(date +%s); el=$(( now - start ))
  # done counts only targets actually run in this invocation, so the estimate
  # stays honest when earlier targets were skipped as already complete.
  ran=$(( ran + 1 )); per=$(( el / ran )); left=$(( (total - i) * per ))
  printf "     done (%s) — %dm elapsed, ~%dm left\n" \
      "$summary" "$(( el / 60 ))" "$(( left / 60 ))"
done

elapsed=$(( $(date +%s) - start ))
echo
echo "All targets finished in $((elapsed/60)) min. Now merge and analyse:"
echo "    uv run python merge_targets.py --dir $OUTDIR"
