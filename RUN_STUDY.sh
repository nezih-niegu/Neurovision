#!/usr/bin/env bash
# RUN_STUDY.sh — every run the paper needs, in dependency order.
#
#   bash RUN_STUDY.sh            # everything
#   bash RUN_STUDY.sh main|base|train|band|figs
#
# Parallelism is inside each run: JOBS workers over recordings. Configurations
# run one at a time, because running several at once would re-preprocess the
# same recordings in each. Preprocessed arrays are cached in .preproc_cache and
# reused across every configuration.
#
# Targets that already have a CSV are skipped, so this is safe to interrupt.
set -uo pipefail
STAGE=${1:-all}
JOBS=${JOBS:-4}            # 8 workers with MINE can exhaust memory on 16 GB
DEVICE=${DEVICE:-auto}
QUIET=${QUIET:-0}          # QUIET=1 to log only, without printing to screen
W="25 50 100 200 400"
export JOBS DEVICE
export PYTHONUNBUFFERED=1  # otherwise progress sits in a pipe buffer
# Cap threads per worker: without this each of the N workers asks BLAS for every
# core, and N x cores threads on a laptop is what turns a long run into swap.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# caffeinate keeps a Mac awake through a long run but does not exist elsewhere.
# Without this guard a missing binary makes every stage fail silently and the
# study appears to finish in seconds with no output.
if command -v caffeinate > /dev/null 2>&1; then CAFF="caffeinate -i"; else CAFF=""; fi

# Ctrl-C must stop the whole study, not just the stage that happens to be
# running: without this the loop simply continues to the next configuration.
trap 'echo; echo "interrupted — rerun the same command to resume"; exit 130' INT TERM

sweep () {  # sweep <logname> VAR=val ...
  local tag=$1; shift
  echo "-- $tag"
  if [ "$QUIET" = 1 ]; then
    $CAFF env "$@" bash run_all_targets.sh > "log_${tag}.log" 2>&1
  else
    $CAFF env "$@" bash run_all_targets.sh 2>&1 | tee "log_${tag}.log"
  fi
}

run () {    # run <logname> <command...>  — same tee-or-log behaviour
  local tag=$1; shift
  if [ "$QUIET" = 1 ]; then "$@" > "log_${tag}.log" 2>&1
  else "$@" 2>&1 | tee "log_${tag}.log"; fi
}

if [ "$STAGE" = all ] || [ "$STAGE" = main ]; then
  echo "== main sweeps =="
  sweep avg MINE_ITERS=1500 WINDOWS="$W" OUTDIR=results/targets_mine
  sweep csd MINE_ITERS=1500 WINDOWS="$W" REFERENCE=csd OUTDIR=results/targets_csd_mine
fi

if [ "$STAGE" = all ] || [ "$STAGE" = base ]; then
  echo "== baselines =="
  sweep pearson  RANK_ESTIMATOR=pearson WINDOWS="$W" OUTDIR=results/base_pearson
  sweep ridge    PREDICTOR=ridge        WINDOWS="$W" OUTDIR=results/base_ridge
  sweep nearest  SELECTIONS="top random bottom nearest single" WINDOWS="$W" \
                 OUTDIR=results/base_nearest
  sweep nearcsd  SELECTIONS="top nearest" REFERENCE=csd WINDOWS="25 100" \
                 OUTDIR=results/base_nearest_csd
fi

if [ "$STAGE" = all ] || [ "$STAGE" = train ]; then
  echo "== does MINE help as a training signal? =="
  # lambda sweep on one target first: if the MI term does nothing here, the
  # 16-target arms below are not worth 5 hours.
  # Tags carry no dot: Path.with_suffix() would read ".03" in "lam_0.03" as the
  # file extension, so every lambda would write to results/lam_0.csv.
  for spec in "000:0.0" "003:0.03" "010:0.1" "030:0.3" "100:1.0"; do
    tag=${spec%%:*}; lam=${spec#*:}
    run "lam_$tag" $CAFF uv run neurovision mine --predictor mine \
        --mine-lambda "$lam" --target PO3 --k 6 --windows 25 100 400 \
        --mine-iters 0 --no-traces --jobs "$JOBS" --device "$DEVICE" \
        --out "results/lam_${tag}"
  done
  # the two arms, all targets. PREDICTOR=mine with MINE_LAMBDA=0 would be
  # identical to mlp, so the mlp arm uses the plain path for a fair timing too.
  sweep train_mse  PREDICTOR=mlp  WINDOWS="$W" OUTDIR=results/train_mse
  sweep train_mine PREDICTOR=mine MINE_LAMBDA="${MINE_LAMBDA:-0.1}" \
                   WINDOWS="$W" OUTDIR=results/train_mine
fi

if [ "$STAGE" = all ] || [ "$STAGE" = band ]; then
  echo "== band-resolved, matched target =="
  for b in theta alpha beta; do
    run "band_$b" $CAFF uv run neurovision mine --band "$b" \
        --target PO3 --k 6 --windows 25 100 --mine-iters 0 --no-traces \
        --jobs "$JOBS" --device "$DEVICE" --out "results/band_${b}_po3"
  done
fi

if [ "$STAGE" = all ] || [ "$STAGE" = figs ]; then
  echo "== traced run for the two missing figures =="
  run traces $CAFF uv run neurovision mine --target PO3 --k 6 \
      --windows $W --mine-iters 1500 --jobs "$JOBS" --device "$DEVICE" \
      --out results/benchmark
  # The pairwise geometry test at full coverage is deliberately NOT repeated
  # here: results/pairwise.csv and results/pairwise_csd.csv already cover all
  # 46 recordings. Delete those files first if you want them regenerated.
  for f in results/pairwise.csv results/pairwise_csd.csv; do
    [ -f "$f" ] || echo "MISSING $f — run pairwise_distance.py --recordings 0"
  done
fi

echo
echo "== completeness =="
for d in results/targets_mine results/targets_csd_mine results/base_pearson \
         results/base_ridge results/base_nearest results/base_nearest_csd \
         results/train_mse results/train_mine; do
  [ -d "$d" ] || continue
  n=$(ls "$d"/*.csv 2>/dev/null | grep -vc group_stats)
  echo "$d: $n/16 targets"
  for f in "$d"/*.csv; do
    case "$f" in *group_stats*) continue;; esac
    r=$(($(wc -l < "$f") - 1)); [ "$r" -gt 100 ] || echo "  SHORT: $f ($r rows)"
  done
done

echo
echo "== merges =="
for d in targets_mine targets_csd_mine base_pearson base_ridge base_nearest \
         base_nearest_csd train_mse train_mine; do
  [ -d "results/$d" ] && uv run python merge_targets.py --dir "results/$d" \
      --out "results/all_$d" > "log_merge_$d.log" 2>&1 && echo "merged $d"
done
