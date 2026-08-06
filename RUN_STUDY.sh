#!/usr/bin/env bash
# RUN_STUDY.sh — every run the paper needs, in dependency order.
#
#   bash RUN_STUDY.sh          # everything (~10 h with 4-way parallelism)
#   bash RUN_STUDY.sh main     # just the main sweeps
#   bash RUN_STUDY.sh base     # just the baselines
#   bash RUN_STUDY.sh figs     # just the missing figures
#
# Every stage skips targets that already have a CSV, so this is safe to
# interrupt and rerun.
set -uo pipefail
STAGE=${1:-all}
CH1="Fp1 AF3 F7 F3 FC1 FC5 T7 C3"
CH2="CP1 CP5 P7 P3 Pz PO3 O1 Oz"
W="25 50 100 200 400"

run4 () {   # run one configuration split over both channel halves
  local tag=$1; shift
  caffeinate -i nohup env CHANNELS="$CH1" "$@" bash run_all_targets.sh > "log_${tag}_1.log" 2>&1 &
  caffeinate -i nohup env CHANNELS="$CH2" "$@" bash run_all_targets.sh > "log_${tag}_2.log" 2>&1 &
}

if [ "$STAGE" = all ] || [ "$STAGE" = main ]; then
  echo "== main sweeps: 16 targets x 2 references, with MINE =="
  run4 avg MINE_ITERS=1500 WINDOWS="$W" OUTDIR=results/targets_mine
  run4 csd MINE_ITERS=1500 WINDOWS="$W" REFERENCE=csd OUTDIR=results/targets_csd_mine
  wait
fi

if [ "$STAGE" = all ] || [ "$STAGE" = base ]; then
  echo "== baselines: 16 targets each, no MINE =="
  run4 pearson RANK_ESTIMATOR=pearson WINDOWS="$W" OUTDIR=results/base_pearson
  run4 ridge   PREDICTOR=ridge        WINDOWS="$W" OUTDIR=results/base_ridge
  wait
  run4 near     SELECTIONS="top random bottom nearest single" WINDOWS="$W" OUTDIR=results/base_nearest
  run4 nearcsd  SELECTIONS="top nearest" REFERENCE=csd WINDOWS="25 100" OUTDIR=results/base_nearest_csd
  wait
fi

if [ "$STAGE" = all ] || [ "$STAGE" = band ]; then
  echo "== band-resolved, matched target =="
  for b in theta alpha beta; do
    uv run neurovision mine --band "$b" --target PO3 --k 6 --windows 25 100 \
        --mine-iters 0 --no-traces --out "results/band_${b}_po3" > "log_band_$b.log" 2>&1
  done
fi

if [ "$STAGE" = all ] || [ "$STAGE" = figs ]; then
  echo "== traced run for the two missing figures =="
  uv run neurovision mine --target PO3 --k 6 --windows $W --mine-iters 1500 \
      --out results/benchmark > log_traces.log 2>&1
  echo "== pairwise geometry test, all recordings, both references =="
  uv run python pairwise_distance.py --recordings 0 --out results/pairwise
  uv run python pairwise_distance.py --recordings 0 --reference csd --out results/pairwise_csd
fi

echo
echo "== completeness check =="
for d in results/targets_mine results/targets_csd_mine results/base_pearson \
         results/base_ridge results/base_nearest results/base_nearest_csd; do
  [ -d "$d" ] || continue
  n=$(ls "$d"/*.csv 2>/dev/null | grep -vc group_stats)
  echo "$d: $n/16 targets"
  for f in "$d"/*.csv; do
    case "$f" in *group_stats*) continue;; esac
    rows=$(($(wc -l < "$f") - 1))
    [ "$rows" -gt 100 ] || echo "  SHORT: $f ($rows rows)"
  done
done

echo
echo "== merges =="
for d in targets_mine targets_csd_mine base_pearson base_ridge base_nearest base_nearest_csd; do
  [ -d "results/$d" ] && uv run python merge_targets.py --dir "results/$d" \
      --out "results/all_$d" > "log_merge_$d.log" 2>&1 && echo "merged $d"
done
