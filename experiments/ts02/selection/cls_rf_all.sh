#!/usr/bin/env bash
# Plan 2026-10-07-d2-rf-oracle bước 7: classification report cho 10 run RF (#51), sau khi batch run_extra_all.sh xong.
set -euo pipefail
W=/e/luan_van-02-ci-test-selection
R=$W/results
cd "$W/experiments/ts02/selection"
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8
DEC="#51 (thăm dò) plans/2026-10-07-d2-rf-oracle.md"
for split in ts02_eval_rf ts02_eval_rf_loso; do
  out=${split/ts02_eval_rf/ts02_cls_rf}
  for run in "$R/$split"/*_seed[0-4]; do
    seed=${run##*_seed}
    dst="$R/$out/$(date +%Y%m%d-%H%M%S)_seed$seed"
    echo "$(date +%T) $run -> $dst"
    python cls_report.py "$R/ts02_dataset/20261001-213236_seed0" "$R/build_nocache/20261001-122853_seed0" \
      "$W/data/raw/opentelemetry-demo" "$run" "$dst" "$DEC"
  done
done
echo "XONG"
