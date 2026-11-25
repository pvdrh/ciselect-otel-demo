#!/usr/bin/env bash
# #51 bước 5–6: RF chia chính rồi LOSO, 5 seed, tuần tự; dừng ở run lỗi đầu tiên hoặc khi E: còn < 1 GB.
set -u
R=E:/luan_van-02-ci-test-selection
cd "$R/experiments/ts02/selection"
export PYTHONHASHSEED=0 PYTHONIOENCODING=utf-8
for split in main loso; do
  [ "$split" = main ] && name=ts02_eval_rf d2=ts02_eval mode="" || { name=ts02_eval_rf_loso; d2=ts02_eval_loso; mode=loso; }
  for seed in 0 1 2 3 4; do
    free=$(df -k /e | awk 'NR==2{print $4}')
    [ "$free" -lt 1048576 ] && { echo "E: còn < 1 GB: dừng"; exit 3; }
    ref=$(ls -d "$R/results/$d2/"*_seed$seed | grep -v '\.log$' | head -1)
    out="$R/results/$name/$(date +%Y%m%d-%H%M%S)_seed$seed"
    echo "$(date +%T) $split seed$seed -> $out (ref $ref)"
    python run_extra.py "$R/results/ts02_dataset/20261001-213236_seed0" "$R/results/build_nocache/20261001-122853_seed0" \
      "$R/data/raw/opentelemetry-demo" "$out" $seed "$ref" $mode > "$out.log" 2>&1 || { echo "LỖI $out"; tail -3 "$out.log"; exit 1; }
    echo "$(date +%T) ok $(grep -o '"baseline_mismatches": [0-9]*' "$out/params.json")"
  done
done
echo XONG
