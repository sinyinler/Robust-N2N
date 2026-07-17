#!/usr/bin/env bash
# 只训练新增的单通道局部 Gaussian feature arm；Original/Mask-C 可复用既有公平 E100 结果。
set -euo pipefail

DATA="${DATA:-/mnt2/songyd/5x5}"
NOISE_DATA="${NOISE_DATA:-/mnt2/songyd/5x5/5x5x4}"
NOISE_STATS="${NOISE_STATS:-results/eval/noise_stats_level4_log1p.json}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
BATCH="${BATCH:-12}"
SAVE_DIR="results/checkpoints/noisetune_E100_N_feature_w010_b${BATCH}_s${SEED}"
RUN_LOG_DIR="results/logs/E100_noise_feature010_b${BATCH}_s${SEED}"

if [[ -e "$SAVE_DIR" ]]; then
  echo "[ERROR] 输出目录已存在；为避免重复追加 history，请先改名或确认旧结果：$SAVE_DIR"
  exit 2
fi

mkdir -p "$(dirname "$NOISE_STATS")" "$RUN_LOG_DIR"

if [[ ! -f "$NOISE_STATS" ]]; then
  echo "[INFO] measuring level4 noise in log1p domain -> $NOISE_STATS"
  python -u scripts/measure_noise.py \
    --data_path "$NOISE_DATA" \
    --intensity_transform log1p \
    --crop 512 \
    --max_frames_per_seq 12 \
    --max_seqs_per_level 40 \
    --out "$NOISE_STATS" \
    > "$RUN_LOG_DIR/measure_noise.log" 2>&1
fi

echo "[INFO] data=$DATA seed=$SEED physical_gpu=$GPU batch=$BATCH"
echo "[INFO] noise_stats=$NOISE_STATS"
echo "[INFO] git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

CUDA_VISIBLE_DEVICES="$GPU" python -u train_masked.py \
  --data_path "$DATA" \
  --levels 4 \
  --intervals 5 7 9 \
  --epochs 100 \
  --crop_size 512 \
  --batch_size "$BATCH" \
  --lr 0.01 \
  --lr_final 0.0005 \
  --warmup_pct 0.1 \
  --rtv_weight 0.01 \
  --weight_decay 0.0001 \
  --train_fraction 0.99 \
  --val_limit_batches 20 \
  --corruption_mode gaussian \
  --mask_ratio 0.25 \
  --mask_patch 16 \
  --noise_stats_json "$NOISE_STATS" \
  --noise_sigma_min_scale 0.25 \
  --noise_sigma_max_scale 0.75 \
  --w_mask_pixel 0 \
  --w_mask_feature 0.10 \
  --mask_feature_scales encoder2 encoder3 \
  --predictor_hidden_ratio 1.0 \
  --ema_decay 0.996 \
  --feature_warmup_frac 0.1 \
  --freeze_masked_bn_stats 1 \
  --deterministic_loader_rng 1 \
  --grad_diag_every 100 \
  --grad_diag_scales encoder2 encoder3 \
  --data_parallel 0 \
  --plot_loss_curve 1 \
  --seed "$SEED" \
  --device cuda \
  --save_dir "$SAVE_DIR" \
  > "$RUN_LOG_DIR/noise_feature_w010.log" 2>&1

echo "[OK] 100-epoch local Gaussian feature training completed: $SAVE_DIR"
