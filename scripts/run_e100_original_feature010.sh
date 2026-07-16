#!/usr/bin/env bash
# 两张 GPU 并行训练：原始单通道 N2N 与 w_mask_feature=0.10。
set -euo pipefail

DATA="${DATA:-/mnt2/songyd/5x5}"
SEED="${SEED:-42}"
GPU_N2N="${GPU_N2N:-0}"
GPU_FEATURE="${GPU_FEATURE:-1}"

N2N_DIR="results/checkpoints/n2n_original_E100_s${SEED}"
FEATURE_DIR="results/checkpoints/masktune_E100_C_feature_w010_s${SEED}"
RUN_LOG_DIR="results/logs/E100_original_feature010_s${SEED}"

if [[ -e "$N2N_DIR" || -e "$FEATURE_DIR" ]]; then
  echo "[ERROR] 输出目录已存在；为避免覆盖或重复追加 history，请先改目录名或确认旧结果。"
  echo "        $N2N_DIR"
  echo "        $FEATURE_DIR"
  exit 2
fi

mkdir -p "$RUN_LOG_DIR"

echo "[INFO] data=$DATA seed=$SEED"
echo "[INFO] original N2N -> physical GPU $GPU_N2N"
echo "[INFO] feature w=0.10 -> physical GPU $GPU_FEATURE"
echo "[INFO] git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

CUDA_VISIBLE_DEVICES="$GPU_N2N" python -u train_n2n.py \
  --data_path "$DATA" \
  --levels 4 \
  --intervals 5 7 9 \
  --epochs 100 \
  --crop_size 512 \
  --batch_size 16 \
  --lr 0.01 \
  --lr_final 0.0005 \
  --warmup_pct 0.1 \
  --rtv_weight 0.01 \
  --weight_decay 0.01 \
  --train_fraction 0.99 \
  --val_limit_batches 20 \
  --deterministic_loader_rng 1 \
  --data_parallel 0 \
  --plot_loss_curve 1 \
  --seed "$SEED" \
  --device cuda \
  --save_dir "$N2N_DIR" \
  --log_dir "$RUN_LOG_DIR/tensorboard_n2n" \
  > "$RUN_LOG_DIR/original_n2n.log" 2>&1 &
PID_N2N=$!

CUDA_VISIBLE_DEVICES="$GPU_FEATURE" python -u train_masked.py \
  --data_path "$DATA" \
  --levels 4 \
  --intervals 5 7 9 \
  --epochs 100 \
  --crop_size 512 \
  --batch_size 16 \
  --lr 0.01 \
  --lr_final 0.0005 \
  --warmup_pct 0.1 \
  --rtv_weight 0.01 \
  --weight_decay 0.0001 \
  --train_fraction 0.99 \
  --val_limit_batches 20 \
  --mask_ratio 0.25 \
  --mask_patch 16 \
  --mask_fill zero \
  --w_mask_pixel 0 \
  --w_mask_feature 0.10 \
  --mask_feature_scales encoder2 encoder3 \
  --predictor_hidden_ratio 1.0 \
  --ema_decay 0.996 \
  --feature_warmup_frac 0.1 \
  --freeze_masked_bn_stats 1 \
  --deterministic_loader_rng 1 \
  --data_parallel 0 \
  --plot_loss_curve 1 \
  --seed "$SEED" \
  --device cuda \
  --save_dir "$FEATURE_DIR" \
  > "$RUN_LOG_DIR/feature_w010.log" 2>&1 &
PID_FEATURE=$!

printf '%s\n' "$PID_N2N" > "$RUN_LOG_DIR/original_n2n.pid"
printf '%s\n' "$PID_FEATURE" > "$RUN_LOG_DIR/feature_w010.pid"
echo "[INFO] started original PID=$PID_N2N, feature PID=$PID_FEATURE"

STATUS_N2N=0
STATUS_FEATURE=0
wait "$PID_N2N" || STATUS_N2N=$?
wait "$PID_FEATURE" || STATUS_FEATURE=$?

echo "[INFO] original exit=$STATUS_N2N, feature exit=$STATUS_FEATURE"
if [[ "$STATUS_N2N" -ne 0 || "$STATUS_FEATURE" -ne 0 ]]; then
  exit 1
fi
echo "[OK] both 100-epoch training jobs completed"
