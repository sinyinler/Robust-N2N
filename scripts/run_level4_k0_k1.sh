#!/usr/bin/env bash
# 在两张 GPU 上并行训练 K0/K1，随后与 W1 做 Level4 scene0 前 500 帧配对评估。
# 该脚本用于服务器正式实验；重复运行时会跳过已完成模型，并从最新 epoch 自动续训中断任务。
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/songyd/Projects/Robust-N2N}"
DATA="${DATA:-/mnt2/songyd/5x5}"
NOISE_DATA="${NOISE_DATA:-/mnt2/songyd/5x5/5x5x4}"
NOISE_STATS="${NOISE_STATS:-results/eval/noise_stats_level4_log1p.json}"
REFERENCE="${REFERENCE:-/home/songyd/Projects/Robust-N2N/reference.npy}"
SCENE_DIR="${SCENE_DIR:-/mnt2/songyd/5x5/5x5x4/0/npy}"
SEED="${SEED:-42}"
BATCH="${BATCH:-12}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
RUN_TESTS="${RUN_TESTS:-1}"

CHECKPOINT_ROOT="results/checkpoints/paper_level4_K0_K1_E100_v1"
LOG_ROOT="results/logs/paper_level4_K0_K1_E100_v1"
EVAL_ROOT="results/eval_paper/level4_K0_K1_E100_v1"
W1_PARENT="${W1_PARENT:-results/checkpoints/paper_lv4_scenes0_29_E100_v1}"

K0_DIR="$CHECKPOINT_ROOT/K0_Warm20_Weight005_Patch8_s${SEED}"
K1_DIR="$CHECKPOINT_ROOT/K1_NoGaussian_EMA0999_Warm20_Weight005_Patch8_s${SEED}"

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$NOISE_STATS")" "$CHECKPOINT_ROOT" "$LOG_ROOT" "$EVAL_ROOT"

if [[ ! -f "$REFERENCE" ]]; then
  echo "[ERROR] reference 不存在：$REFERENCE" >&2
  exit 2
fi
if [[ ! -d "$SCENE_DIR" ]]; then
  echo "[ERROR] Level4 scene0 目录不存在：$SCENE_DIR" >&2
  exit 2
fi

# K0 需要从训练数据估计 log1p 域 Gaussian 参考强度；K1 虽为零强度，
# 仍读取同一份统计文件，确保除了明确列出的两个变量外训练入口完全一致。
if [[ ! -f "$NOISE_STATS" ]]; then
  echo "[INFO] 测量 Level4 log1p 噪声强度 -> $NOISE_STATS"
  python -u scripts/measure_noise.py \
    --data_path "$NOISE_DATA" \
    --intensity_transform log1p \
    --crop 512 \
    --max_frames_per_seq 12 \
    --max_seqs_per_level 40 \
    --out "$NOISE_STATS" \
    > "$LOG_ROOT/measure_noise.log" 2>&1
fi

if [[ "$RUN_TESTS" == "1" ]]; then
  python -m unittest \
    tests.test_masked_resume \
    tests.test_gaussian_corruption \
    tests.test_summarize_level4_k0_k1 \
    > "$LOG_ROOT/unit_tests.log" 2>&1
  echo "[OK] resume、Gaussian=0与汇总单元测试通过"
fi

echo "[INFO] git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[INFO] seed=$SEED batch=$BATCH physical_GPUs=$GPU0,$GPU1"
echo "[INFO] K0: Gaussian=0.25--0.75sigma EMA=0.996 warmup=20% weight=0.05 patch=8"
echo "[INFO] K1: Gaussian=0 EMA=0.999 warmup=20% weight=0.05 patch=8"

latest_epoch_checkpoint() {
  local save_dir="$1"
  find "$save_dir" -maxdepth 1 -type f -name 'model_epoch_*.pth' -print 2>/dev/null \
    | sort -V \
    | tail -n 1
}

train_one() {
  local gpu="$1"
  local tag="$2"
  local save_dir="$3"
  local sigma_min_scale="$4"
  local sigma_max_scale="$5"
  local ema_decay="$6"
  local train_log="$LOG_ROOT/${tag}_train.log"
  local final_checkpoint="$save_dir/model_epoch_100.pth"
  local -a resume_args=()

  if [[ -f "$final_checkpoint" ]]; then
    echo "[SKIP] $tag 已训练完成：$final_checkpoint"
    return 0
  fi

  if [[ -d "$save_dir" ]]; then
    local resume_checkpoint
    resume_checkpoint="$(latest_epoch_checkpoint "$save_dir")"
    if [[ -z "$resume_checkpoint" ]]; then
      echo "[ERROR] $tag 输出目录已存在但没有 epoch checkpoint：$save_dir" >&2
      echo "[ERROR] 请人工检查目录，避免覆盖不完整实验。" >&2
      return 3
    fi
    resume_args=(--resume_checkpoint "$resume_checkpoint")
    echo "[RESUME] $tag 从 $resume_checkpoint 继续"
  else
    mkdir -p "$save_dir"
    : > "$train_log"
    echo "[TRAIN] $tag 从头训练"
  fi

  {
    echo "========== $tag $(date --iso-8601=seconds) =========="
    echo "physical_gpu=$gpu save_dir=$save_dir"
    CUDA_VISIBLE_DEVICES="$gpu" python -u train_masked.py \
      --data_path "$DATA" \
      --levels 4 \
      --data_index_min 0 \
      --data_index_max 29 \
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
      --intensity_transform log1p \
      --corruption_mode gaussian \
      --mask_ratio 0.25 \
      --mask_patch 8 \
      --noise_stats_json "$NOISE_STATS" \
      --noise_sigma_min_scale "$sigma_min_scale" \
      --noise_sigma_max_scale "$sigma_max_scale" \
      --w_mask_pixel 0 \
      --w_mask_feature 0.05 \
      --mask_feature_scales encoder2 encoder3 \
      --predictor_hidden_ratio 1.0 \
      --ema_decay "$ema_decay" \
      --feature_warmup_frac 0.20 \
      --freeze_masked_bn_stats 1 \
      --deterministic_loader_rng 1 \
      --grad_diag_every 100 \
      --grad_diag_scales encoder2 encoder3 \
      --data_parallel 0 \
      --plot_loss_curve 1 \
      --progress 0 \
      --seed "$SEED" \
      --device cuda \
      --save_dir "$save_dir" \
      "${resume_args[@]}"
  } >> "$train_log" 2>&1

  if [[ ! -f "$final_checkpoint" ]]; then
    echo "[ERROR] $tag 进程结束但缺少 $final_checkpoint" >&2
    return 4
  fi
  echo "[OK] $tag 训练完成"
}

# K0/K1 使用两张物理 GPU 同时训练；每个进程只看到一张卡，避免 DataParallel 串卡。
train_one "$GPU0" K0 "$K0_DIR" 0.25 0.75 0.996 &
PID_K0=$!
train_one "$GPU1" K1 "$K1_DIR" 0.0 0.0 0.999 &
PID_K1=$!
wait "$PID_K0"
wait "$PID_K1"

resolve_w1_checkpoint() {
  if [[ -n "${W1_CHECKPOINT:-}" ]]; then
    if [[ ! -f "$W1_CHECKPOINT" ]]; then
      echo "[ERROR] W1_CHECKPOINT 不存在：$W1_CHECKPOINT" >&2
      return 5
    fi
    printf '%s\n' "$W1_CHECKPOINT"
    return 0
  fi

  local -a candidates=()
  mapfile -t candidates < <(
    find "$W1_PARENT" -mindepth 2 -maxdepth 2 -type f \
      -path "*/W1*s${SEED}*/model_epoch_100.pth" | sort
  )
  if [[ "${#candidates[@]}" -ne 1 ]]; then
    echo "[ERROR] 应当唯一找到 W1 seed${SEED} epoch100，实际为 ${#candidates[@]} 个：" >&2
    printf '  %s\n' "${candidates[@]:-<none>}" >&2
    echo "[ERROR] 可通过环境变量 W1_CHECKPOINT=/absolute/path/model_epoch_100.pth 明确指定。" >&2
    return 5
  fi
  printf '%s\n' "${candidates[0]}"
}

W1_CHECKPOINT_RESOLVED="$(resolve_w1_checkpoint)"
echo "[INFO] W1 baseline=$W1_CHECKPOINT_RESOLVED"

eval_one() {
  local gpu="$1"
  local tag="$2"
  local checkpoint="$3"
  local out_dir="$EVAL_ROOT/level4_seen_${tag}_vs_W1_s${SEED}"

  echo "[EVAL] $tag vs W1 physical_gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u eval_curve.py \
    --checkpoint "$checkpoint" \
    --masked_model 0 \
    --baseline_checkpoint "$W1_CHECKPOINT_RESOLVED" \
    --baseline_masked_model 0 \
    --scene_dir "$SCENE_DIR" \
    --n_frames 500 \
    --reference "$REFERENCE" \
    --max 255 \
    --photometric_diagnostic 0 \
    --device cuda \
    --out_dir "$out_dir" \
    > "$LOG_ROOT/${tag}_eval.log" 2>&1
  echo "[OK] $tag 评估完成：$out_dir"
}

eval_one "$GPU0" K0 "$K0_DIR/model_epoch_100.pth" &
PID_EVAL_K0=$!
eval_one "$GPU1" K1 "$K1_DIR/model_epoch_100.pth" &
PID_EVAL_K1=$!
wait "$PID_EVAL_K0"
wait "$PID_EVAL_K1"

python -u scripts/summarize_level4_k0_k1.py \
  --eval_root "$EVAL_ROOT" \
  --seed "$SEED" \
  | tee "$LOG_ROOT/final_summary.log"

echo "========== K0/K1 TRAINING, EVALUATION AND SUMMARY COMPLETED =========="
