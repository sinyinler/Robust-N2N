#!/usr/bin/env bash
# Train J0/J1/J2 from the H4 configuration, evaluate Level4 scene0 first500,
# and write a final H4-vs-hybrid table. Intended to be launched under nohup.
set -euo pipefail

DATA="${DATA:-/mnt2/songyd/5x5}"
NOISE_DATA="${NOISE_DATA:-/mnt2/songyd/5x5/5x5x4}"
NOISE_STATS="${NOISE_STATS:-results/eval/noise_stats_level4_log1p.json}"
REFERENCE="${REFERENCE:-/home/songyd/Projects/Robust-N2N/reference.npy}"
SEED="${SEED:-42}"
BATCH="${BATCH:-12}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
REUSE_COMPLETED="${REUSE_COMPLETED:-0}"

CHECKPOINT_ROOT="results/checkpoints/paper_lv4_hybrid_H4_E100_v1"
LOG_ROOT="results/logs/paper_lv4_hybrid_H4_E100_v1"
EVAL_ROOT="results/eval_paper/level4_hybrid_H4_E100_v1"
H4_PARENT="results/checkpoints/paper_lv4_scenes0_29_E100_v1"

J0_DIR="${CHECKPOINT_ROOT}/J0_GammaOnly_s${SEED}"
J1_DIR="${CHECKPOINT_ROOT}/J1_PatchMix_s${SEED}"
J2_DIR="${CHECKPOINT_ROOT}/J2_SequentialEnergyMatched_s${SEED}"

mkdir -p "$(dirname "$NOISE_STATS")" "$CHECKPOINT_ROOT" "$LOG_ROOT" "$EVAL_ROOT"

if [[ ! -f "$NOISE_STATS" ]]; then
  echo "[INFO] measuring Level4 log1p noise -> $NOISE_STATS"
  python -u scripts/measure_noise.py \
    --data_path "$NOISE_DATA" \
    --intensity_transform log1p \
    --crop 512 \
    --max_frames_per_seq 12 \
    --max_seqs_per_level 40 \
    --out "$NOISE_STATS" \
    > "$LOG_ROOT/measure_noise.log" 2>&1
fi

echo "[INFO] git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[INFO] data=$DATA seed=$SEED batch=$BATCH GPUs=$GPU0,$GPU1"
echo "[INFO] noise_stats=$NOISE_STATS"

python -m unittest \
  tests.test_hybrid_corruption \
  tests.test_gamma_corruption \
  tests.test_masked_resume \
  > "$LOG_ROOT/unit_tests.log" 2>&1
echo "[OK] hybrid/resume unit tests passed on server"

CUDA_VISIBLE_DEVICES="$GPU0" python -u scripts/smoke_hybrid_feature.py \
  > "$LOG_ROOT/smoke_hybrid.log" 2>&1
echo "[OK] hybrid CUDA smoke passed"

train_one() {
  local gpu="$1"
  local tag="$2"
  local mode="$3"
  local save_dir="$4"
  local gamma_min="$5"
  local gamma_max="$6"
  local gaussian_min_scale="$7"
  local gaussian_max_scale="$8"

  if [[ -f "$save_dir/model_epoch_100.pth" && "$REUSE_COMPLETED" == "1" ]]; then
    echo "[SKIP] completed training exists: $save_dir/model_epoch_100.pth"
    return 0
  fi
  if [[ -e "$save_dir" ]]; then
    echo "[ERROR] output exists but reuse is disabled/incomplete: $save_dir"
    return 2
  fi

  mkdir -p "$save_dir"
  echo "[TRAIN] $tag mode=$mode physical_gpu=$gpu -> $save_dir"

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
    --corruption_mode "$mode" \
    --mask_ratio 0.25 \
    --mask_patch 8 \
    --noise_stats_json "$NOISE_STATS" \
    --noise_sigma_min_scale "$gaussian_min_scale" \
    --noise_sigma_max_scale "$gaussian_max_scale" \
    --gamma_cv_min "$gamma_min" \
    --gamma_cv_max "$gamma_max" \
    --hybrid_gamma_probability 0.5 \
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
    --progress 0 \
    --seed "$SEED" \
    --device cuda \
    --save_dir "$save_dir" \
    > "$LOG_ROOT/${tag}_train.log" 2>&1

  echo "[OK] $tag training completed"
}

# Wave 1: Gamma-only and patchwise mixture in parallel.
train_one "$GPU0" J0 gamma "$J0_DIR" \
  0.025 0.075 0.25 0.75 &
PID_J0=$!

train_one "$GPU1" J1 hybrid_mixture "$J1_DIR" \
  0.025 0.075 0.25 0.75 &
PID_J1=$!

wait "$PID_J0"
wait "$PID_J1"

# Wave 2: same-patch sequential perturbation. Both standard-deviation-like
# strengths are multiplied by 1/sqrt(2) to avoid doubling the noise budget.
train_one "$GPU0" J2 hybrid_sequential "$J2_DIR" \
  0.01767766952966369 0.053033008588991064 \
  0.1767766952966369 0.5303300858899106

mapfile -t H4_DIRS < <(
  find "$H4_PARENT" -mindepth 1 -maxdepth 1 -type d -name "H4*s${SEED}*" | sort
)
if [[ "${#H4_DIRS[@]}" -ne 1 ]]; then
  echo "[ERROR] expected one H4 directory, found ${#H4_DIRS[@]}"
  printf "  %s\n" "${H4_DIRS[@]:-<none>}"
  exit 3
fi
H4_CHECKPOINT="${H4_DIRS[0]}/model_epoch_100.pth"
if [[ ! -f "$H4_CHECKPOINT" ]]; then
  echo "[ERROR] H4 checkpoint not found: $H4_CHECKPOINT"
  exit 3
fi

eval_one() {
  local gpu="$1"
  local tag="$2"
  local checkpoint="$3"
  local out_dir="$EVAL_ROOT/level4_seen_${tag}_vs_H4_s${SEED}"

  if [[ -f "$out_dir/per_frame.csv" && "$REUSE_COMPLETED" == "1" ]]; then
    echo "[SKIP] completed evaluation exists: $out_dir/per_frame.csv"
    return 0
  fi
  echo "[EVAL] $tag vs H4 physical_gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -u eval_curve.py \
    --checkpoint "$checkpoint" \
    --masked_model 0 \
    --baseline_checkpoint "$H4_CHECKPOINT" \
    --baseline_masked_model 0 \
    --scene_dir /mnt2/songyd/5x5/5x5x4/0/npy \
    --n_frames 500 \
    --reference "$REFERENCE" \
    --max 255 \
    --device cuda \
    --out_dir "$out_dir" \
    > "$LOG_ROOT/${tag}_eval.log" 2>&1
  echo "[OK] $tag evaluation completed"
}

eval_one "$GPU0" J0 "$J0_DIR/model_epoch_100.pth" &
PID_E0=$!
eval_one "$GPU1" J1 "$J1_DIR/model_epoch_100.pth" &
PID_E1=$!
wait "$PID_E0"
wait "$PID_E1"
eval_one "$GPU0" J2 "$J2_DIR/model_epoch_100.pth"

python -u scripts/summarize_level4_hybrid.py \
  --eval_root "$EVAL_ROOT" \
  --seed "$SEED" \
  | tee "$LOG_ROOT/final_summary.log"

echo "========== ALL J0/J1/J2 TRAINING AND LEVEL4 EVALUATION COMPLETED =========="
