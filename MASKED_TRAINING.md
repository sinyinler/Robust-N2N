# Masked N2N + Masked Feature Prediction

This path replaces the old full-image feature consistency objective with two
mask-conditioned objectives:

```text
L = Charb(student(n1), n2)
  + w_mask_pixel * Charb(student(mask(n1)), n2) on hidden pixels only
  + w_mask_feature * cosine(student features, EMA-teacher(n2) features)
                         on hidden locations only
  + 0.01 * RTV(student(n1))
```

The model receives image + visibility-mask channels. At inference the wrapper
automatically supplies an all-visible mask, so evaluation still calls
`model(image)`.

## 1. Get the branch on the training server

```bash
cd /home/songyd/Projects/Robust-N2N
git fetch origin
git switch --track origin/codex/masked-feature-prediction
```

If the branch already exists locally:

```bash
git switch codex/masked-feature-prediction
git pull --ff-only
```

## 2. Mandatory smoke test

```bash
python scripts/smoke_masked.py --device cuda
```

Expected final line starts with `[OK] masked smoke`.

## 3. Four-arm clean ablation (seed 42 pilot)

All four arms use the same two-channel architecture. Arm A is therefore the
correct baseline; do not compare B/C/D only against an older one-channel
checkpoint.

```bash
DATA=/mnt2/songyd/5x5
COMMON="--data_path $DATA --levels 4 --epochs 3 --crop_size 512 --batch_size 16 --lr 0.01 --rtv_weight 0.01 --mask_ratio 0.25 --mask_patch 16 --seed 42"

# A: fair two-channel N2N baseline
python train_masked.py $COMMON \
  --w_mask_pixel 0 --w_mask_feature 0 \
  --save_dir results/checkpoints/mask_A_base_s42

# B: masked-pixel reconstruction only
python train_masked.py $COMMON \
  --w_mask_pixel 1 --w_mask_feature 0 \
  --save_dir results/checkpoints/mask_B_pixel_s42

# C: masked-feature prediction only
python train_masked.py $COMMON \
  --w_mask_pixel 0 --w_mask_feature 0.05 \
  --save_dir results/checkpoints/mask_C_feature_s42

# D: full masked pixel + feature objective
python train_masked.py $COMMON \
  --w_mask_pixel 1 --w_mask_feature 0.05 \
  --save_dir results/checkpoints/mask_D_full_s42
```

Each directory contains `run_config.json`, `history.jsonl`, and one checkpoint
per epoch. The default masked feature scales are encoder2 + encoder3. The EMA
teacher and feature predictor are stored in the checkpoint, but inference loads
only its `model` entry.

## 4. In-distribution paired evaluation

```bash
python eval_curve.py \
  --checkpoint results/checkpoints/mask_D_full_s42/model_epoch_3.pth \
  --masked_model 1 \
  --baseline_checkpoint results/checkpoints/mask_A_base_s42/model_epoch_3.pth \
  --baseline_masked_model 1 \
  --scene_dir /mnt2/songyd/5x5/5x5x4/0/npy \
  --n_frames 50 \
  --reference /home/songyd/Projects/Robust-N2N/reference.npy \
  --out_dir results/eval_curve/mask_D_vs_A_s42
```

Repeat with B and C as `--checkpoint` to isolate each objective.

## 5. Level-1 OOD evaluation

```bash
python eval_ood_robust.py \
  --data_path /mnt2/songyd/5x5 \
  --eval_level 1 --gt_level 4 --max_frames_per_scene 3 \
  --n2n_checkpoint results/checkpoints/mask_A_base_s42/model_epoch_3.pth \
  --n2n_masked_model 1 \
  --robust_checkpoint results/checkpoints/mask_D_full_s42/model_epoch_3.pth \
  --masked_model 1 \
  --out_dir results/eval_ood/mask_D_s42
```

This is the strict architecture-controlled OOD comparison: both checkpoints
use the same image+mask input architecture, and Arm A receives an all-visible
mask at inference. To compare against the historical one-channel N2N instead,
omit `--n2n_masked_model 1` and pass its old checkpoint.

## 6. Three-seed confirmation

Only run this after the seed-42 pilot passes the smoke test and produces finite
losses:

```bash
for SEED in 42 187 2413; do
  for ARM in A B C D; do
    case $ARM in
      A) PIX=0; FEAT=0 ;;
      B) PIX=1; FEAT=0 ;;
      C) PIX=0; FEAT=0.05 ;;
      D) PIX=1; FEAT=0.05 ;;
    esac
    python train_masked.py \
      --data_path /mnt2/songyd/5x5 --levels 4 \
      --epochs 3 --crop_size 512 --batch_size 16 --lr 0.01 \
      --rtv_weight 0.01 --mask_ratio 0.25 --mask_patch 16 \
      --w_mask_pixel $PIX --w_mask_feature $FEAT --seed $SEED \
      --save_dir results/checkpoints/mask_${ARM}_s${SEED}
  done
done
```

The `mask_ratio=0.25, mask_patch=16` values are engineering pilot defaults, not
final scientific choices. Select the final patch size from measured spatial
noise autocorrelation and repeat at least the winning arm with multiple scenes.
