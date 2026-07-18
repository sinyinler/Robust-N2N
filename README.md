# NTN: N2N-bootstrap Noise Translation Network

本项目复刻并改造 `Learning to Translate Noise for Robust Image Denoising`
的 Noise Translation Network 思路，用现有 Noise2Noise 数据条件实现无真值
GT 的 NTN 训练。

## 核心改造

原论文训练 `T` 时使用 noisy-clean pair。本项目没有真 GT，因此采用
`N2N-bootstrap` 版本：

- `I1`：第一路 noisy observation，输入 Noise Translator `T`。
- `I2`：同场景第二路独立 noisy observation，替代 clean GT 参与 implicit loss。
- `C_hat`：伪干净图，默认来自多帧均值，也可由已训练 N2N checkpoint 生成。
- `D_prime`：先用 `C_hat + synthetic Gaussian noise -> C_hat` 训练出的 Gaussian expert。
- `T`：把真实噪声翻译为更接近 Gaussian 的噪声，再交给 `D_prime` 去噪。

## 项目结构

```text
NTN/
├── models/              # Denoiser 复制件、NoiseTranslator、GIBlock
├── data/                # N2N-bootstrap 三元组数据集
├── losses/              # Charbonnier 与 NTN explicit loss
├── utils/               # VST、IO、checkpoint、metrics
├── configs/             # 默认实验配置
├── scripts/             # smoke test 等辅助脚本
├── results/             # checkpoint、图像、评估输出
├── train_gaussian_expert.py
├── train_translator.py
├── inference_ntn.py
├── eval.py
├── experiment_log.md
└── README.md
```

## 泛化对照实验协议（重要）

为了公平地证明「NTN 比普通 N2N 更泛化」，**N2N 基线必须和 NTN 用完全相同的训练数据**
（level 2/3/4），把最噪的 **level1 留作 OOD 测试**。这个 N2N 同时担任两个角色：
(1) NTN 训练时的 Ĉ 生成器（`--bootstrap_checkpoint`）；(2) 对照基线本身。

> 注意：原有的 N2N checkpoint 是在**全部层级（含 level1）**上用跨层级配对训练的，
> 既不能当 OOD 基线，也会把 level1 信息泄漏进 NTN，**必须重训**。

### Step 0：重训 N2N（仅 level 2/3/4）

`train_n2n.py` 保留原训练策略：`AdamW` + `OneCycleLR`（warmup→cosine）。三个训练入口在
多 GPU 时默认启用 `DataParallel`，强制单卡加 `--data_parallel 0`。

```bash
python train_n2n.py \
  --data_path /mnt2/songyd/5x5 --data_subdir npy --strict_data_subdir 1 \
  --levels 2 3 4 \
  --intensity_transform log1p --crop_size 512 --batch_size 48 \
  --epochs 5 --lr 0.01 --lr_final 0.0005 --warmup_pct 0.1 \
  --save_dir results/checkpoints/n2n_lv234
```

之后所有命令里的 `<N2N_CKPT>` 即 `results/checkpoints/n2n_lv234/model_epoch_5.pth`。

先训练 Gaussian expert `D_prime`（盲高斯专家，Ĉ=N2N(I1)，σ 覆盖实测真实噪声跨度；
用 `--levels 2 3 4` 把最噪的 level1 留作 OOD 测试）：

```bash
python train_gaussian_expert.py \
  --data_path /mnt2/songyd/5x5 --data_subdirs npy --strict_data_subdir 1 \
  --levels 2 3 4 \
  --bootstrap_checkpoint results/checkpoints/n2n_lv234/model_epoch_5.pth \
  --intensity_transform log1p \
  --sigma_min 0.08 --sigma_max 0.6 \
  --epochs 5
```

再冻结 `D_prime` 训练 Noise Translator `T`（implicit/explicit 同锚 Ĉ=N2N(I1)，
explicit 后半段才启用）：

```bash
python train_translator.py \
  --data_path /mnt2/songyd/5x5 --data_subdirs npy --strict_data_subdir 1 \
  --levels 2 3 4 \
  --bootstrap_checkpoint results/checkpoints/n2n_lv234/model_epoch_5.pth \
  --gaussian_expert_checkpoint results/checkpoints/gaussian_expert/gaussian_expert_epoch_5.pth \
  --intensity_transform log1p \
  --implicit_target pseudo_clean \
  --alpha 0.05 --beta 0.002 --explicit_start_frac 0.5 \
  --epochs 5
```

> 噪声水平先用 `python scripts/measure_noise.py --data_path /mnt2/songyd/5x5 --data_subdirs npy`
> 实测确定。当前 σ 区间 `[0.08, 0.6]` 即依据 log1p 域实测（level4≈0.10 ~ level1≈0.43）。

推理：

查看单独 N2N baseline 效果：

```powershell
python inference_n2n.py `
  --input /mnt2/songyd/5x5/5x5x1/0/npy `
  --checkpoint results\checkpoints\n2n_5x5_log1p\model_epoch_5.pth `
  --intensity_transform log1p `
  --limit 20 `
  --out_dir results\images\n2n_5x5_train_preview
```

```powershell
python inference_ntn.py `
  --input path\to\noisy.npy `
  --translator_checkpoint results\checkpoints\translator\translator_epoch_5.pth `
  --gaussian_expert_checkpoint results\checkpoints\gaussian_expert\gaussian_expert_epoch_5.pth `
  --out_dir results\images
```

评估与出图：

```powershell
python eval.py `
  --noisy path\to\noisy.npy `
  --denoised results\images\noisy_ntn.npy `
  --reference path\to\pseudo_or_clean.npy `
  --out_dir results\eval
```

## 复现实验注意

`experiment_log.md` 必须记录每次实验的配置、checkpoint、指标和视觉判断。
去噪任务不能只看 PSNR/SSIM；需要同时检查血管结构是否光滑、细小血管是否被磨平。
# SIDD sRGB 监督基线（独立路径）

SIDD 代码不会修改原 BFI/N2N 的单通道数据和模型路径。默认实验使用
SIDD-Small 的 scene-disjoint 划分：训练 `001-006,009,010`（120 对），
验证 `007`（20 对），内部测试 `008`（20 对）。输入和目标分别是
`NOISY_SRGB_010.PNG` 与 `GT_SRGB_010.PNG`，直接归一化到 `[0,1]`，不使用
`log1p/expm1`、RTV 或 feature loss。

本机已配置的 Conda 环境运行方式：

```powershell
# GPU/data/反向传播 smoke（只跑 2 step，写入独立输出目录）
D:\Anaconda\envs\denoise\python.exe train_sidd.py `
  --config configs/sidd_supervised.json `
  --epochs 1 --smoke_steps 2 `
  --out_dir results/sidd/smoke_s42

# 20 epoch 监督基线
D:\Anaconda\envs\denoise\python.exe train_sidd.py `
  --config configs/sidd_supervised.json

# scene 008 完整图 tiled inference（512 tile / 64 overlap）
D:\Anaconda\envs\denoise\python.exe eval_sidd.py `
  --data_root "D:\Desktop\数据集\SIDD\SIDD_Small_sRGB_Only" `
  --checkpoint results/sidd/supervised_charbonnier_s42/best.pt `
  --scenes 008 `
  --out_dir results/sidd/internal_test_scene008
```

公开 SIDD Validation blocks 的本地评测：

```powershell
# 无 checkpoint 时先计算 noisy baseline，并核验 MAT 变量和维度
D:\Anaconda\envs\denoise\python.exe eval_sidd_blocks.py `
  --noisy_mat "D:\Desktop\数据集\SIDD\Validation\ValidationNoisyBlocksSrgb.mat" `
  --gt_mat "D:\Desktop\数据集\SIDD\Validation\ValidationGtBlocksSrgb.mat" `
  --out_dir results/sidd/validation_noisy_baseline

# 训练后计算模型结果；--save_mat 可额外生成 Idenoised.mat
D:\Anaconda\envs\denoise\python.exe eval_sidd_blocks.py `
  --noisy_mat "D:\Desktop\数据集\SIDD\Validation\ValidationNoisyBlocksSrgb.mat" `
  --gt_mat "D:\Desktop\数据集\SIDD\Validation\ValidationGtBlocksSrgb.mat" `
  --checkpoint results/sidd/supervised_charbonnier_s42/best.pt `
  --out_dir results/sidd/validation_trained --save_mat
```

主要文件：

- `data/sidd_dataset.py`：RGB pair 发现、scene 过滤、同步 crop/增强；
- `models/sidd_rgb_denoiser.py`：不压灰度的 3→3 通道轻量 U-Net；
- `train_sidd.py`：纯 Charbonnier 监督训练与 best/last checkpoint；
- `eval_sidd.py`：内部完整图 tiled inference、逐图指标和可视化；
- `eval_sidd_blocks.py`：公开 Validation MAT 的 1280-block 标准评测。
