# Robust-N2N Experiment Log

本项目从 `D:\Desktop\NTN` 复制源码而来（不改动 NTN 原项目），目标：把原先"两步训练"
（① 训 N2N → 得伪干净 Ĉ；② 再训翻译器 T + 高斯去噪器 D′）**合并为一步端到端训练**，
用"轻量 U-Net + GIBlock + 组合损失"直接得到一个**鲁棒、可泛化**的去噪器 f。

## 2026-07-02 项目初始化（从 NTN 复制）

- 改动：新建独立项目 `D:\Desktop\Robust-N2N`，从 NTN 复制 models/data/losses/utils/scripts/configs
  与各训练/评估脚本；不携带 .git、pptx、ppt_assets、results。
- 核心方案（一步法）：
  1. 网络 f = 轻量 U-Net（原 `models/denoiser.py`：LRB + ELA）**内嵌 GIBlock**（高斯注入）；
  2. 损失从原 `Charbonnier + RTV` 扩展为：
     `L = α·Charb(f(n1),n2) + β·Charb(f(n2),n1) + γ·‖f(n1)−f(n2)‖₁
          + σ·L_spatial(w) + δ·L_freq(w) + λ·RTV(f(n1))`，其中 `w = f(n1)−f(n2)`；
     - α/β = 对称 N2N（Charbonnier 已并入，不再单列）；
     - γ = 一致性项（依据 DenoiseGAN, Jiang 2026 的 self-constraint loss 第三项）；
     - σ/δ = 残差白度正则（复用 `losses/ntn_losses.py` 的 spatial_gaussian / frequency_rayleigh，
       作用于 w=f(n1)−f(n2)；该量信号无关(y 相消)，不误伤血管；方向：惩罚残差里的相关结构）；
       explicit 项延迟到训练过半再开。
  3. GIBlock：训练时注入高斯作随机正则，**推理时关闭**（已在 `models/ntn.py` 的
     `GaussianInjectionBlock.forward` 加 `not self.training` gating）。
- 叙事（无翻译器）：改用"**残差白度自正则（residual-whiteness regularization）**"——理想去噪器
  对同场景两次独立观测的两个估计应仅相差白高斯残差；对 f(n1)−f(n2) 施加白度约束驱动去噪器
  彻底去除空间相关散斑噪声，获得跨条件鲁棒泛化。
- 结果：待训练后回填（指标 + 血管局部放大图）。

## 2026-07-02 落地一步法（模型 + 损失 + 训练脚本）

用户拍板：① GIBlock 插 3 处（瓶颈 + 最深编码块 LRB3 + 第一个解码块 decoder_1）；
② 权重按推荐（α=β=1, γ=0.1, w_white=0.05, beta_freq=2e-3, RTV=0.01），白度延迟到训练过半开；
③ 数据 = `/mnt2/songyd/mix`（334 个文件夹，npy 与 lbf 两类，**文件夹内** Δ∈{5,7,9} 配对），**全部拿来训练**（不在 mix 内留 OOD，OOD 用外部数据如脚/3×3 另测）。

- 新增文件：
  - `models/robust_denoiser.py`：RobustDenoiser = 复用 Encoder/Bridge/Decoder/Transformer_unit +
    在 out3(64,@H/4)、bridge(80,@H/8)、decoder_1(64,@H/4) 后各插 1 个 GaussianInjectionBlock。
  - `losses/robust_n2n_loss.py`：RobustN2NLoss = α·Charb(f(n1),n2)+β·Charb(f(n2),n1)+γ·|f(n1)−f(n2)|
    + λ·RTV(f(n1)) + [过半后] w_white·(spatial+beta_freq·freq)，白度作用于 w=f(n1)−f(n2)。
  - `train_robust.py`：复用 train_n2n 的 build_loaders/set_seed/build_onecycle；每 batch 对 n1、n2
    各前向一次；白度在 global_step≥whiten_start_frac·total_steps 后开；GIBlock 训练注入/推理关闭。
- GIBlock gating：`models/ntn.py` GaussianInjectionBlock 只在 self.training 时注入（推理确定性）。
- 校验：三文件 py_compile 通过（本机无 torch，前向 smoke 需在服务器跑）。git commit dfd21f7。
- 注意：对 n1/n2 各前向一次 → 显存约 2×，故 batch_size 默认降到 24（原 N2N 48），按显存再调。
- 待办：服务器 smoke（确认前向/推理确定性/参数量）→ 整体训练 → 三组 OOD 评测回填结果。
