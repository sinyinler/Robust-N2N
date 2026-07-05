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

## 2026-07-04 结果（一）：一步法首训 → 模型塌缩成常数输出（失败）

- 配置：全量 mix（发现 5914 batch/epoch），5 epoch，crop512/batch24，log1p，lr0.01，
  α=β=1, γ=0.1, w_white=0.05(过半开), RTV=0.01, GIBlock inject_sigma=1.0。commit dfd21f7。
- 推理(raw.npy vs reference.npy, dr=255)：**Robust-N2N 15.50 / 0.435 / r=−0.00**；
  N2N(ours.npy) 33.92 / 0.877 / 0.901。→ Robust-N2N **r≈0 = 输出与内容无关**。
- 训练日志铁证：epoch2 起 **cons=0、rtv=0、rec 卡在 2.298**（常数图的 Charbonnier 地板）
  → 网络**无视输入、输出一张常数图**（经典 self-supervised collapse）。
- 根因：**一致性项 γ|f(n1)−f(n2)| 的平凡最优解=常数**；GIBlock 训练注入又要求"对输入不变"
  → 合力把模型推进常数陷阱。塌缩发生在 epoch2（白度 epoch3 才开）→ **白度不是元凶**。
  （教训：DenoiseGAN 的一致性项靠对抗+cycle 防塌缩；我们纯 N2N+一致性没有保险，露出 collapse。）
- 处置：① 损失加未加权诊断量 diff=mean|f(n1)−f(n2)|（γ=0 时也能看塌缩）；
  ② 下一版**去掉一致性(γ=0)**、GIBlock 注入调小(inject_sigma=0.3)、保留白度，3 epoch 验；
     N2N+RTV 已是验证过的好基线(=ours.npy 33.9dB)，不再重训对照。
- 结果：见下"结果（二）"。

## 2026-07-04/05 结果（二）：塌缩定位 + 白度暂不成立（归档，暂停白度）

在 raw.npy 上对 reference.npy 评测（dr=255，PSNR/MSSIM/r）：

| 版本 | 配置 | 结果 |
|---|---|---|
| v1 | γ=0.1, inject=1.0, w_white=0.05 (5ep) | 15.50 / 0.435 / −0.00 **塌缩(常数)** |
| v2 | γ=0, inject=0.3, w_white=0.05 (3ep) | epoch1 后半 diff→0 **塌缩** |
| v3 | γ=0, inject=0, w_white=0 (1ep) | **33.25 / 0.861 / 0.883**（≈N2N，健康）|
| v4 | γ=0, inject=0, w_white=0.05 (3ep) | 32.91 / 0.856 / 0.882（比 v3/N2N 略低）|
| — | N2N(ours.npy) baseline | 33.92 / 0.877 / 0.901 |

- **塌缩元凶 = GIBlock 训练期注入**（BN+注入不兼容）：v2 关一致性仍塌、v3 关注入立刻健康 → 一锤定音。
- **对称 N2N + RTV + NAFBlock 架构都没问题**（v3 只 1 epoch 就 33.25 ≈ N2N）。
- **白度（作用于 f(n1)−f(n2)）在 in-dist 上没帮忙、反而略降**（v4 32.91 < v3 33.25 < N2N）。
  训练日志 `freq` 45→89 翻倍增长而 |w| 稳定 → 去噪器越好、残差 w 越"有结构"；白度硬掰、和"保血管"打架。
  隐患：白度前提"好去噪器残差应白"对**有结构图像不成立**（误差集中在边缘/血管、本就不白）。
- **决策**：暂停白度（未在 OOD 上验证前不下最终定论，但 in-dist 无收益 + 理论隐患 → 信心下调）。
  OOD 数据 + raw.npy 分布归属仍待用户提供。
- **下一方向（用户指定）**：删掉 GIBlock，改测 **JEPA / 跨视图特征一致性**（编码器多尺度特征上的
  自监督一致性，用 n1/n2 作两视图）。实现方案见项目讨论（先对齐后编码）。

## 2026-07-05 落地跨视图特征一致性（SimSiam 式，删 GIBlock）

用户拍板：深层 out3+bridge 两尺度、projector 用两层 1×1(BN+ReLU)、SimSiam(stop-grad+predictor,
无EMA)、w_feat=0.1、深重浅轻。删掉 GIBlock。

- 新增 `models/denoiser_feats.py`：`DenoiserWithFeats` = 原轻量 U-Net(无 GIBlock)，forward 可返回
  编码器深层特征 [out3(64,@H/4), bridge(80,@H/8)]（浅层不返回，避免强制不变性磨细血管）。
- 新增 `losses/feature_consistency.py`：`FeatureConsistencyLoss`（SimSiam 式，多尺度）：
  每尺度 projector g(两层1×1+BN)、predictor h(1×1 bottleneck)，逐像素负余弦，
  `0.5·D(h(g(f1)),sg(g(f2))) + 0.5·D(h(g(f2)),sg(g(f1)))`，深权重>浅权重。
  防塌缩靠 **stop-gradient + predictor**（+重建损失拉住编码器）。附**塌缩监控**：归一化投影特征
  逐维 std（健康≈1/√dim，塌→0），随训练打印。
- `train_robust.py`：改用 DenoiserWithFeats（去 GIBlock），对 n1/n2 各 return_feats 前向，
  总损失 = 对称N2N(Charbonnier) + RTV + w_feat·特征一致性；新增 --w_feat/--feat_dim/
  --feat_pred_hidden/--feat_w_out3/--feat_w_bridge；进度条/汇总打印 feat 与 std0/std1。
- 校验：三文件 py_compile 通过（本机无 torch，前向 smoke 服务器跑）。
- 预期：in-dist 大概率 ≈ N2N；价值(若有)在 OOD 编码器噪声不变性。仍需 OOD 数据 + raw.npy 分布归属。
- 结果：待训练后回填（重点盯 std 别→0、细血管别被磨）。
