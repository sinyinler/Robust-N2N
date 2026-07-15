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
- 结果（v5，3 epoch，raw.npy vs reference.npy, dr=255）：**Robust-N2N 34.079 / 0.8785 / 0.9031
  vs N2N 33.921 / 0.8768 / 0.9013 → 三项全胜 (+0.16dB / +0.0017 / +0.0018)**。
  - **首个有效改动**：GIBlock 塌缩、白度更差，唯有特征一致性既不塌又略胜 N2N。
  - 训练健康：rec 0.743→0.677→0.669（正常下降）；feat≈−1.43（贴权重和 −1.5，强对齐）。
  - **警告：std1(bridge) 0.0814→0.0534→0.0375 持续下滑**（std0/out3 稳在 ~0.067）。没塌但在流失
    多样性，训久/权重大可能塌。→ 后续考虑：降 feat_w_bridge、或给 bridge 上 EMA(BYOL)、或只在 out3 做。
  - 局限：in-dist +0.16dB 幅度小；**价值仍需 OOD 验证**（raw.npy 分布归属 + mix 外 OOD 数据仍待提供）；
    细血管是否被磨需看 quad 图。

## 2026-07-06 改动：特征一致性 SimSiam → 1×1 卷积 + Charbonnier（v6，用户指定）

- 用户要求：out3/bridge 改用**单个 1×1 卷积**提取特征、跨视图损失改为 Charbonnier 差；输出层用三项
  [f(n1)−f(n2)]+[f(n1)−n2]+[f(n2)−n1]（= 对称N2N Charbonnier + 一致性γ）；总损失=charbonnier+rtv+特征损失。
- 实现：
  - `losses/feature_consistency.py`：改为每尺度一个 Conv1×1(C→dim)，
    L_feat = Σ_s w_s·Charbonnier(φ_s(feat_n1), φ_s(feat_n2))。**去掉 projector/predictor/stop-grad/负余弦**。
  - `train_robust.py`：criterion_feat 去 pred_hidden；一致性项 γ 通过 --gamma 打开（默认0.1）。
  - 总损失 = α·Charb(f(n1),n2)+β·Charb(f(n2),n1)+γ·|f(n1)−f(n2)| + λ_rtv·RTV + w_feat·L_feat。
- ⚠ 塌缩风险：同时"去掉 SimSiam 防塌机制"+"打开 γ 一致性" = 两个塌缩驱动。保留 std/diff 监控，盯早停。
- 也新增 eval_ood_robust.py（N2N vs Robust-N2N，level1 OOD 对照）。
- 结果：待训练后回填。

## 2026-07-06 修正特征损失（选项 A：归一化+stop-grad，解决量级问题）

- 诊断：v6（1×1+Charbonnier）特征损失量级 ~1e-3，w_feat=0.1 → 贡献 ~1e-4，**基本不起作用**；
  且 v6 在 raw.npy(5x5 OOD) 上 33.08 < v5(mix)34.08，但 v5 是 mix 训、可能见过 5x5，非公平基线。
- 改动 `losses/feature_consistency.py`（选项 A）：1×1 卷积提特征后 **L2 归一化 + 对称负余弦 + stop-grad**
  （单 1×1 卷积、无 predictor）。→ 尺度无关(∈[−1,1])，w_feat=0.1 生效；stop-grad 防塌。
- 公平评测：改用 eval_ood_robust.py 在 5x5 level1 OOD 上比 N2N(lv234)，不再用 raw.npy 单图。
- 结果：待 v7 训练回填。

## 2026-07-15 Masked N2N 首轮失败与实验控制修正

- 首轮配置（分支 `codex/masked-feature-prediction`，commit `337118a`）：level4 训练，3 epoch，
  crop512/batch16，seed42，mask ratio=0.25、patch=16；A=(pixel0, feature0)，
  B=(pixel1, feature0)，C=(pixel0, feature0.05)，D=(pixel1, feature0.05)。
- ID 单场景 50 帧（PSNR/MSSIM）：A=32.519/0.863，B=31.972/0.859，
  C=31.559/0.840，D=32.340/0.865。D 相对 A 为 −0.179 dB；只有 MSSIM +0.002。
- OOD level1、39 场景、level4 前50帧均值伪GT（PSNR/SSIM）：A=22.465/0.6224，
  B=21.272/0.6048，C=21.165/0.5063，D=22.153/0.6068。相对 A：
  B=−1.193 dB/−0.0177，C=−1.301 dB/−0.1161，D=−0.312 dB/−0.0157。
- 训练日志：epoch3 时 mask pixel raw loss≈0.270、主 N2N≈0.256，`w=1` 使辅助像素项与主任务等量；
  C 的 feature 实际贡献仅 `0.009346×0.05≈0.000467`，却出现明显验证/OOD退化，不能用
  “feature 权重过大”解释。
- 定位到两个实验控制缺陷：① masked forward 同样更新 student 的 BatchNorm running statistics，
  污染 all-visible 推理分布；② DataLoader shuffle/worker crop 和 mask 共用全局 RNG，predictor 初始化及
  mask 随机数会改变不同 arm 的数据轨迹，seed42 并未形成严格配对。
- 本次修正（本条记录所在提交）：masked forward 使用 batch statistics 和 BN affine 梯度，但暂停写入
  running statistics；DataLoader/worker、mask 与 predictor 初始化使用相互隔离的 seeded RNG；日志新增
  weighted RTV/pixel/feature；mask pixel pilot 权重降为0.1。旧 checkpoint 不与新结果混用，四个 arm 全部输出到
  `results/checkpoints/maskfix_*`。
- 修正后结果：待服务器重新训练并回填指标；必须同时检查
  `results/eval_ood/maskfix_D_s42/compare/` 的全图与细血管局部放大，不能只凭 PSNR/SSIM 下结论。

## 2026-07-15 Masked feature 微调阶段 0：epoch 曲线与梯度诊断

- 目的：在改变 feature weight、mask ratio 或 projector 前，先用现有 A/C checkpoint 判断 3 epoch 是否训练充分，
  并测量 N2N 与 masked feature 两项目标在 Encoder2/3 上的真实梯度强度和方向，避免只按 loss 标量猜权重。
- 本条所在提交新增 `eval_masked_epochs.py`：统一评估 A-base/C-feature 的 seed=42/187/2413、epoch=1/2/3，
  固定使用同一 level4 场景前 50 帧和同一 reference；输出逐帧 CSV、seed/epoch 汇总、跨 seed epoch 曲线、
  paired bootstrap 95% CI，以及同窗宽全图/中心局部放大。ID test 曲线只作学习过程诊断，checkpoint 选择仍以
  `history.jsonl` 的 validation loss 为先，避免用 test PSNR 直接选 epoch。
- `train_masked.py` 新增可选 `--grad_diag_every` 和 `--grad_diag_scales`。诊断通过 `torch.autograd.grad`
  临时读取梯度，不写入参数 `.grad`，记录到每次训练目录的 `grad_diagnostics.jsonl`；默认关闭，后续微调命令
  显式使用 `--grad_diag_every 100 --grad_diag_scales encoder2 encoder3`。新增
  `scripts/summarize_grad_diagnostics.py`，默认排除 warmup（`ramp<0.99`），汇总 feature/N2N 梯度范数比、
  梯度余弦、负余弦比例和强冲突（默认 cosine<-0.2）比例。
- 第一阶段服务器评估命令：

  ```bash
  python eval_masked_epochs.py \
    --checkpoint_root results/checkpoints \
    --a_dir_template 'maskfix_A_base_s{seed}' \
    --c_dir_template 'maskfix_C_feature_s{seed}' \
    --seeds 42 187 2413 \
    --epochs 1 2 3 \
    --scene_dir /mnt2/songyd/5x5/5x5x4/0/npy \
    --reference /home/songyd/Projects/Robust-N2N/reference.npy \
    --n_frames 50 \
    --max_vis_frames 1 \
    --device cuda \
    --out_dir results/eval_id/maskfix_epoch_sweep
  ```

- 结果：待服务器执行后回填 `results/eval_id/maskfix_epoch_sweep/summary.json`、`epoch_summary.csv` 和
  `compare/` 的视觉结论；确认 epoch 3 是否仍改善后，才进入第一个单变量微调。

## 2026-07-15 Masked feature 微调阶段 1：seed42、5 epoch、加入原始 N2N

- 阶段 0 结果：A/C 三 seed 在 epoch3 的 ID 配对增益为 `+0.226±0.023 dB`，三个 seed 分别
  `+0.214/+0.252/+0.212 dB`，总胜出 `127/150` 帧；跨 seed MSSIM `+0.00130`、Pearson r
  `+0.00251`。A/C 六条 validation 曲线从 epoch2 到 epoch3 全部继续下降，因此先验证 5 epoch，
  暂不调 feature weight，也不加 projector。
- 本轮只跑 seed42，但同时从头训练三组：
  1. Original：`train_n2n.py` 单通道网络、原始 N2N loss/optimizer 配方，`weight_decay=0.01`；
  2. A-base：双通道 all-visible 公平基线，`weight_decay=1e-4`；
  3. C-feature：A-base + masked feature prediction，`weight_decay=1e-4`。
- 解释边界：`C-A` 隔离 feature-loss；`C-Original` 回答相对原始系统的实际净提升；`A-Original`
  量化双通道/trainer/weight-decay 等非 feature 因素，三者不可互相替代。三组都固定 level4、crop512、
  batch16、seed42、同一独立 DataLoader RNG 和 5 epoch OneCycle；5-epoch OneCycle 会重定义整个学习率轨迹，
  因此必须从头训练，不能把旧 3-epoch checkpoint 直接续两轮。
- 实现：`train_n2n.py` 新增显式 `--weight_decay`、`--deterministic_loader_rng` 和 `history.jsonl`；
  `train_masked.py` 将既有 `1e-4` 暴露为参数但默认行为不变；`eval_masked_epochs.py` 新增可选
  `--original_dir_template`，同帧输出 Original/A/C 指标、C-A、C-Original、A-Original、bootstrap CI
  及五列同窗宽局部放大图。结果待服务器训练后回填。
