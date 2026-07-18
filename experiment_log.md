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

## 2026-07-16 Masked feature 微调阶段 2：100 epoch 收敛曲线与 feature weight=0.10

- 多 seed 5-epoch 结果表明：`w_mask_feature=0.05` 在 epoch4 相对 A-base 的 level1 OOD PSNR
  三个 seed 均提升且逐场景 bootstrap CI 均大于零；到 epoch5 后独立增益减弱。下一项保持数据、网络、
  optimizer 和其他 loss 不变，只把 feature weight 从 0.05 调到 0.10，并与原始单通道 N2N 同时从头
  训练 100 epoch，观察辅助约束能否在训练后期维持作用。
- 新增 `utils/training_curves.py`。两个训练入口默认每个 epoch 从 `history.jsonl` 自动刷新
  `loss_curve.png` 和 `loss_history.csv`。原始 N2N 画可直接比较的 train/validation loss；masked
  模型同时画 train total、validation reconstruction、可比的 train reconstruction（N2N+RTV）以及
  weighted feature/RTV 等分量，避免把含辅助项的 train total 与不含辅助项的 validation 误当成同一目标。
- 两个100-epoch进程分别固定到两张24GB GPU，并显式关闭 DataParallel；batch size 均为16。100-epoch
  OneCycleLR 会重定义完整学习率轨迹，因此两组都必须从头训练，不能续接5-epoch checkpoint。
- 新增 `scripts/run_e100_original_feature010.sh`，默认 seed42，分别绑定物理 GPU0/GPU1 并行启动；
  独立保存 stdout 日志、PID、checkpoint 和 loss 曲线。脚本拒绝复用已有输出目录，避免重复运行时把
  不同训练轨迹追加进同一 `history.jsonl`。这两组只能衡量 C-system 相对 Original 的净变化；若要严格
  隔离 `0.10-0.05` 或 feature 本身，仍需同调度的 `w=0.05` 或 A-base 对照。
- 首次双 GPU 启动时 feature 组 batch16 在 GPU1 OOM。100 epoch 与 feature loss 的标量权重不改变单步
  激活规模，因此先检查 GPU1 占用，并将 feature 默认 batch 降为12；原始 N2N 保持16。启动脚本新增
  `N2N_BATCH`/`FEATURE_BATCH` 环境变量，checkpoint 和日志目录显式包含 batch，避免混淆失败的 batch16
  轨迹与重新从头训练的 batch12 轨迹。
- 公平性修正：Original 和 feature 两组必须使用相同 batch。最终协议将两组默认值都设为12，并废弃已启动
  的 Original batch16 轨迹；两组从 epoch1 重新训练。原因不仅是单批梯度统计不同，batch 还会改变每个
  epoch 的 optimizer step 数和100-epoch OneCycleLR 的完整轨迹，不能从 batch16 checkpoint 续训。

## 2026-07-17 局部 Gaussian feature：避免长期硬掩码产生假血管

- 现象与假设：100-epoch hard-mask feature 模型在背景产生微小假血管。16×16 block 完全置零会把辅助任务
  变成局部 inpainting；长期优化可能过度学习“沿上下文补出连续血管”。本轮只替换辅助分支的 corruption，
  验证保留像素证据能否降低这种结构幻觉。
- 实现：`train_masked.py` 新增向后兼容的 `--corruption_mode gaussian`。仍随机选取25%的16×16区域，但在
  `log1p` 域只对选中区域加入独立 Gaussian noise，不再置零；区域图只负责扰动和圈定 encoder2/encoder3
  feature loss，**不作为网络输入**。Gaussian 模式使用 `DenoiserWithFeats(input_channels=1)`，projector/
  predictor 和 EMA teacher 仍只存在于训练期，checkpoint 可按原始单通道 N2N 结构推理。
- 噪声标定：新增训练参数可读取 `scripts/measure_noise.py` 的 `recommended_sigma`；每张图独立采样
  `sigma ~ Uniform(0.25*sigma_real, 0.75*sigma_real)`，并使用独立 noise RNG，避免改变 region/DataLoader
  随机轨迹。辅助 forward 继续冻结 BN running statistics，只允许正常 N2N 分支维护推理统计。
- 实验控制：新增 `scripts/run_e100_noise_feature010.sh`，固定 level4、crop512、batch12、seed42、100 epoch、
  `w_feature=0.10`、weight decay=1e-4，与既有 Mask-C 配方一致；Original/Mask-C 的既有 E100 结果可以复用。
  对比必须同时报告 best-validation 与 epoch100 的 ID/OOD PSNR、SSIM，以及假血管背景区域局部放大和
  difference map。结果待服务器训练后回填，不能仅凭 loss 或平均 PSNR 判断是否消除假血管。
# 2026-07-18 SIDD-Small sRGB 监督基线与公开 Validation 管线

- 目的：在不下载 SIDD-Full 的前提下，用本地完整的 `SIDD_Small_sRGB_Only` 建立真实噪声
  `NOISY -> GT` 监督基线，并为公开 SIDD Validation blocks 建立可复查的外部评测。
- 数据核验：160 个完整 RGB pair、320 张 PNG、NOISY/GT 尺寸不匹配为 0。按 scene 隔离：
  train=`001-006,009,010`（120 对），validation=`007`（20 对），internal test=`008`（20 对）。
- 实现：新增独立 `data/sidd_dataset.py`、`models/sidd_rgb_denoiser.py`、`train_sidd.py`、
  `eval_sidd.py`、`eval_sidd_blocks.py` 和 `configs/sidd_supervised.json`；原单通道 BFI/N2N
  路径不改。RGB 网络不压灰度，输入/输出均为 3 通道，共 67,614 个参数；训练域为 `[0,1]`
  sRGB，首轮只用 Charbonnier，RTV/feature loss 均为 0。
- 默认训练配置：crop256；一次大图解码取 4 个同步增强 crop；image batch=4、有效 patch batch=16；
  AdamW，lr=`1e-3 -> 1e-5` cosine，weight decay=`1e-4`，20 epoch，seed42，AMP。
- Validation 下载：York 官方主机连接超时，公开 Google Drive `test.zip` 触发下载配额；改用
  Hugging Face `talib-sid/sidd-val` 中由原 MAT 无损导出的 1280 对 PNG LMDB，按原 key 顺序
  重建两个标准 MAT。两者变量均为 `uint8 (40,32,256,256,3)`。中转 LMDB 验证后已删除
  （315,785,044 bytes），只保留：
  - `D:/Desktop/数据集/SIDD/Validation/ValidationNoisyBlocksSrgb.mat`：229,019,520 bytes，
    local SHA256 `5A9F84EA873A3B347103740CDE7DCEA9BF3F1012FB75828A464DAB8E868AA02C`；
  - `D:/Desktop/数据集/SIDD/Validation/ValidationGtBlocksSrgb.mat`：229,831,366 bytes，
    local SHA256 `D208846192DCD219B1DD17A46CC2CF931449A26CD17019E48DCE79B1CA67914F`。
  MAT 容器由 SciPy 重建，压缩字节数/哈希不等同于官方 MATLAB 容器，但数组来自无损 PNG blocks。
- 校验：PyTorch 2.8/CUDA 12.6/RTX 3060 上数据读取、一次 forward/backward、checkpoint 严格加载和
  700x900 tiled inference 均通过；输出 shape 正确且无 NaN/Inf。两 step CLI smoke 的 train/val
  Charbonnier 为 `0.25408/0.29063`，只用于程序校验，不作为实验结果。
- 公开 Validation noisy baseline（1280 blocks）：RGB PSNR=`23.66238 dB`，SSIM=`0.333469`。
  SSIM 明确采用 11x11 Gaussian、sigma=1.5、`data_range=1`；逐块 CSV、summary 和 5 张视觉图位于
  `results/sidd/validation_noisy_baseline/`。这一定义用于本地一致对照；最终官方 benchmark 仍以
  Kaggle 返回数值为准。
- 当前结论：代码、数据、指标和可视化链路已打通；尚未把 2-step smoke 冒充去噪结果。下一步是
  从头运行 20 epoch baseline，再用 `best.pt` 同时评估 scene008 完整图和公开 Validation blocks。

## 2026-07-18 SIDD 监督基线训练完成与双重评估

- 训练完成：20 epoch、4800 optimizer steps；train Charbonnier 从 `0.140206` 降到 `0.013278`，validation
  从 `0.069786` 降到 `0.009648`。最佳 checkpoint 为 epoch 19，validation=`0.009596`；epoch 20 仅轻微回升，
  没有明显过拟合。checkpoint 与曲线位于 `results/sidd/supervised_charbonnier_s42/`。
- 公开 SIDD Validation blocks（1280 blocks，和 noisy baseline 使用完全相同的本地 RGB 指标定义）：
  noisy=`23.66238 dB / 0.333469 SSIM`，模型=`35.17941 dB / 0.845399 SSIM`，增益
  `+11.51703 dB / +0.511930`；paired bootstrap 95% CI 分别为 `[11.38497, 11.64759] dB` 和
  `[0.505619, 0.518158]`。PSNR 胜出 `1280/1280` blocks，SSIM 胜出 `1279/1280` blocks。
  结果位于 `results/sidd/validation_trained/`。这属于公开 Validation 本地评估，不是隐藏 GT 的官方 benchmark 分数。
- scene-disjoint internal test（scene 008，20 张完整高分辨率图，tile 512/overlap 64）：
  noisy=`26.46744 dB / 0.555477 SSIM`，模型=`31.83252 dB / 0.786656 SSIM`，增益
  `+5.36508 dB / +0.231179`；instance bootstrap 95% CI 分别为 `[3.45692, 7.13857] dB` 和
  `[0.143378, 0.314838]`。PSNR 胜出 `18/20`，SSIM 胜出 `16/20`。这些实例共享同一 held-out scene 内容，
  因而 CI 只能作为内部诊断，不能当成跨场景泛化置信区间。结果位于 `results/sidd/internal_test_scene008/`。
- 失败样本集中在已经很干净的 ISO 100 输入：`0180_008_GP_00100_00100_5500_N` 为
  `31.5265 -> 27.7523 dB`、`0.8788 -> 0.7072 SSIM`；`0188_008_IP_00100_00100_3200_N` 为
  `31.9034 -> 29.2139 dB`、`0.8784 -> 0.7969 SSIM`。当前非 noise-aware 模型会对低噪声图继续强去噪，
  导致布料细纹理被抹除。
- 视觉结论：高噪声区域的彩色噪声显著减少、未观察到 tile 拼接缝或明显颜色漂移；但细密布料纹理存在
  可见过度平滑。基线已证明真实 RGB noisy-to-GT 监督链路有效，下一阶段的主要问题是低噪声自适应与保纹理，
  不是简单增加 epoch。

## 2026-07-18 SIDD Gaussian feature-loss 与 RTV 消融

- 目的：在完全相同的 SIDD-Small scene-disjoint 划分、网络、优化器、20 epoch 和 seed42 下，从头训练两个单变量递进实验：
  1. `NOISY -> GT + Gaussian masked feature prediction`，`rtv_weight=0`；
  2. 在 1 的基础上增加 `rtv_weight=1e-4`。
- feature 分支只在训练期存在：随机选择 25% 的 16x16 区域，在该区域加入 `sigma ~ U(0.0072677, 0.0218030)` 的 Gaussian noise；
  student predictor 对齐 GT 的 EMA teacher encoder2/encoder3 特征，`feature_weight=0.10`。推理网络和基线完全同构，仍为 67,614 参数，
  不增加推理参数或额外噪声。噪声与 feature-loss 绑定，不作用于主监督输入。
- 训练健康：两组均完成 20 epoch/4,800 steps，无 NaN、AMP skip 或发散。feature-only 最佳为 epoch19，scene007 validation
  Charbonnier=`0.00941634`；feature+RTV 最佳为 epoch19，`0.00953235`；纯监督基线为 epoch19，`0.00959633`。
- 公开 SIDD Validation blocks（1280 blocks，RGB PSNR/SSIM）：
  - baseline：`35.17941 / 0.845399`；
  - feature-only：`35.29428 / 0.847705`，相对 baseline `+0.11486 dB / +0.002306`；配对 bootstrap 95% CI
    `[0.09225,0.13700] dB / [0.001768,0.002826]`；胜出 `848/1280 PSNR`、`957/1280 SSIM`；
  - feature+RTV：`35.36451 / 0.853399`，相对 baseline `+0.18510 dB / +0.008000`；相对 feature-only
    `+0.07024 dB / +0.005694`，增量 95% CI `[0.04956,0.09133] dB / [0.005295,0.006106]`，
    胜出 `861/1280 PSNR`、`1121/1280 SSIM`。
- scene008 完整图 internal test（20 instances，tile512/overlap64）：
  - baseline：`31.83252 / 0.786656`；
  - feature-only：`31.87815 / 0.787629`，相对 baseline `+0.04563 dB / +0.000973`；PSNR CI 不跨 0，SSIM CI 跨 0；
  - feature+RTV：`31.92025 / 0.790455`，相对 baseline `+0.08773 dB / +0.003798`，95% CI
    `[0.03083,0.14546] dB / [0.000522,0.007335]`；相对 feature-only `+0.04210 dB / +0.002825`，
    95% CI `[0.01363,0.07133] dB / [0.001595,0.004079]`，20 张中两项均胜出 15 张。
- 视觉复核：高噪声 `0170` 中 feature 和 RTV 都降低彩噪，RTV 未观察到新增假结构或 tile 接缝；但三个模型都明显抹平 GT 的织物网格。
  低 ISO `0180/0188` 仍是主要失败点：输入约 `31.53/31.90 dB`，feature+RTV 仅 `27.90/29.17 dB`，说明当前模型仍会对已较干净图像过度去噪。
  RTV 相对 feature-only 在这些样本没有一致恶化，但也没有解决 noise-aware 问题。
- 决策：`feature+RTV(1e-4)` 是当前三组中量化指标最好的配置，可以作为下一阶段候选；feature-loss 的独立收益成立，低权重 RTV 也有额外收益。
  结论仍限于单 seed、公开 Validation 和单个 held-out scene，不能等同于隐藏 GT 的官方 benchmark；在扩大模型/训练轮数前，优先解决低 ISO 自适应与纹理保持。
- 产物：三组配对统计、CI、曲线和高/低噪声五列视觉图位于 `results/sidd/ablation_comparison/`；最佳 checkpoint 分别位于
  `results/sidd/supervised_feature_gaussian_s42/best.pt` 和 `results/sidd/supervised_feature_gaussian_rtv1e4_s42/best.pt`。

## 2026-07-18 SIDD-Medium sRGB 下载、校验与解压

- 目的：把训练数据从 SIDD-Small 的每个 scene instance 一对图扩展到论文常用的 SIDD-Medium sRGB（每个 scene instance 两对图），
  同时区分 SIDD-Medium 与不适合本机磁盘容量的 SIDD-Full。
- 官方来源：`http://130.63.97.225/share/SIDD_Medium_Srgb.zip`；官网标称约 12 GB，实际压缩包
  `13,234,744,070 bytes = 12.326 GiB`。下载到 `E:/SIDD/SIDD_Medium_Srgb.zip`，解压目录为
  `E:/SIDD/SIDD_Medium_Srgb/`。
- 下载过程：官方单连接续传仅约 20--45 KiB/s，改用官网同一 URL 的 aria2 16 段 Range 续传，平均约 3.7 MiB/s。
  aria2 1.37.0 Windows 64-bit 工具来自官方 GitHub release，工具压缩包 SHA256
  `67D015301EEF0B612191212D564C5BB0A14B5B9C4796B76454276A4D28D9B288`。
- 官方完整性校验全部通过：MD5=`F95B4BC9EC1DD3FE4EBD61AEACAD3991`；
  SHA1=`B0F895258112DB896D6ADE0A8DDAFC8CFC9BD54D`。解压后为 160 个 scene instance、320 对 noisy/GT、
  640 张 RGB PNG；逐对尺寸不匹配为 0，解压文件总量约 12.325 GiB。
- SIDD-Full 容量核算：官网 Full 清单中的 320 个 sRGB 压缩分卷（160 scene instance × noisy/GT）Content-Length
  合计 `986,172,587,651 bytes = 918.445 GiB`，还不含 Raw-RGB 和 metadata；E 盘下载前只有约 245.4 GiB 空闲，
  因此不能在该盘保存 SIDD-Full。对标准 sRGB 监督训练，应使用 SIDD-Medium，而不是全量采集帧。
- 新增可复用脚本：`scripts/download_sidd_medium.ps1` 支持断点续传和多连接；
  `scripts/verify_extract_sidd_medium.ps1` 强制检查字节数、MD5、SHA1、解压结果和 320/320 图像计数。
- 当前边界：数据已准备好，但现有 `data/sidd_dataset.py` 仍按 SIDD-Small 文件名只读取 `NOISY/GT_SRGB_010.PNG`；
  Medium 文件还带 scene-instance 前缀并包含 `010/011`。正式训练前必须先扩展 pair discovery，并明确是用全部 320 对训练后只在公开
  Validation/Benchmark 评估，还是继续保留 scene-disjoint internal test；本次仅完成下载与数据完整性验证，未擅自启动新训练。

## 2026-07-18 SIDD-Medium scene-disjoint feature/RTV 重跑

- 用户要求：数据下载后按上一轮顺序重跑，先 Gaussian feature-loss、RTV=0，再在完全相同配置上增加 `rtv_weight=1e-4`，
  两组结束后自动评估公开 Validation blocks 和 scene008 完整图。
- 对比协议：为了和 SIDD-Small 结果保持严格可比并避免 test leakage，继续按 scene 划分：train=`001-006,009,010`、
  val=`007`、test=`008`。Medium 使每个 scene instance 从 1 对变为 2 对，因此 train/val/test 从 `120/20/20`
  增加到 `240/40/40`；crop256、每 pair repeats=8、有效 batch16、20 epoch、seed42 及其余优化参数不变。
- 数据适配：`data/sidd_dataset.py` 现在同时兼容 Small 的 `NOISY_SRGB_010.PNG` 和 Medium 带 scene-instance 前缀的
  `*_NOISY_SRGB_010/011.PNG`，并用对应文件名自动寻找 GT；Small/Medium 发现数量分别回归验证为 160/320。
- Medium 噪声重标定：240 个 train pair 的固定中心 256 crop 上，noisy-GT residual robust MAD sigma 中位数为
  `0.0348847583`，因此 feature corruption 使用 `sigma ~ U(0.0087211896, 0.0261635687)`。
- 两组各 2 optimizer-step smoke 通过：train_pairs=240、val_pairs=40、模型参数 67,614，forward/backward、EMA、
  Gaussian feature 和 float32 RTV 均无 NaN/Inf 或 AMP skip。正式 20 epoch 结果待后台顺序训练后回填。
