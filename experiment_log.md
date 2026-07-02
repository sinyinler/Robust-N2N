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
- 待办 / 开工前对齐（见下方与用户确认后再实现）：GIBlock 插入位置、损失权重初值、训练数据路径。
- 结果：待训练后回填（指标 + 血管局部放大图）。
