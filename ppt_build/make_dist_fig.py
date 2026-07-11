# -*- coding: utf-8 -*-
"""分布扫描（128 projector + 池化 + 3 seed）汇总图：分布轴不可分辨 + feat 与 PSNR 无关联。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)
NAVY, PINK, BLUE, GREEN, RED, GRAY, ORANGE = "#1E2761", "#D4537E", "#378ADD", "#1D9E75", "#C0392B", "#AAB0BB", "#F0997B"

arms = ["bn 单", "e3+bn", "e2e3bn", "均匀", "深重浅轻\n.1/.2/.3/.4", "浅重深轻\n.4/.3/.2/.1"]
# 每条臂在 seed 42/187/2413 上的 ΔPSNR（vs 同 seed w_feat=0 基线）
d = {
    "seed42":  [-0.183, -0.101, -0.382, -0.247, -0.073, -0.033],
    "seed187": [-0.297, -0.065, -0.022, -0.266, -0.488, -0.294],
    "seed2413":[-0.044, +0.169, +0.361, +0.141, +0.377, +0.205],
}
M = np.array(list(d.values()))            # 3 x 6
mean = M.mean(0); std = M.std(0, ddof=1)
x = np.arange(len(arms))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.1, 1]})

# ---- 左：每条臂 mean±std + 三个 seed 散点 ----
axL.bar(x, mean, yerr=std, capsize=5, color=GRAY, alpha=0.55, width=0.6, label="跨 seed mean±std")
for (k, v), c in zip(d.items(), [BLUE, ORANGE, PINK]):
    axL.scatter(x, v, s=48, color=c, zorder=5, label=k)
axL.axhline(0, color="#444441", lw=1.2)
axL.set_xticks(x); axL.set_xticklabels(arms, fontsize=9.5)
axL.set_ylabel("ΔPSNR vs 同 seed 基线 (dB)")
axL.set_title("分布扫描（128 projector + 区域级一致性，3 seed）：无一条臂 mean>std", color=NAVY, fontsize=12.5)
axL.legend(fontsize=9, loc="lower left", ncol=2)
axL.grid(alpha=0.3, axis="y")
# 标注：最优臂在不同 seed 排名 1↔6
axL.annotate("深重浅轻：seed2413 排第1，seed187 排第6", (4, -0.488), textcoords="offset points",
             xytext=(-70, -18), fontsize=9, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED))
axL.set_ylim(-0.62, 0.50)

# ---- 右：feat 饱和与 PSNR 无关联 ----
seeds = ["s42", "s187", "s2413"]
base_feat = [0.003, -0.014, 0.015]        # w_feat=0 时特征天然几乎不对齐
arm_feat = [-0.955, -0.949, -0.951]       # 加损失后立刻拉到 ~-0.95（各臂均值）
axR.bar(np.arange(3) - 0.18, base_feat, 0.36, color=GRAY, label="w_feat=0（基线）")
axR.bar(np.arange(3) + 0.18, arm_feat, 0.36, color=PINK, label="加损失（各臂均值）")
axR.axhline(-1.0, color=NAVY, ls=":", lw=1, label="理论下限 −1")
axR.set_xticks(range(3)); axR.set_xticklabels(seeds)
axR.set_ylabel("feat（负余弦，越负越对齐）")
axR.set_title("损失轻松拉到 −0.95\n但 PSNR 一 seed 正两 seed 负", color=NAVY, fontsize=11.5)
axR.legend(fontsize=8.5, loc="lower center")
axR.grid(alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(os.path.join(OUT, "A9_分布扫描_不可分辨.png"), dpi=160, bbox_inches="tight")
print("-> A9_分布扫描_不可分辨.png")
print("means:", np.round(mean, 3), "\nstds :", np.round(std, 3))
