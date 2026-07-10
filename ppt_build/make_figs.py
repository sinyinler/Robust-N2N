# -*- coding: utf-8 -*-
"""为 PPT 附录重画缺失的证据图（数据全部来自本会话已记录的实测结果）。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)
NAVY, PINK, BLUE, GREEN, RED, GRAY = "#1E2761", "#D4537E", "#378ADD", "#1D9E75", "#C0392B", "#AAB0BB"


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches="tight"); plt.close(fig)
    print("->", name)


# ============ 图1 · 引子：噪声与信号同量级 ============
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
w = [0.01, 0.04, 0.05, 0.06, 0.1, 1.0]
p = [33.541, 33.637, 33.814, 33.651, 33.733, 33.483]
ax[0].plot(range(len(w)), p, "-o", color=PINK, lw=2, ms=7)
ax[0].scatter([2], [33.814], s=180, facecolors="none", edgecolors=RED, lw=2.2, zorder=5)
ax[0].annotate("当时选中 0.05", (2, 33.814), textcoords="offset points", xytext=(12, -6), color=RED, fontsize=10)
ax[0].annotate("相邻点上下跳 0.16~0.18 dB", (3, 33.651), textcoords="offset points", xytext=(-30, -30),
               color=NAVY, fontsize=10, arrowprops=dict(arrowstyle="->", color=NAVY))
ax[0].set_xticks(range(len(w))); ax[0].set_xticklabels([str(x) for x in w])
ax[0].set_xlabel("w_feat"); ax[0].set_ylabel("PSNR (dB)")
ax[0].set_ylim(33.42, 33.90)
ax[0].set_title("w_feat 扫描：是抖动，不是趋势", color=NAVY, fontsize=12, pad=12)
ax[0].grid(alpha=0.3)

ax[1].bar([0, 1], [33.951, 33.560], color=[BLUE, PINK], width=0.5)
for x, v in zip([0, 1], [33.951, 33.560]):
    ax[1].text(x, v + 0.02, f"{v:.3f}", ha="center", fontsize=12, fontweight="bold", color=NAVY)
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(["encoder1 扫描表\n(e1=0.7 行)", "γ 扫描表\n(γ=0.1 行)"], fontsize=10)
ax[1].set_ylim(33.3, 34.1); ax[1].set_ylabel("PSNR (dB)")
ax[1].annotate("", xy=(0, 33.951), xytext=(1, 33.560),
               arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
ax[1].text(0.5, 33.79, "同一配置，两次训练\nΔ = 0.391 dB", ha="center", color=RED,
           fontsize=11, fontweight="bold")
ax[1].set_title("配置完全相同，读数却差 0.391 dB", color=NAVY, fontsize=12)
ax[1].grid(alpha=0.3, axis="y")
save(fig, "A1_引子_噪声底.png")

# ============ 图2 · 阶段2 多 seed ============
fig, ax = plt.subplots(figsize=(9, 4.4))
x = np.arange(2); wdt = 0.25
vals = {"seed42": [0.762, 0.710], "seed43": [-0.060, -2.374], "seed44": [-0.598, -0.369]}
for i, (k, v) in enumerate(vals.items()):
    ax.bar(x + (i - 1) * wdt, v, wdt, label=k, color=[BLUE, "#F0997B", PINK][i])
ax.axhline(0, color="#444441", lw=1)
ax.set_xticks(x); ax.set_xticklabels(["w_feat = 0.3", "w_feat = 0.15"], fontsize=11)
ax.set_ylabel("ΔPSNR vs N2N (dB)")
ax.set_title("阶段2 · 同一配置换 seed，结果天差地别（seed42 的「冠军」是运气）", color=NAVY, fontsize=12)
ax.legend(); ax.grid(alpha=0.3, axis="y")
save(fig, "A5_阶段2_多seed.png")

# ============ 图3 · 阶段3 OOD（无 projector）============
lv = ["level4\n(见过)", "level3", "level2", "level1\n(最OOD)"]
fig, ax = plt.subplots(figsize=(9, 4.6))
for k, v, c in [("seed42", [0.762, 0.928, 0.534, -0.794], BLUE),
                ("seed43", [-0.060, -0.408, -1.030, -1.370], "#F0997B"),
                ("seed44", [-0.598, -1.674, -2.981, -2.725], PINK)]:
    ax.plot(lv, v, "-o", label=k, color=c, lw=1.8, ms=6)
ax.plot(lv, [0.03, -0.38, -1.16, -1.63], "-o", label="均值", color=NAVY, lw=3, ms=7)
ax.axhline(0, color=GRAY, lw=2, label="打平线")
ax.set_ylabel("ΔPSNR vs N2N (dB)")
ax.set_title("阶段3 · 越 OOD 越差（三 seed 一致、单调）", color=NAVY, fontsize=12)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
save(fig, "A6_阶段3_OOD趋势.png")

# ============ 图4 · 阶段5 projector 消融 ============
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(lv, [0.03, -0.38, -1.16, -1.63], "-o", label="无 projector", color=PINK, lw=2.6, ms=8)
ax.plot(lv, [0.14, 0.05, -0.09, -0.67], "-o", label="带 projector", color=GREEN, lw=2.6, ms=8)
ax.axhline(0, color=GRAY, lw=2, label="打平线")
for i, (a, b) in enumerate(zip([0.03, -0.38, -1.16, -1.63], [0.14, 0.05, -0.09, -0.67])):
    ax.annotate(f"+{b-a:.2f}", (i, (a + b) / 2), color=GREEN, fontsize=10,
                fontweight="bold", ha="center")
ax.set_ylabel("ΔPSNR vs N2N (dB)")
ax.set_title("阶段5 · projector 大幅压平 OOD 衰退（挽回约 1 dB）", color=NAVY, fontsize=12)
ax.legend(fontsize=10); ax.grid(alpha=0.3)
save(fig, "A7_阶段5_projector消融.png")

# ============ 图5 · 阶段6 干净消融汇总 ============
fig, ax = plt.subplots(figsize=(9, 4.4))
seeds = ["seed42", "seed43", "seed44"]
d = [0.109, -0.349, -0.848]; e = [0.400, 0.295, 0.690]
cols = [GRAY if abs(m) < s else (GREEN if m > 0 else RED) for m, s in zip(d, e)]
ax.bar(seeds, d, yerr=e, capsize=5, color=cols, width=0.5)
ax.axhline(0, color="#444441", lw=1)
ax.axhline(-0.362, color=NAVY, ls="--", lw=2, label="跨 seed 均值 −0.362")
ax.set_ylabel("配对 ΔPSNR (dB)")
ax.set_title("阶段6 · 唯一差异=特征损失项：跨 seed −0.362±0.479（打平·点估计为负）", color=NAVY, fontsize=11.5)
ax.legend(); ax.grid(alpha=0.3, axis="y")
save(fig, "A8_阶段6_干净消融.png")
print("done")
