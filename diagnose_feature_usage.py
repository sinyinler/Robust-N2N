# -*- coding: utf-8 -*-
"""诊断 #0：解码器实际依赖哪条路径（bridge 瓶颈 vs out3 skip）。

**测量的是「输出敏感性」，不是「去噪质量」。**
早期版本用「扰动后对 reference 的 PSNR」来判断依赖性，这是错的：破坏深层特征会让输出变平滑，
对偏平滑的 reference 反而 PSNR 更高（实测 N2N 上 bridge-zero 使 PSNR +0.73 dB）。
质量指标无法回答「解码器用不用这个张量」。

正确的问法：扰动该分支后，**输出本身**变了多少？
    rel_Δ = ‖y_p − y_0‖₂ / ‖y_0‖₂        （越接近 0 = 解码器越不依赖）
    PSNR(y_p ‖ y_0)                        （以未扰动输出为参考，越高 = 越不依赖）
不需要任何 reference，也不受过平滑影响。

decoder 有四个输入（bridge, out1, out2, out3），全部单独测一遍，画出信息流地图。
out3 同时 (a) 作为 Bridge 输入、(b) 直接进 decoder 的 cat1 skip，故必须分支隔离：
  bridge-only    : 用原始 out3 正常算 bridge，只替换送进 decoder 的 bridge
  out3-skip-only : 用原始 out3 算 bridge，只替换 decoder 拼接处的 out3
  joint          : 两者同时替换

扰动：zero / batch-shuffle（沿样本维）/ spatial-shuffle（沿 H·W）。
**batch-shuffle 只有在 batch 由不同场景组成时才有意义**（同场景不同帧的深层特征本就该相似）。
默认从 --scene_root 下多个场景各取一帧。

**能回答**：当前已训练好的模型在推理时是否依赖该张量。
**不能回答**：换新损失重训后模型会不会重新建立依赖。低敏感性只否定「继续在该张量上堆普通
SimSiam 一致性会自然增益」，不否定 cross-recon / teacher-student / VICReg 等改变训练动力学的路线。
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from diag_common import collect_frames, load_batch
from models.denoiser_feats import DenoiserWithFeats
from utils.checkpoint import load_weights_flexible

# decoder 的四个输入全部单独测 + 深/浅分组 + 全部(应等于天花板) + 输入置换(天花板刻度)
MODES = ["bridge", "out3skip", "out2skip", "out1skip",
         "deep(bridge+out3)", "shallow(out1+out2)", "all(4条)"]
KINDS = ["zero", "batch_shuffle", "spatial_shuffle"]

GROUPS = {                                          # mode -> 该干预命中哪几个张量
    "bridge": {"bridge"}, "out3skip": {"out3"}, "out2skip": {"out2"}, "out1skip": {"out1"},
    "deep(bridge+out3)": {"bridge", "out3"}, "shallow(out1+out2)": {"out1", "out2"},
    "all(4条)": {"bridge", "out3", "out2", "out1"},
}


def perturb(t: torch.Tensor, kind: str, bperm: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """bperm 由调用方**统一**给出：组干预必须用同一个样本置换，否则是拼接嵌合体而非「整体换场景」。"""
    if kind == "zero":
        return torch.zeros_like(t)
    if kind == "batch_shuffle":
        return t[bperm]
    if kind == "spatial_shuffle":
        n, c, h, w = t.shape                        # 空间置换按各自尺度独立抽（跨尺度无自然对应）
        idx = torch.randperm(h * w, generator=gen, device=t.device)
        return t.reshape(n, c, h * w)[:, :, idx].reshape(n, c, h, w)
    raise ValueError(kind)


@torch.no_grad()
def forward_intervened(model, x, mode=None, kind="zero", bperm=None, gen=None):
    out1, out2, out3 = model.encoder(x)
    bridge = model.bridge(out3)                     # 始终用**原始** out3 计算 bridge
    tensors = {"bridge": bridge, "out3": out3, "out2": out2, "out1": out1}
    if mode is not None:
        for name in GROUPS[mode]:
            tensors[name] = perturb(tensors[name], kind, bperm, gen)
    return model.transformer_unit(
        model.decoder(tensors["bridge"], tensors["out1"], tensors["out2"], tensors["out3"]))


@torch.no_grad()
def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    frames, multi_scene = collect_frames(args.scene_root, args.n_scenes, args.frame_idx,
                                         args.scene_dir, args.n_frames)
    if not multi_scene:
        print("[WARN] 同场景 batch：batch_shuffle 一列不可解读，请改用 --scene_root。")

    x = load_batch(frames, args.crop, device)      # 中心裁剪到统一尺寸（不同场景尺寸不同）
    print(f"[INFO] batch shape = {tuple(x.shape)}")

    for tag, ckpt in [("N2N", args.n2n_checkpoint), ("+feature", args.feat_checkpoint)]:
        if not ckpt:
            continue
        model = DenoiserWithFeats(input_channels=1).to(device).eval()
        print(f"\n=== {tag} === {ckpt}", load_weights_flexible(model, ckpt, device))
        y0 = forward_intervened(model, x)                      # 未扰动输出（作为参考）
        n0 = y0.flatten(1).norm(dim=1)

        def stats(yp):
            rel = ((yp - y0).flatten(1).norm(dim=1) / n0.clamp_min(1e-8)).mean().item()
            mse = ((yp - y0) ** 2).flatten(1).mean(dim=1)
            rng = (y0.flatten(1).amax(dim=1) - y0.flatten(1).amin(dim=1)).clamp_min(1e-8)
            psnr = (10 * torch.log10(rng ** 2 / mse.clamp_min(1e-12))).mean().item()
            return rel, psnr

        # ---- 天花板：直接把**输入**在样本维置换，得到「换成另一个场景」的输出差异 ----
        bperm = torch.randperm(x.shape[0], generator=torch.Generator(device=device).manual_seed(args.seed),
                               device=device)
        ceil_rel, ceil_psnr = stats(forward_intervened(model, x[bperm]))
        print(f"  【天花板】input_shuffle（整张输入换场景）: rel_Δ={ceil_rel * 100:6.2f}%  PSNR={ceil_psnr:6.2f}"
              f"   ← 这就是「场景身份」能造成的最大输出变化，下表 batch_shuffle 列按它归一化")

        print(f"\n  {'干预分支':<20} | " + " | ".join(f"{k:^30}" for k in KINDS))
        print("  " + "-" * 116)
        for mode in MODES:
            cells = []
            for kind in KINDS:
                gen = torch.Generator(device=device); gen.manual_seed(args.seed)
                rel, psnr = stats(forward_intervened(model, x, mode, kind, bperm, gen))
                frac = f" ={rel / max(ceil_rel, 1e-8) * 100:5.1f}%上限" if kind == "batch_shuffle" else ""
                cells.append(f"rel_Δ={rel * 100:6.2f}% PSNR={psnr:6.2f}{frac}")
            print(f"  {mode:<20} | " + " | ".join(f"{c:^30}" for c in cells))

    print("\n判读：")
    print("  · batch_shuffle 列请看「占上限的百分比」——绝对 rel_Δ 无法单独解读。")
    print("  · 组干预（deep/shallow/all）使用**同一个样本置换**，即「整体换成另一个场景的特征」。")
    print("  · all(4条) 的 batch_shuffle 应≈100% 上限（decoder 只有这四个输入），可作自检。")
    print("  · 别用「对 reference 的 PSNR」判断依赖性：破坏特征会过平滑，反而抬高 PSNR。")


def parse_args():
    p = argparse.ArgumentParser(description="分支隔离的 zero/shuffle 输出敏感性诊断")
    p.add_argument("--n2n_checkpoint", default="")
    p.add_argument("--feat_checkpoint", default="")
    # 推荐：多场景（batch_shuffle 才有意义）
    p.add_argument("--scene_root", default="/mnt2/songyd/5x5/5x5x4", help="其下每个子目录是一个场景")
    p.add_argument("--n_scenes", type=int, default=8)
    p.add_argument("--frame_idx", type=int, default=0, help="每个场景取第几帧")
    # 退回：同场景（仅对照）
    p.add_argument("--scene_dir", default="")
    p.add_argument("--n_frames", type=int, default=8)
    p.add_argument("--crop", type=int, default=512, help="中心裁剪到该尺寸（32 的倍数）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="")
    a = p.parse_args()
    if a.scene_dir:
        a.scene_root = ""
    return a


if __name__ == "__main__":
    main(parse_args())
