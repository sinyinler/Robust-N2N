# -*- coding: utf-8 -*-
"""诊断 #0：解码器实际依赖哪条路径（bridge 瓶颈 vs out3 skip）。

背景：out3 同时 (a) 作为 Bridge 的输入、(b) 直接进 decoder 的 cat1 skip
（见 models/denoiser.py 的 Decoder.forward: cat((up(bridge), out3))）。
因此不能只“扰动 out3”就下结论，必须分支隔离：

  bridge-only    : 用原始 out3 正常算出 bridge，只替换送进 decoder 的 bridge
  out3-skip-only : 用原始 out3 算 bridge，只替换 decoder 拼接处的 out3
  joint          : 两者同时替换

扰动方式三种（zero 属于明显分布外干预，不能单独作结论，故与 shuffle 并列）：
  zero / batch-shuffle（沿样本维置换）/ spatial-shuffle（沿 H·W 置换）

**这个诊断能回答的**：当前**已训练好**的模型在推理时是否依赖该张量。
**它不能回答**：换新损失重训后模型会不会重新建立依赖。因此低敏感性只能否定
“继续在当前 bridge 上堆普通 SimSiam 一致性会自然增益”，不能否定 cross-reconstruction /
feature-specific teacher / VICReg 等会改变训练动力学的路线。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from infer_eval_robust import load2d, metrics
from models.denoiser_feats import DenoiserWithFeats
from utils.checkpoint import load_weights_flexible

MODES = ["none", "bridge", "out3skip", "joint"]
KINDS = ["zero", "batch_shuffle", "spatial_shuffle"]


def perturb(t: torch.Tensor, kind: str, gen: torch.Generator) -> torch.Tensor:
    """对 (N,C,H,W) 张量做指定扰动。"""
    if kind == "zero":
        return torch.zeros_like(t)
    if kind == "batch_shuffle":
        if t.shape[0] < 2:
            raise ValueError("batch_shuffle 需要 batch>=2，请增大 --batch")
        idx = torch.randperm(t.shape[0], generator=gen, device=t.device)
        return t[idx]
    if kind == "spatial_shuffle":
        n, c, h, w = t.shape
        flat = t.reshape(n, c, h * w)
        idx = torch.randperm(h * w, generator=gen, device=t.device)
        return flat[:, :, idx].reshape(n, c, h, w)
    raise ValueError(kind)


@torch.no_grad()
def forward_intervened(model, x, mode="none", kind="zero", gen=None):
    """手动走一遍前向，在指定分支上施加扰动。x 已是 log1p 域、已 pad。"""
    out1, out2, out3 = model.encoder(x)
    bridge = model.bridge(out3)               # 始终用**原始** out3 计算 bridge

    b_in, o3_in = bridge, out3
    if mode in ("bridge", "joint"):
        b_in = perturb(bridge, kind, gen)
    if mode in ("out3skip", "joint"):
        o3_in = perturb(out3, kind, gen)

    dec = model.decoder(b_in, out1, out2, o3_in)
    return model.transformer_unit(dec)


def pad32(t):
    h, w = t.shape[-2:]
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    return (F.pad(t, (0, pw, 0, ph), mode="reflect") if (ph or pw) else t), ph, pw


def unpad(t, ph, pw):
    if ph:
        t = t[..., :-ph, :]
    if pw:
        t = t[..., :-pw]
    return t


@torch.no_grad()
def run(model, frames, ref, dr, device, mode, kind, seed):
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    batch = torch.stack([torch.from_numpy(np.log1p(np.clip(load2d(f), 0, None)).astype(np.float32))
                         for f in frames])[:, None].to(device)
    xb, ph, pw = pad32(batch)
    out = unpad(forward_intervened(model, xb, mode, kind, gen), ph, pw)
    imgs = np.expm1(out.squeeze(1).cpu().numpy().astype(np.float32))
    return float(np.mean([metrics(img, ref, dr)[0] for img in imgs]))


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ref = load2d(args.reference)
    from eval_curve import natural_key
    frames = sorted(Path(args.scene_dir).glob("*.npy"), key=natural_key)[:args.batch]
    if len(frames) < 2:
        raise RuntimeError("至少需要 2 帧（batch_shuffle 要求 batch>=2）")
    dr = float(args.max)

    for tag, ckpt in [("N2N", args.n2n_checkpoint), ("+feature", args.feat_checkpoint)]:
        if not ckpt:
            continue
        model = DenoiserWithFeats(input_channels=1).to(device).eval()
        print(f"\n[INFO] {tag}: {ckpt}", load_weights_flexible(model, ckpt, device))
        base = run(model, frames, ref, dr, device, "none", "zero", args.seed)
        print(f"  未扰动 PSNR = {base:.3f} dB  （{len(frames)} 帧均值）")
        print(f"  {'干预分支':<14} | {'zero':>16} | {'batch_shuffle':>16} | {'spatial_shuffle':>16}")
        print("  " + "-" * 74)
        for mode in MODES[1:]:
            cells = []
            for kind in KINDS:
                p = run(model, frames, ref, dr, device, mode, kind, args.seed)
                cells.append(f"{p:.3f} ({p - base:+.3f})")
            print(f"  {mode:<14} | " + " | ".join(f"{c:>16}" for c in cells))
    print("\n判读：ΔPSNR 越接近 0，说明解码器越不依赖该分支。"
          "\n注意：低敏感性只否定「在该张量上继续堆普通一致性会自然增益」，"
          "\n      不否定会改变训练动力学的路线（cross-recon / teacher-student / VICReg）。")


def parse_args():
    p = argparse.ArgumentParser(description="分支隔离的 zero/shuffle 特征依赖诊断")
    p.add_argument("--n2n_checkpoint", default="", help="纯 N2N checkpoint")
    p.add_argument("--feat_checkpoint", default="", help="N2N + 特征损失 checkpoint")
    p.add_argument("--reference", default="/home/songyd/Projects/Robust-N2N/reference.npy")
    p.add_argument("--scene_dir", default="/mnt2/songyd/5x5/5x5x4/0/npy")
    p.add_argument("--batch", type=int, default=8, help="同时前向的帧数（≥2，batch_shuffle 需要）")
    p.add_argument("--max", type=float, default=255.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
