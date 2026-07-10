# -*- coding: utf-8 -*-
"""只用瓶颈能去噪到什么程度？

训练日志显示 AuxDec（无 skip、只吃 bridge）的 Charbonnier 只比完整 U-Net 高 4.2%
（aux=0.2655 vs rec=0.2547）。但那是对**带噪靶子**的损失，主要由不可约噪声主导，
不能直接换算成去噪质量。这里直接对同一 reference 测两条路的 PSNR/MSSIM/r：

    完整路径 : x → encoder → bridge → decoder(带 out1/out2/out3 skip) → y_full
    瓶颈路径 : x → encoder → bridge → AuxDec(无任何 skip)            → y_aux

若 y_aux 与 y_full 接近，则「瓶颈携带了几乎全部去噪所需信息，只是主解码器不取用」；
若 y_aux 明显更差，则瓶颈只是学到了一个够用于骗过带噪靶子的粗糙码。
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from diag_common import collect_frames, load_batch, center_crop   # center_crop(a, size)
from infer_eval_robust import load2d, metrics
from models.denoiser_feats import DenoiserWithFeats
from models.aux_decoder import AuxDecoder
from utils.checkpoint import load_weights_flexible


@torch.no_grad()
def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    frames, _ = collect_frames("", 0, 0, args.scene_dir, args.n_frames)
    x = load_batch(frames, args.crop, device)
    ref = center_crop(load2d(args.reference), args.crop)     # reference 也裁到同一窗口
    dr = float(args.max)

    model = DenoiserWithFeats(input_channels=1).to(device).eval()
    print("[INFO] denoiser:", load_weights_flexible(model, args.checkpoint, device))
    aux = AuxDecoder(in_channels=80).to(device).eval()
    print("[INFO] aux_dec :", load_weights_flexible(aux, args.aux_checkpoint, device))

    out1, out2, out3 = model.encoder(x)
    bridge = model.bridge(out3)
    y_full = model.transformer_unit(model.decoder(bridge, out1, out2, out3))
    y_aux = aux(bridge, x.shape[-2:])

    print(f"\n  reference 裁剪窗口 {ref.shape}，{len(frames)} 帧，data_range={dr:g}")
    print(f"  {'路径':<28} | {'PSNR':>8} | {'MSSIM':>8} | {'r':>8}")
    print("  " + "-" * 62)
    rows = [("完整路径（带 skip）", y_full), ("瓶颈路径（AuxDec，无 skip）", y_aux)]
    res = {}
    for name, y in rows:
        imgs = np.expm1(y.squeeze(1).cpu().numpy().astype(np.float32))
        m = np.array([metrics(img, ref, dr) for img in imgs]).mean(axis=0)
        res[name] = m
        print(f"  {name:<28} | {m[0]:>8.3f} | {m[1]:>8.4f} | {m[2]:>8.4f}")

    d = res["瓶颈路径（AuxDec，无 skip）"][0] - res["完整路径（带 skip）"][0]
    print(f"\n  ΔPSNR（瓶颈路径 − 完整路径）= {d:+.3f} dB")
    print("  判读：|Δ| 很小 → 瓶颈已携带几乎全部去噪信息，主解码器只是不取用（走 skip 更省事）；")
    print("        Δ 很负   → 瓶颈只学到粗糙码，够骗过带噪靶子，不够真正去噪。")


def parse_args():
    p = argparse.ArgumentParser(description="对比「完整路径」与「仅瓶颈路径」的去噪质量")
    p.add_argument("--checkpoint", required=True, help="去噪器 checkpoint")
    p.add_argument("--aux_checkpoint", required=True, help="aux_dec_epoch_N.pth")
    p.add_argument("--reference", default="/home/songyd/Projects/Robust-N2N/reference.npy")
    p.add_argument("--scene_dir", default="/mnt2/songyd/5x5/5x5x4/0/npy")
    p.add_argument("--n_frames", type=int, default=8)
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--max", type=float, default=255.0)
    p.add_argument("--device", default="")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
