from __future__ import annotations

"""测 N2N(单 D') vs NTN(T+D') 的单图推理延迟，评估 real-time 可行性。

NTN = D'(T(I))，N2N = D'(I)（D' 与 N2N 同为 Denoiser），
所以 NTN 相对 N2N 的额外开销 = 翻译器 T 的一次前向。本脚本直接量到真实数字。

时间与权重无关，故随机初始化即可。
"""

import argparse
import contextlib
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.denoiser import Denoiser
from models.ntn import NoiseTranslator


def params_m(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def pad_mult(x: torch.Tensor, m: int = 32) -> torch.Tensor:
    h, w = x.shape[-2:]
    ph, pw = (m - h % m) % m, (m - w % m) % m
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    return x


@torch.no_grad()
def bench(fn, x, iters: int, device, autocast: bool) -> float:
    """返回每次调用的平均毫秒。"""
    cm = (lambda: torch.autocast(device_type=device.type, dtype=torch.float16)) if autocast else contextlib.nullcontext
    with cm():
        for _ in range(5):  # warmup
            fn(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def main(args) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[INFO] device={device} ({gpu}), size={args.height}x{args.width}, iters={args.iters}")

    expert = Denoiser(input_channels=1).to(device).eval()
    translator = NoiseTranslator(
        input_channels=1, width=args.t_width, middle_blocks=args.middle_blocks,
        inject_sigma=args.inject_sigma, residual_scale=args.residual_scale,
    ).to(device).eval()
    print(f"[INFO] params: Denoiser(D'/N2N)={params_m(expert):.3f}M, Translator(T)={params_m(translator):.3f}M")

    x = pad_mult(torch.randn(1, 1, args.height, args.width, device=device))

    def run(tag, autocast):
        n2n_ms = bench(lambda z: expert(z), x, args.iters, device, autocast)
        ntn_ms = bench(lambda z: expert(translator(z)), x, args.iters, device, autocast)
        t_ms = bench(lambda z: translator(z), x, args.iters, device, autocast)
        over = ntn_ms - n2n_ms
        print(f"\n==== {tag} ====")
        print(f"  N2N  (D' 单次)     : {n2n_ms:7.2f} ms   ({1000/n2n_ms:6.1f} FPS)")
        print(f"  NTN  (T + D')      : {ntn_ms:7.2f} ms   ({1000/ntn_ms:6.1f} FPS)")
        print(f"  其中 T 单独         : {t_ms:7.2f} ms")
        print(f"  NTN 比 N2N 多       : +{over:6.2f} ms  (+{over/n2n_ms*100:5.1f}%)")

    run("FP32", autocast=False)
    if device.type == "cuda":
        run("FP16 (autocast)", autocast=True)
    print("\n[提示] real-time 看 NTN 的 FPS；若 T 占比偏高，可减小 --t_width 或上 TensorRT/FP16。")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark single-image latency: N2N (D') vs NTN (T+D').")
    p.add_argument("--height", type=int, default=1216, help="测试分辨率高（自动 pad 到 32 的倍数）。")
    p.add_argument("--width", type=int, default=1376, help="测试分辨率宽。")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--t_width", type=int, default=32, help="翻译器 T 的通道宽度（与训练一致）。")
    p.add_argument("--middle_blocks", type=int, default=2)
    p.add_argument("--inject_sigma", type=float, default=1.0)
    p.add_argument("--residual_scale", type=float, default=1.0)
    p.add_argument("--device", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
