from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from models.denoiser import Denoiser
from models.ntn import NoiseTranslator
from utils.checkpoint import load_weights_flexible
from utils.intensity import IntensityTransform, append_condition_channel, lambda_condition_value
from utils.io import list_inputs, read_image_any, save_npy_and_png


def build_condition(x: torch.Tensor, args) -> torch.Tensor | None:
    if not (args.lambda_conditioned and args.intensity_transform == "boxcox"):
        return None
    value = lambda_condition_value(args.boxcox_lam, args.lambda_min, args.lambda_max)
    return torch.full_like(x, fill_value=value)


def run_single(path: Path, args, transform: IntensityTransform, translator, gaussian_expert, device: torch.device) -> None:
    raw = read_image_any(path)
    max_value = float(np.max(raw)) if args.max_value <= 0 else float(args.max_value)
    x = torch.from_numpy(raw).float().unsqueeze(0).unsqueeze(0).to(device)
    z = transform.forward(x, lam=args.boxcox_lam)
    condition = build_condition(z, args)

    with torch.no_grad():
        translated = translator(append_condition_channel(z, condition))
        denoised_z = gaussian_expert(append_condition_channel(translated, condition))
        denoised = transform.inverse(denoised_z, max_value=max_value, lam=args.boxcox_lam)

    out = denoised.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    out_dir = Path(args.out_dir)
    stem = path.stem
    save_npy_and_png(out, out_dir / f"{stem}_ntn.npy", out_dir / f"{stem}_ntn.png")

    if args.save_translated:
        translated_np = translated.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
        save_npy_and_png(translated_np, out_dir / f"{stem}_translated_vst.npy", out_dir / f"{stem}_translated_vst.png")


def main(args) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    input_channels = 2 if args.lambda_conditioned and args.intensity_transform == "boxcox" else 1

    transform = IntensityTransform(
        name=args.intensity_transform,
        boxcox_lam=args.boxcox_lam,
        boxcox_eps=args.boxcox_eps,
        vst_lut=args.vst_lut,
    )
    translator = NoiseTranslator(
        input_channels=input_channels,
        width=args.width,
        middle_blocks=args.middle_blocks,
        inject_sigma=args.inject_sigma,
        residual_scale=args.residual_scale,
    ).to(device)
    gaussian_expert = Denoiser(input_channels=input_channels).to(device)

    print("[INFO] Loading translator:", load_weights_flexible(translator, args.translator_checkpoint, device))
    print("[INFO] Loading Gaussian expert:", load_weights_flexible(gaussian_expert, args.gaussian_expert_checkpoint, device))
    translator.eval()
    gaussian_expert.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_inputs(args.input)
    if not files:
        raise RuntimeError(f"No supported inputs found: {args.input}")
    for path in files:
        print(f"[INFO] Inference {path}")
        run_single(path, args, transform, translator, gaussian_expert, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NTN inference: noisy -> T -> D' -> denoised.")
    parser.add_argument("--input", type=str, required=True, help="Input file or folder.")
    parser.add_argument("--out_dir", type=str, default="results/images")
    parser.add_argument("--translator_checkpoint", type=str, required=True)
    parser.add_argument("--gaussian_expert_checkpoint", type=str, required=True)
    parser.add_argument("--intensity_transform", choices=["none", "log1p", "boxcox", "learned_vst"], default="log1p")
    parser.add_argument("--vst_lut", type=str, default="")
    parser.add_argument("--boxcox_lam", type=float, default=-0.15)
    parser.add_argument("--boxcox_eps", type=float, default=1e-6)
    parser.add_argument("--lambda_conditioned", type=int, default=0)
    parser.add_argument("--lambda_min", type=float, default=-0.3)
    parser.add_argument("--lambda_max", type=float, default=0.2)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--middle_blocks", type=int, default=2)
    parser.add_argument("--inject_sigma", type=float, default=1.0)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--max_value", type=float, default=0.0, help="<=0 means use input max.")
    parser.add_argument("--save_translated", action="store_true")
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
