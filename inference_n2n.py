from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from eval import make_comparison
from models.denoiser import Denoiser
from utils.checkpoint import load_weights_flexible
from utils.intensity import IntensityTransform, append_condition_channel, lambda_condition_value
from utils.io import list_inputs, read_image_any, save_npy_and_png


def build_condition(x: torch.Tensor, args) -> torch.Tensor | None:
    if not (args.lambda_conditioned and args.intensity_transform == "boxcox"):
        return None
    value = lambda_condition_value(args.boxcox_lam, args.lambda_min, args.lambda_max)
    return torch.full_like(x, fill_value=value)


def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    h, w = x.shape[-2:]
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, pad_h, pad_w


def crop_back(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    if pad_h > 0:
        x = x[..., :-pad_h, :]
    if pad_w > 0:
        x = x[..., :-pad_w]
    return x


def run_single(path: Path, args, transform: IntensityTransform, model: Denoiser, device: torch.device) -> None:
    raw = read_image_any(path)
    max_value = float(np.max(raw)) if args.max_value <= 0 else float(args.max_value)
    x = torch.from_numpy(raw).float().unsqueeze(0).unsqueeze(0).to(device)
    z = transform.forward(x, lam=args.boxcox_lam)
    condition = build_condition(z, args)
    model_input = append_condition_channel(z, condition)
    model_input, pad_h, pad_w = pad_to_multiple(model_input, args.size_multiple)

    with torch.no_grad():
        pred_z = model(model_input)
        pred_z = crop_back(pred_z, pad_h, pad_w)
        denoised = transform.inverse(pred_z, max_value=max_value, lam=args.boxcox_lam)

    out_dir = Path(args.out_dir)
    stem = path.stem
    denoised_np = denoised.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

    save_npy_and_png(raw.astype(np.float32, copy=False), out_dir / "data_npy" / f"{stem}_input.npy", out_dir / "view_png" / f"{stem}_input.png")
    save_npy_and_png(denoised_np, out_dir / "data_npy" / f"{stem}_n2n.npy", out_dir / "view_png" / f"{stem}_n2n.png")
    make_comparison(raw, denoised_np, None, out_dir / "comparison" / f"{stem}_input_vs_n2n.png", zoom_size=args.zoom_size)


def main(args) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    input_channels = 2 if args.lambda_conditioned and args.intensity_transform == "boxcox" else 1
    transform = IntensityTransform(
        name=args.intensity_transform,
        boxcox_lam=args.boxcox_lam,
        boxcox_eps=args.boxcox_eps,
        vst_lut=args.vst_lut,
    )
    model = Denoiser(input_channels=input_channels).to(device)
    print("[INFO] Loading N2N:", load_weights_flexible(model, args.checkpoint, device))
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_inputs(args.input)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise RuntimeError(f"No supported inputs found: {args.input}")

    for path in files:
        print(f"[INFO] N2N inference {path}")
        run_single(path, args, transform, model, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained N2N denoiser and save visual comparisons.")
    parser.add_argument("--input", type=str, required=True, help="Input file or folder.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained N2N checkpoint.")
    parser.add_argument("--out_dir", type=str, default="results/images/n2n")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--intensity_transform", choices=["none", "log1p", "boxcox", "learned_vst"], default="log1p")
    parser.add_argument("--vst_lut", type=str, default="")
    parser.add_argument("--boxcox_lam", type=float, default=-0.15)
    parser.add_argument("--boxcox_eps", type=float, default=1e-6)
    parser.add_argument("--lambda_conditioned", type=int, default=0)
    parser.add_argument("--lambda_min", type=float, default=-0.3)
    parser.add_argument("--lambda_max", type=float, default=0.2)
    parser.add_argument("--max_value", type=float, default=0.0, help="<=0 means use input max.")
    parser.add_argument("--size_multiple", type=int, default=32)
    parser.add_argument("--zoom_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
