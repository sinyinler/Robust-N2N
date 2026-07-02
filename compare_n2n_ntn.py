from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from models.denoiser import Denoiser
from models.ntn import NoiseTranslator
from utils.checkpoint import load_weights_flexible
from utils.intensity import IntensityTransform, append_condition_channel, lambda_condition_value
from utils.io import read_image_any, save_npy_and_png


def normalize_to_u8(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """把浮点图像按统一窗宽窗位转换为 8-bit，便于公平目视比较。"""

    if vmax <= vmin:
        return np.zeros_like(x, dtype=np.uint8)
    y = np.clip((x.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


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


def build_condition(x: torch.Tensor, args) -> torch.Tensor | None:
    if not (args.lambda_conditioned and args.intensity_transform == "boxcox"):
        return None
    value = lambda_condition_value(args.boxcox_lam, args.lambda_min, args.lambda_max)
    return torch.full_like(x, fill_value=value)


def load_denoiser(checkpoint: str, device: torch.device, input_channels: int, name: str) -> Denoiser:
    model = Denoiser(input_channels=input_channels).to(device)
    print(f"[INFO] Loading {name}:", load_weights_flexible(model, checkpoint, device))
    model.eval()
    return model


def load_translator(args, device: torch.device, input_channels: int) -> NoiseTranslator:
    model = NoiseTranslator(
        input_channels=input_channels,
        width=args.width,
        middle_blocks=args.middle_blocks,
        inject_sigma=args.inject_sigma,
        residual_scale=args.residual_scale,
    ).to(device)
    print("[INFO] Loading translator:", load_weights_flexible(model, args.translator_checkpoint, device))
    model.eval()
    return model


def run_n2n(raw: np.ndarray, args, transform: IntensityTransform, model: Denoiser, device: torch.device) -> np.ndarray:
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
    return denoised.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)


def run_ntn(
    raw: np.ndarray,
    args,
    transform: IntensityTransform,
    translator: NoiseTranslator,
    gaussian_expert: Denoiser,
    device: torch.device,
) -> np.ndarray:
    max_value = float(np.max(raw)) if args.max_value <= 0 else float(args.max_value)
    x = torch.from_numpy(raw).float().unsqueeze(0).unsqueeze(0).to(device)
    z = transform.forward(x, lam=args.boxcox_lam)
    condition = build_condition(z, args)
    model_input = append_condition_channel(z, condition)
    model_input, pad_h, pad_w = pad_to_multiple(model_input, args.size_multiple)
    if condition is not None:
        condition, _, _ = pad_to_multiple(condition, args.size_multiple)

    with torch.no_grad():
        translated = translator(model_input)
        denoised_z = gaussian_expert(append_condition_channel(translated, condition))
        denoised_z = crop_back(denoised_z, pad_h, pad_w)
        denoised = transform.inverse(denoised_z, max_value=max_value, lam=args.boxcox_lam)
    return denoised.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)


def make_triplet(raw: np.ndarray, n2n: np.ndarray, ntn: np.ndarray, out_path: Path, title: str, zoom_size: int) -> None:
    h = min(raw.shape[-2], n2n.shape[-2], ntn.shape[-2])
    w = min(raw.shape[-1], n2n.shape[-1], ntn.shape[-1])
    raw = raw[:h, :w]
    n2n = n2n[:h, :w]
    ntn = ntn[:h, :w]
    images = [raw, n2n, ntn]
    labels = ["Input", "N2N baseline", "NTN T + D_prime"]
    values = np.concatenate([x.reshape(-1) for x in images])
    vmin = float(np.percentile(values, 1))
    vmax = float(np.percentile(values, 99))
    panels = [Image.fromarray(normalize_to_u8(x, vmin, vmax), mode="L").convert("RGB") for x in images]

    zoom = min(zoom_size, h, w)
    top = max(0, (h - zoom) // 2)
    left = max(0, (w - zoom) // 2)
    line_width = max(1, w // 256)
    zoom_panels = []
    for panel in panels:
        draw = ImageDraw.Draw(panel)
        draw.rectangle([left, top, left + zoom, top + zoom], outline=(255, 0, 0), width=line_width)
        crop = panel.crop((left, top, left + zoom, top + zoom)).resize((w, h), Image.Resampling.BICUBIC)
        zoom_panels.append(crop)

    label_h = 48 if title else 32
    canvas = Image.new("RGB", (w * 3, h * 2 + label_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    if title:
        draw.text((8, 6), title, fill=(180, 180, 180), font=font)
    for idx, (panel, zoom_panel, label) in enumerate(zip(panels, zoom_panels, labels)):
        x0 = idx * w
        canvas.paste(panel, (x0, label_h))
        canvas.paste(zoom_panel, (x0, label_h + h))
        draw.text((x0 + 8, label_h - 18), label, fill=(255, 255, 255), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def output_stem(path: Path, prefix: str) -> str:
    """根据输入路径生成稳定且不冲突的输出名，避免多个 0.npy 互相覆盖。"""

    parts = list(path.parts)
    if path.parent.name == "npy" and len(parts) >= 4:
        tail = parts[-4:-1] + [path.stem]
    else:
        tail = parts[-3:-1] + [path.stem]
    safe = [part.replace(":", "").replace("\\", "_").replace("/", "_") for part in tail]
    return prefix + "_".join(safe)


def main(args) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    input_channels = 2 if args.lambda_conditioned and args.intensity_transform == "boxcox" else 1
    transform = IntensityTransform(
        name=args.intensity_transform,
        boxcox_lam=args.boxcox_lam,
        boxcox_eps=args.boxcox_eps,
        vst_lut=args.vst_lut,
    )
    n2n_model = load_denoiser(args.n2n_checkpoint, device, input_channels, "N2N")
    gaussian_expert = load_denoiser(args.gaussian_expert_checkpoint, device, input_channels, "Gaussian expert D_prime")
    translator = load_translator(args, device, input_channels)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in args.inputs:
        path = Path(item)
        raw = read_image_any(path).astype(np.float32, copy=False)
        print(f"[INFO] Comparing {path}")
        n2n = run_n2n(raw, args, transform, n2n_model, device)
        ntn = run_ntn(raw, args, transform, translator, gaussian_expert, device)
        stem = output_stem(path, args.name_prefix)
        save_npy_and_png(raw, out_dir / "data_npy" / f"{stem}_input.npy", out_dir / "view_png" / f"{stem}_input.png")
        save_npy_and_png(n2n, out_dir / "data_npy" / f"{stem}_n2n.npy", out_dir / "view_png" / f"{stem}_n2n.png")
        save_npy_and_png(ntn, out_dir / "data_npy" / f"{stem}_ntn.npy", out_dir / "view_png" / f"{stem}_ntn.png")
        make_triplet(raw, n2n, ntn, out_dir / "comparison" / f"{stem}_input_n2n_ntn.png", title=str(path), zoom_size=args.zoom_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare noisy input, N2N baseline, and NTN generalized output.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--n2n_checkpoint", type=str, required=True)
    parser.add_argument("--translator_checkpoint", type=str, required=True)
    parser.add_argument("--gaussian_expert_checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/images/n2n_vs_ntn")
    parser.add_argument("--name_prefix", type=str, default="")
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
    parser.add_argument("--max_value", type=float, default=0.0)
    parser.add_argument("--size_multiple", type=int, default=32)
    parser.add_argument("--zoom_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
