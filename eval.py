from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from utils.io import read_image_any
from utils.metrics import center_crop_to_match, psnr, ssim_simple


def normalize_to_u8(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        return np.zeros_like(x, dtype=np.uint8)
    y = np.clip((x.astype(np.float32) - vmin) / (vmax - vmin), 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


def make_comparison(noisy: np.ndarray, denoised: np.ndarray, reference: np.ndarray | None, out_path: Path, zoom_size: int) -> None:
    images = [noisy, denoised] if reference is None else [noisy, denoised, reference]
    if reference is not None:
        noisy, reference = center_crop_to_match(noisy, reference)
        denoised, reference = center_crop_to_match(denoised, reference)
        images = [noisy, denoised, reference]
    else:
        noisy, denoised = center_crop_to_match(noisy, denoised)
        images = [noisy, denoised]

    vmin = float(np.percentile(np.concatenate([x.reshape(-1) for x in images]), 1))
    vmax = float(np.percentile(np.concatenate([x.reshape(-1) for x in images]), 99))
    panels = [Image.fromarray(normalize_to_u8(x, vmin, vmax), mode="L").convert("RGB") for x in images]

    h, w = images[0].shape
    zoom = min(zoom_size, h, w)
    top = max(0, (h - zoom) // 2)
    left = max(0, (w - zoom) // 2)
    zoom_panels = []
    for panel in panels:
        draw = ImageDraw.Draw(panel)
        draw.rectangle([left, top, left + zoom, top + zoom], outline=(255, 0, 0), width=max(1, w // 256))
        crop = panel.crop((left, top, left + zoom, top + zoom)).resize((w, h), Image.Resampling.BICUBIC)
        zoom_panels.append(crop)

    canvas = Image.new("RGB", (w * len(panels), h * 2), color=(0, 0, 0))
    for idx, panel in enumerate(panels):
        canvas.paste(panel, (idx * w, 0))
        canvas.paste(zoom_panels[idx], (idx * w, h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main(args) -> None:
    noisy = read_image_any(args.noisy)
    denoised = read_image_any(args.denoised)
    reference = read_image_any(args.reference) if args.reference else None

    result = {}
    if reference is not None:
        result["psnr_noisy"] = psnr(noisy, reference, data_range=args.data_range if args.data_range > 0 else None)
        result["psnr_denoised"] = psnr(denoised, reference, data_range=args.data_range if args.data_range > 0 else None)
        result["ssim_noisy"] = ssim_simple(noisy, reference, data_range=args.data_range if args.data_range > 0 else None)
        result["ssim_denoised"] = ssim_simple(denoised, reference, data_range=args.data_range if args.data_range > 0 else None)
    result["noisy"] = str(Path(args.noisy).resolve())
    result["denoised"] = str(Path(args.denoised).resolve())
    if args.reference:
        result["reference"] = str(Path(args.reference).resolve())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    make_comparison(noisy, denoised, reference, out_dir / "comparison_with_zoom.png", zoom_size=args.zoom_size)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate denoising with metrics and visual comparison.")
    parser.add_argument("--noisy", type=str, required=True)
    parser.add_argument("--denoised", type=str, required=True)
    parser.add_argument("--reference", type=str, default="", help="Optional clean/pseudo-clean reference.")
    parser.add_argument("--out_dir", type=str, default="results/eval")
    parser.add_argument("--data_range", type=float, default=0.0)
    parser.add_argument("--zoom_size", type=int, default=96)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
