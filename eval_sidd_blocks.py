"""在公开 SIDD Validation sRGB blocks 上评估 noisy 基线或训练 checkpoint。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.io import loadmat, savemat
from tqdm import tqdm

from utils.sidd import load_sidd_model, paired_bootstrap_ci, psnr_rgb, ssim_rgb


def load_blocks(path: str | Path, variable: str) -> np.ndarray:
    payload = loadmat(path)
    if variable not in payload:
        public = sorted(key for key in payload if not key.startswith("__"))
        raise KeyError(f"{path} 缺少变量 {variable}，现有变量: {public}")
    array = np.asarray(payload[variable])
    expected = (40, 32, 256, 256, 3)
    if array.shape != expected:
        raise ValueError(f"{variable} 维度应为 {expected}，实际为 {array.shape}")
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = np.clip(np.rint(array * 255.0), 0, 255).astype(np.uint8)
        else:
            array = np.clip(np.rint(array), 0, 255).astype(np.uint8)
    return array


def save_block_comparison(noisy, output, gt, path: Path) -> None:
    error = np.clip(np.abs(output - gt) * 4.0, 0.0, 1.0)
    arrays = [noisy, output, gt, error]
    labels = ["Noisy", "Output", "GT", "|Output-GT| x4"]
    canvas = Image.new("RGB", (256 * 4, 280), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (array, label) in enumerate(zip(arrays, labels)):
        image = Image.fromarray(np.clip(np.rint(array * 255), 0, 255).astype(np.uint8), "RGB")
        canvas.paste(image, (index * 256, 24))
        draw.text((index * 256 + 6, 5), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main(args: argparse.Namespace) -> None:
    noisy_u8 = load_blocks(args.noisy_mat, "ValidationNoisyBlocksSrgb")
    gt_u8 = load_blocks(args.gt_mat, "ValidationGtBlocksSrgb")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_sidd_model(args.checkpoint, device) if args.checkpoint else None
    denoised_all = np.empty(noisy_u8.shape, dtype=np.float32) if args.save_mat and model else None

    rows: list[dict] = []
    noisy_psnr: list[float] = []
    noisy_ssim: list[float] = []
    model_psnr: list[float] = []
    model_ssim: list[float] = []
    visual_indices = {0, 320, 640, 960, 1279}
    flat_noisy = noisy_u8.reshape(-1, 256, 256, 3)
    flat_gt = gt_u8.reshape(-1, 256, 256, 3)
    batch_size = int(args.batch_size)

    for start in tqdm(range(0, len(flat_noisy), batch_size), desc="SIDD Validation"):
        stop = min(start + batch_size, len(flat_noisy))
        noisy_batch = flat_noisy[start:stop].astype(np.float32) / 255.0
        gt_batch = flat_gt[start:stop].astype(np.float32) / 255.0
        if model is not None:
            tensor = torch.from_numpy(noisy_batch).permute(0, 3, 1, 2).to(device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, enabled=args.amp and device.type == "cuda"
            ):
                output_batch = model(tensor).clamp_(0.0, 1.0)
            output_batch = output_batch.permute(0, 2, 3, 1).float().cpu().numpy()
            if denoised_all is not None:
                denoised_all.reshape(-1, 256, 256, 3)[start:stop] = output_batch
        else:
            output_batch = noisy_batch

        for offset in range(stop - start):
            flat_index = start + offset
            image_index, block_index = divmod(flat_index, 32)
            n_psnr = psnr_rgb(noisy_batch[offset], gt_batch[offset])
            n_ssim = ssim_rgb(noisy_batch[offset], gt_batch[offset])
            o_psnr = psnr_rgb(output_batch[offset], gt_batch[offset])
            o_ssim = ssim_rgb(output_batch[offset], gt_batch[offset])
            noisy_psnr.append(n_psnr)
            noisy_ssim.append(n_ssim)
            model_psnr.append(o_psnr)
            model_ssim.append(o_ssim)
            rows.append(
                {
                    "image": image_index + 1,
                    "block": block_index + 1,
                    "noisy_psnr": n_psnr,
                    "noisy_ssim": n_ssim,
                    "output_psnr": o_psnr,
                    "output_ssim": o_ssim,
                    "delta_psnr": o_psnr - n_psnr,
                    "delta_ssim": o_ssim - n_ssim,
                }
            )
            if flat_index in visual_indices:
                save_block_comparison(
                    noisy_batch[offset], output_batch[offset], gt_batch[offset],
                    out_dir / "visuals" / f"image_{image_index + 1:02d}_block_{block_index + 1:02d}.png",
                )

    delta_psnr = [row["delta_psnr"] for row in rows]
    delta_ssim = [row["delta_ssim"] for row in rows]
    psnr_ci = paired_bootstrap_ci(delta_psnr)
    ssim_ci = paired_bootstrap_ci(delta_ssim)
    summary = {
        "count": len(rows),
        "metric": "RGB [0,1]; PSNR per block; SSIM 11x11 Gaussian sigma=1.5",
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "noisy_psnr": float(np.mean(noisy_psnr)),
        "noisy_ssim": float(np.mean(noisy_ssim)),
        "output_psnr": float(np.mean(model_psnr)),
        "output_ssim": float(np.mean(model_ssim)),
        "delta_psnr": float(np.mean(delta_psnr)),
        "delta_ssim": float(np.mean(delta_ssim)),
        "delta_psnr_bootstrap95": psnr_ci,
        "delta_ssim_bootstrap95": ssim_ci,
        "blocks_won_psnr": int(np.sum(np.asarray(delta_psnr) > 0)),
        "blocks_won_ssim": int(np.sum(np.asarray(delta_ssim) > 0)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "per_block.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if denoised_all is not None:
        savemat(out_dir / "Idenoised.mat", {"Idenoised": denoised_all}, do_compression=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SIDD Validation sRGB blocks")
    parser.add_argument("--noisy_mat", required=True)
    parser.add_argument("--gt_mat", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--out_dir", default="results/sidd/validation_blocks")
    parser.add_argument("--device", default="")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_mat", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
