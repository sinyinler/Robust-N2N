"""SIDD-Small scene-disjoint 完整图 tiled inference 与逐图评测。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.sidd_dataset import discover_sidd_pairs, load_sidd_pair
from utils.sidd import (
    load_sidd_model,
    paired_bootstrap_ci,
    psnr_rgb,
    save_rgb,
    save_sidd_comparison,
    ssim_rgb,
    tiled_inference,
)


def main(args: argparse.Namespace) -> None:
    scenes = tuple(scene.zfill(3) for scene in args.scenes)
    pairs = discover_sidd_pairs(args.data_root, scenes)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_sidd_model(args.checkpoint, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for pair_index, pair in enumerate(tqdm(pairs, desc="SIDD full-image eval")):
        noisy_u8, gt_u8 = load_sidd_pair(pair)
        denoised = tiled_inference(
            model,
            noisy_u8,
            device,
            tile_size=args.tile_size,
            overlap=args.tile_overlap,
            amp=args.amp,
        )
        noisy = noisy_u8.astype(np.float32) / 255.0
        gt = gt_u8.astype(np.float32) / 255.0
        n_psnr = psnr_rgb(noisy, gt)
        n_ssim = ssim_rgb(noisy, gt)
        o_psnr = psnr_rgb(denoised, gt)
        o_ssim = ssim_rgb(denoised, gt)
        rows.append(
            {
                "name": pair.name,
                "scene": pair.scene,
                "noisy_psnr": n_psnr,
                "noisy_ssim": n_ssim,
                "output_psnr": o_psnr,
                "output_ssim": o_ssim,
                "delta_psnr": o_psnr - n_psnr,
                "delta_ssim": o_ssim - n_ssim,
            }
        )
        save_rgb(out_dir / "images" / f"{pair.name}_output.png", denoised)
        if pair_index < args.max_visuals:
            save_sidd_comparison(
                noisy_u8,
                denoised,
                gt_u8,
                out_dir / "comparisons" / f"{pair.name}.png",
                zoom_size=args.zoom_size,
            )

    delta_psnr = [row["delta_psnr"] for row in rows]
    delta_ssim = [row["delta_ssim"] for row in rows]
    summary = {
        "scenes": scenes,
        "pair_count": len(rows),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "noisy_psnr": float(np.mean([row["noisy_psnr"] for row in rows])),
        "noisy_ssim": float(np.mean([row["noisy_ssim"] for row in rows])),
        "output_psnr": float(np.mean([row["output_psnr"] for row in rows])),
        "output_ssim": float(np.mean([row["output_ssim"] for row in rows])),
        "delta_psnr": float(np.mean(delta_psnr)),
        "delta_ssim": float(np.mean(delta_ssim)),
        "delta_psnr_bootstrap95": paired_bootstrap_ci(delta_psnr),
        "delta_ssim_bootstrap95": paired_bootstrap_ci(delta_ssim),
        "images_won_psnr": int(np.sum(np.asarray(delta_psnr) > 0)),
        "images_won_ssim": int(np.sum(np.asarray(delta_ssim) > 0)),
        "caveat": "CI unit is scene instance; these instances share held-out scene content.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SIDD-Small full RGB images")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenes", nargs="+", default=["008"])
    parser.add_argument("--out_dir", default="results/sidd/internal_test_scene008")
    parser.add_argument("--device", default="")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_overlap", type=int, default=64)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_visuals", type=int, default=4)
    parser.add_argument("--zoom_size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
