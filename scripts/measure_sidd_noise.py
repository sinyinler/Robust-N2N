"""用对齐的 SIDD noisy/GT 固定中心 crop 标定 sRGB 真实噪声幅度。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.sidd_dataset import discover_sidd_pairs, load_sidd_pair  # noqa: E402


MAD_TO_SIGMA = 0.6744897501960817


def center_crop(array: np.ndarray, size: int) -> np.ndarray:
    height, width = array.shape[:2]
    if min(height, width) < size:
        raise ValueError(f"图像尺寸 {(height, width)} 小于 crop={size}")
    top = (height - size) // 2
    left = (width - size) // 2
    return array[top : top + size, left : left + size]


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p25": float(np.percentile(array, 25)),
        "p75": float(np.percentile(array, 75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main(args: argparse.Namespace) -> None:
    pairs = discover_sidd_pairs(args.data_root, args.scenes)
    robust_sigmas: list[float] = []
    standard_deviations: list[float] = []
    for pair in tqdm(pairs, desc="Measure SIDD noise"):
        noisy, gt = load_sidd_pair(pair)
        residual = (
            center_crop(noisy, args.crop).astype(np.float32)
            - center_crop(gt, args.crop).astype(np.float32)
        ) / 255.0
        median = float(np.median(residual))
        robust_sigmas.append(float(np.median(np.abs(residual - median)) / MAD_TO_SIGMA))
        standard_deviations.append(float(residual.std()))

    robust = summarize(robust_sigmas)
    standard = summarize(standard_deviations)
    result = {
        "data_root": str(Path(args.data_root).resolve()),
        "scenes": [str(scene).zfill(3) for scene in args.scenes],
        "crop": args.crop,
        "pairs": len(pairs),
        "robust_sigma": robust,
        "std": standard,
        "recommended_feature_sigma": [0.25 * robust["median"], 0.75 * robust["median"]],
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--out", default="results/sidd/noise_calibration.json")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
